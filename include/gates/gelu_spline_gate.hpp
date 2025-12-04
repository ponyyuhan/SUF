#pragma once

#include <algorithm>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>
#include "core/ring.hpp"
#include "compiler/utils.hpp"
#include "mpc/beaver.hpp"
#include "mpc/masked_wire.hpp"
#include "mpc/net.hpp"
#include "pfss/backend_cleartext.hpp"
#include "pfss/pfss.hpp"
#include "pfss/program_desc.hpp"

namespace gates {

struct GeluSplineParams {
  int frac_bits = 0;      // f
  int poly_deg = 1;       // d <= 3 typical
  uint64_t T = 0;         // clipping bound in scaled units (>=0)
};

struct GeluSplinePiece {
  // polynomial coeffs c0..cd for delta(x) on this interval (additive in Z2^64)
  std::vector<uint64_t> coeffs;
  int64_t start_signed = 0;  // inclusive boundary in signed fixed-point
  int64_t end_signed = 0;    // exclusive boundary in signed fixed-point
};

struct GeluSplineSpec {
  GeluSplineParams params;
  std::vector<int64_t> knots_signed;      // sorted from -T to T
  std::vector<GeluSplinePiece> pieces;    // size == knots.size()-1
};

struct GeluSplineGateKey {
  GeluSplineParams params;
  mpc::AddShare<core::Z2n<64>> r_in_share;
  mpc::AddShare<core::Z2n<64>> r_out_share;
  pfss::Key lut_key;
  int idx_bits = 0;
  std::vector<mpc::BeaverTripleA<core::Z2n<64>>> triples;  // size poly_deg+1 (1 for x_plus, d for delta)
};

struct GeluSplineGateKeys {
  GeluSplineGateKey party0;
  GeluSplineGateKey party1;
};

inline uint64_t to_u64_twos(int64_t x) { return static_cast<uint64_t>(x); }

inline pfss_desc::PiecewiseVectorDesc build_gelu_piecewise_lut(
    const GeluSplineSpec& spec,
    uint64_t r_in,
    int& out_idx_bits) {
  pfss_desc::PiecewiseVectorDesc desc;
  desc.n_bits = 64;

  const auto& p = spec.params;
  const int d = p.poly_deg;
  const uint64_t SIGN = (uint64_t(1) << 63);
  const uint64_t T = p.T;
  const uint64_t NEG_START = uint64_t(0) - T;

  // Define unsigned cutpoints covering: [0,T), [T,SIGN), [SIGN,NEG_START), [NEG_START, 2^64)
  std::vector<uint64_t> cut = {0};
  if (T != 0) cut.push_back(T);
  cut.push_back(SIGN);
  cut.push_back(NEG_START);
  // Ensure strictly increasing modulo 2^64
  std::sort(cut.begin(), cut.end());
  cut.erase(std::unique(cut.begin(), cut.end()), cut.end());
  if (cut.empty() || cut.front() != 0) cut.insert(cut.begin(), 0);

  // Generate payload per interval between consecutive cutpoints (last wraps to 2^64).
  int idx_bits = 0;
  while ((1 << idx_bits) < static_cast<int>(cut.size())) idx_bits++;
  out_idx_bits = idx_bits;
  int out_words = 2 + (d + 1) + 2 + idx_bits;  // x_plus(2) + delta(d+1) + w,c + idx bits

  auto choose_delta_coeffs = [&](uint64_t start_u) -> std::vector<uint64_t> {
    // Pick first matching piece whose interval contains start_u when mapped to signed space.
    for (const auto& pc : spec.pieces) {
      uint64_t L = to_u64_twos(pc.start_signed);
      uint64_t U = to_u64_twos(pc.end_signed);
      bool in = false;
      if (pc.start_signed < 0 && pc.end_signed <= 0) {  // negative interval, wraps
        if (start_u >= L || start_u < U) in = true;
      } else if (pc.start_signed <= 0 && pc.end_signed >= 0) {  // crosses zero
        if (start_u >= L || start_u < U) in = true;
      } else {  // non-negative
        if (start_u >= L && start_u < U) in = true;
      }
      if (in) return pc.coeffs;
    }
    return std::vector<uint64_t>(static_cast<size_t>(d + 1), 0ull);
  };

  for (size_t i = 0; i < cut.size(); ++i) {
    uint64_t L = cut[i];
    uint64_t U = (i + 1 < cut.size()) ? cut[i + 1] : 0ull;  // wrap at end

    bool nonneg = (L < SIGN);
    bool central = (L < T) || (L >= NEG_START);

    uint64_t a0 = 0;
    uint64_t a1 = nonneg ? 1ull : 0ull;

    auto delta = choose_delta_coeffs(L);
    if (delta.size() < static_cast<size_t>(d + 1)) delta.resize(static_cast<size_t>(d + 1), 0ull);

    uint64_t w = nonneg ? 1ull : 0ull;
    uint64_t c = central ? 1ull : 0ull;

    std::vector<uint64_t> bits(static_cast<size_t>(idx_bits), 0);
    int interval_idx = static_cast<int>(i);
    for (int b = 0; b < idx_bits; ++b) {
      bits[static_cast<size_t>(b)] = ((interval_idx >> b) & 1) ? 1ull : 0ull;
    }

    std::vector<uint64_t> payload;
    payload.reserve(static_cast<size_t>(out_words));
    payload.push_back(a0);
    payload.push_back(a1);
    payload.insert(payload.end(), delta.begin(), delta.end());
    payload.push_back(w);
    payload.push_back(c);
    payload.insert(payload.end(), bits.begin(), bits.end());

    // Rotate by r_in and split wrap if needed.
    uint64_t Lr = L + r_in;
    uint64_t Ur = U + r_in;
    bool wrap = (U != 0 && Ur < Lr) || (U == 0);  // last interval wraps
    if (!wrap) {
      desc.pieces.push_back({Lr, Ur, payload});
    } else {
      desc.pieces.push_back({0ull, Ur, payload});
      desc.pieces.push_back({Lr, ~0ull, payload});
    }
  }

  return desc;
}

// Dealer: generate keys and masks.
template<typename CoeffBackend>
inline GeluSplineGateKeys dealer_make_gelu_spline_keys(
    CoeffBackend& coeff_backend,
    const pfss::PublicParams& pp_coeff,
    const GeluSplineSpec& spec,
    std::mt19937_64& rng) {
  GeluSplineGateKeys out;

  uint64_t r_in = rng();
  uint64_t r_out = rng();
  auto mask_pair = compiler::split_u64(rng, r_in);
  out.party0.r_in_share = {core::Z2n<64>(mask_pair.first)};
  out.party1.r_in_share = {core::Z2n<64>(mask_pair.second)};
  auto mask_pair_out = compiler::split_u64(rng, r_out);
  out.party0.r_out_share = {core::Z2n<64>(mask_pair_out.first)};
  out.party1.r_out_share = {core::Z2n<64>(mask_pair_out.second)};

  int idx_bits = 0;
  auto desc = build_gelu_piecewise_lut(spec, r_in, idx_bits);
  pfss::ProgramDesc prog;
  prog.kind = "interval_lut";
  prog.dealer_only_desc = pfss_desc::serialize_piecewise(desc);
  auto [k0, k1] = coeff_backend.prog_gen(pp_coeff, prog);
  out.party0.lut_key = std::move(k0);
  out.party1.lut_key = std::move(k1);

  int d = spec.params.poly_deg;
  // Beaver triples: 1 for x_plus (a1*x) + d for delta Horner.
  for (int i = 0; i < d + 1; ++i) {
    auto pair = mpc::dealer_make_tripleA<core::Z2n<64>>(rng);
    out.party0.triples.push_back(pair.first);
    out.party1.triples.push_back(pair.second);
  }

  out.party0.params = spec.params;
  out.party1.params = spec.params;
  out.party0.idx_bits = idx_bits;
  out.party1.idx_bits = idx_bits;
  return out;
}

// Horner evaluation helper
inline mpc::AddShare<core::Z2n<64>> horner_eval(
    int party,
    net::Chan& ch,
    const std::vector<mpc::AddShare<core::Z2n<64>>>& coeffs,
    mpc::AddShare<core::Z2n<64>> x,
    const std::vector<mpc::BeaverTripleA<core::Z2n<64>>>& triples) {
  if (coeffs.empty()) return {core::Z2n<64>(0)};
  mpc::AddShare<core::Z2n<64>> acc = coeffs.back();
  int d = static_cast<int>(coeffs.size()) - 1;
  for (int i = d - 1; i >= 0; --i) {
    acc = mpc::mul_share(party, ch, acc, x, triples[static_cast<size_t>(d - 1 - i)]);
    acc.s += coeffs[static_cast<size_t>(i)].s;
  }
  return acc;
}

// Online evaluation: y = x_plus + delta; exposes helper bits w,c and idx bits.
inline void eval_gelu_spline_gate(
    const GeluSplineGateKey& k,
    int party,
    net::Chan& ch,
    const pfss::Backend<pfss::CoeffPayload>& coeff_backend,
    const pfss::PublicParams& pp_coeff,
    uint64_t x_hat_public,
    mpc::AddShare<core::Z2n<64>>& out_y_hat_share,
    mpc::AddShare<core::Z2n<64>>& out_w,
    mpc::AddShare<core::Z2n<64>>& out_c,
    std::vector<mpc::AddShare<core::Z2n<64>>>& out_idx_bits) {
  const int d = k.params.poly_deg;
  int idx_bits = k.idx_bits;

  // masked -> shares
  mpc::AddShare<core::Z2n<64>> x_share =
      (party == 0) ? mpc::AddShare<core::Z2n<64>>{core::Z2n<64>(x_hat_public - k.r_in_share.s.v)}
                   : mpc::AddShare<core::Z2n<64>>{core::Z2n<64>(0ull - k.r_in_share.s.v)};

  // Evaluate LUT
  pfss::CoeffPayload payload = coeff_backend.eval(party, pp_coeff, k.lut_key, x_hat_public);
  size_t expected_words = static_cast<size_t>(2 + (d + 1) + 2 + idx_bits);
  if (payload.size() < expected_words) {
    payload.resize(expected_words, 0);
  }

  size_t pos = 0;
  mpc::AddShare<core::Z2n<64>> a0{core::Z2n<64>(payload[pos++])};
  mpc::AddShare<core::Z2n<64>> a1{core::Z2n<64>(payload[pos++])};
  std::vector<mpc::AddShare<core::Z2n<64>>> delta_coeff;
  delta_coeff.reserve(static_cast<size_t>(d + 1));
  for (int i = 0; i <= d; ++i) delta_coeff.push_back({core::Z2n<64>(payload[pos++])});
  out_w = {core::Z2n<64>(payload[pos++])};
  out_c = {core::Z2n<64>(payload[pos++])};

  out_idx_bits.clear();
  for (int i = 0; pos < payload.size() && i < idx_bits; ++i) {
    out_idx_bits.push_back({core::Z2n<64>(payload[pos++])});
  }

  // x_plus = a0 + a1*x
  auto a1x = mpc::mul_share(party, ch, a1, x_share, k.triples[0]);
  mpc::AddShare<core::Z2n<64>> x_plus{core::Z2n<64>(a0.s.v + a1x.s.v)};

  // delta via Horner
  std::vector<mpc::BeaverTripleA<core::Z2n<64>>> tri_delta;
  for (size_t i = 1; i < k.triples.size(); ++i) tri_delta.push_back(k.triples[i]);
  auto delta = (d == 0) ? delta_coeff[0] : horner_eval(party, ch, delta_coeff, x_share, tri_delta);

  mpc::AddShare<core::Z2n<64>> y_share{core::Z2n<64>(x_plus.s.v + delta.s.v)};
  y_share.s += k.r_out_share.s;
  out_y_hat_share = y_share;
}

}  // namespace gates
