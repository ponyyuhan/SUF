#pragma once

#include <algorithm>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>
#include "core/ring.hpp"
#include "compiler/utils.hpp"
#include "mpc/arithmetic_mpc.hpp"
#include "mpc/beaver.hpp"
#include "mpc/net.hpp"
#include "pfss/backend_cleartext.hpp"
#include "pfss/pfss.hpp"
#include "pfss/program_desc.hpp"

namespace gates {

struct CoeffPack {
  int64_t offset = 0;                // shift for x (in the same fixed scale as input)
  std::vector<int64_t> coeffs;       // Horner order, c0 + c1*t + ...
};

struct PiecewiseInterval {
  uint64_t start = 0;  // inclusive, unsigned domain, non-wrapping
  uint64_t end = 0;    // exclusive, non-wrapping
  CoeffPack pack;
};

struct PiecewisePolySpec {
  int frac_bits_in = 0;
  int frac_bits_out = 0;
  std::vector<PiecewiseInterval> intervals;  // must partition [0,2^64)
};

struct PiecewisePolyKey {
  PiecewisePolySpec spec;
  pfss::Key coeff_key;
  mpc::AddShare<core::Z2n<64>> r_in_share;
  mpc::AddShare<core::Z2n<64>> r_out_share;
  int degree = 0;
  std::vector<mpc::BeaverTripleA<core::Z2n<64>>> triples;
};

struct PiecewisePolyKeys {
  PiecewisePolyKey party0;
  PiecewisePolyKey party1;
};

inline uint64_t to_u64_twos(int64_t x) { return static_cast<uint64_t>(x); }
inline int64_t from_u64_twos(uint64_t x) { return static_cast<int64_t>(x); }

inline int64_t arith_shift_signed(int64_t v, int bits) {
  if (bits <= 0) return v;
  return v >> bits;
}

inline uint64_t arith_shift_u64(uint64_t v, int bits) {
  return static_cast<uint64_t>(arith_shift_signed(static_cast<int64_t>(v), bits));
}

inline int64_t mul_rescale_ring(int64_t a, int64_t b, int frac_bits) {
  core::Z2n<64> prod = core::Z2n<64>(static_cast<uint64_t>(a)) * core::Z2n<64>(static_cast<uint64_t>(b));
  return arith_shift_signed(static_cast<int64_t>(prod.v), frac_bits);
}

inline mpc::AddShare<core::Z2n<64>> mul_rescale_share(
    int party,
    net::Chan& ch,
    mpc::AddShare<core::Z2n<64>> a,
    mpc::AddShare<core::Z2n<64>> b,
    const mpc::BeaverTripleA<core::Z2n<64>>& t,
    int frac_bits) {
  auto prod = mpc::mul_share(party, ch, a, b, t);
  return {core::Z2n<64>(arith_shift_signed(static_cast<int64_t>(prod.s.v), frac_bits))};
}

inline mpc::AddShare<core::Z2n<64>> horner_fixed(
    int party,
    net::Chan& ch,
    const std::vector<mpc::AddShare<core::Z2n<64>>>& coeffs,
    mpc::AddShare<core::Z2n<64>> x,
    int frac_bits,
    const std::vector<mpc::BeaverTripleA<core::Z2n<64>>>& triples) {
  if (coeffs.empty()) return {core::Z2n<64>(0)};
  int d = static_cast<int>(coeffs.size()) - 1;
  if (d < 0) return coeffs[0];
  mpc::AddShare<core::Z2n<64>> acc = coeffs.back();
  for (int i = d - 1; i >= 0; --i) {
    size_t tri_idx = static_cast<size_t>(d - 1 - i);
    if (tri_idx >= triples.size()) break;
    acc = mul_rescale_share(party, ch, acc, x, triples[tri_idx], frac_bits);
    acc.s += coeffs[static_cast<size_t>(i)].s;
  }
  return acc;
}

inline pfss_desc::PiecewiseVectorDesc build_piecewise_poly_desc(
    const PiecewisePolySpec& spec,
    uint64_t r_in,
    int degree,
    int& out_words) {
  pfss_desc::PiecewiseVectorDesc desc;
  desc.n_bits = 64;
  out_words = 1 + (degree + 1);  // offset + coeffs

  for (const auto& iv : spec.intervals) {
    if (iv.start == iv.end) continue;
    uint64_t Lr = iv.start + r_in;
    uint64_t Ur = iv.end + r_in;
    bool wrap = (iv.end == 0) || (Ur < Lr);

    std::vector<uint64_t> payload;
    payload.reserve(static_cast<size_t>(out_words));
    payload.push_back(static_cast<uint64_t>(iv.pack.offset));

    std::vector<int64_t> coeffs = iv.pack.coeffs;
    if (static_cast<int>(coeffs.size()) < degree + 1) {
      coeffs.resize(static_cast<size_t>(degree + 1), 0);
    }
    for (auto c : coeffs) payload.push_back(static_cast<uint64_t>(c));
    if (static_cast<int>(payload.size()) < out_words) {
      payload.resize(static_cast<size_t>(out_words), 0);
    }

    if (!wrap) {
      desc.pieces.push_back({Lr, Ur, payload});
    } else {
      desc.pieces.push_back({0ull, Ur, payload});
      desc.pieces.push_back({Lr, ~0ull, payload});
    }
  }

  return desc;
}

template<typename CoeffBackend>
inline PiecewisePolyKeys dealer_make_piecewise_poly_keys(
    CoeffBackend& coeff_backend,
    const pfss::PublicParams& pp_coeff,
    const PiecewisePolySpec& spec,
    std::mt19937_64& rng) {
  PiecewisePolyKeys out;

  int degree = 0;
  for (const auto& iv : spec.intervals) {
    degree = std::max(degree, static_cast<int>(iv.pack.coeffs.size()) - 1);
  }

  uint64_t r_in = 0;
  uint64_t r_out = rng();
  auto [rin0, rin1] = compiler::split_u64(rng, r_in);
  auto [rout0, rout1] = compiler::split_u64(rng, r_out);
  out.party0.r_in_share = {core::Z2n<64>(rin0)};
  out.party1.r_in_share = {core::Z2n<64>(rin1)};
  out.party0.r_out_share = {core::Z2n<64>(rout0)};
  out.party1.r_out_share = {core::Z2n<64>(rout1)};

  int payload_words = 0;
  auto desc = build_piecewise_poly_desc(spec, r_in, degree, payload_words);
  pfss::ProgramDesc prog;
  prog.kind = "interval_lut";
  prog.dealer_only_desc = pfss_desc::serialize_piecewise(desc);
  auto [k0, k1] = coeff_backend.prog_gen(pp_coeff, prog);
  out.party0.coeff_key = std::move(k0);
  out.party1.coeff_key = std::move(k1);

  for (int i = 0; i < degree; ++i) {
    auto tri_pair = mpc::dealer_make_tripleA<core::Z2n<64>>(rng);
    out.party0.triples.push_back(tri_pair.first);
    out.party1.triples.push_back(tri_pair.second);
  }

  out.party0.spec = spec;
  out.party1.spec = spec;
  out.party0.degree = degree;
  out.party1.degree = degree;
  return out;
}

inline mpc::AddShare<core::Z2n<64>> eval_piecewise_poly(
    const PiecewisePolyKey& k,
    int party,
    net::Chan& ch,
    const pfss::Backend<pfss::CoeffPayload>& coeff_backend,
    const pfss::PublicParams& pp_coeff,
    uint64_t x_hat_public) {
  mpc::AddShare<core::Z2n<64>> x_share =
      (party == 0) ? mpc::AddShare<core::Z2n<64>>{core::Z2n<64>(x_hat_public - k.r_in_share.s.v)}
                   : mpc::AddShare<core::Z2n<64>>{core::Z2n<64>(0ull - k.r_in_share.s.v)};

  auto payload = coeff_backend.eval(party, pp_coeff, k.coeff_key, x_hat_public);
  int expected = 1 + (k.degree + 1);
  if (static_cast<int>(payload.size()) < expected) payload.resize(static_cast<size_t>(expected), 0);

  size_t pos = 0;
  mpc::AddShare<core::Z2n<64>> offset_share{core::Z2n<64>(payload[pos++])};
  std::vector<mpc::AddShare<core::Z2n<64>>> coeffs;
  coeffs.reserve(static_cast<size_t>(k.degree + 1));
  for (int i = 0; i < k.degree + 1; ++i) {
    coeffs.push_back({core::Z2n<64>(payload[pos++])});
  }

  mpc::AddShare<core::Z2n<64>> x_centered{x_share};
  x_centered.s -= offset_share.s;

  auto y_share = (k.degree == 0)
      ? coeffs[0]
      : horner_fixed(party, ch, coeffs, x_centered, k.spec.frac_bits_in, k.triples);

  int delta = k.spec.frac_bits_out - k.spec.frac_bits_in;
  if (delta < 0) {
    y_share.s = core::Z2n<64>(arith_shift_signed(static_cast<int64_t>(y_share.s.v), -delta));
  } else if (delta > 0) {
    y_share.s = core::Z2n<64>(y_share.s.v << delta);
  }

  y_share.s += k.r_out_share.s;
  return y_share;
}

inline int64_t eval_piecewise_poly_ref(
    const PiecewisePolySpec& spec,
    int64_t x_fixed) {
  uint64_t x_u = to_u64_twos(x_fixed);
  const CoeffPack* pack = nullptr;
  for (const auto& iv : spec.intervals) {
    bool in = false;
    if (iv.start <= iv.end) {
      in = (x_u >= iv.start && x_u < iv.end);
    } else {
      in = (x_u >= iv.start || x_u < iv.end);
    }
    if (in) { pack = &iv.pack; break; }
  }
  if (!pack) return 0;

  int d = static_cast<int>(pack->coeffs.size()) - 1;
  int64_t t = x_fixed - pack->offset;
  if (d < 0) return 0;
  core::Z2n<64> acc(pack->coeffs.back());
  for (int i = d - 1; i >= 0; --i) {
    core::Z2n<64> prod = acc * core::Z2n<64>(static_cast<uint64_t>(t));
    int64_t shifted = arith_shift_signed(static_cast<int64_t>(prod.v), spec.frac_bits_in);
    acc = core::Z2n<64>(static_cast<uint64_t>(shifted + pack->coeffs[static_cast<size_t>(i)]));
  }
  int delta = spec.frac_bits_out - spec.frac_bits_in;
  int64_t acc_signed = static_cast<int64_t>(acc.v);
  if (delta < 0) acc_signed = arith_shift_signed(acc_signed, -delta);
  else if (delta > 0) acc_signed <<= delta;
  return acc_signed;
}

inline void append_interval_signed(
    PiecewisePolySpec& spec,
    int64_t start_signed,
    int64_t end_signed,
    const CoeffPack& pack) {
  uint64_t L = to_u64_twos(start_signed);
  uint64_t U = to_u64_twos(end_signed);
  auto push_if = [&](uint64_t a, uint64_t b) {
    if (a != b) spec.intervals.push_back({a, b, pack});
  };
  if (L < U) push_if(L, U);
  else {
    push_if(0ull, U);
    push_if(L, ~0ull);
  }
}

}  // namespace gates
