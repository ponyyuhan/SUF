#pragma once

#include <array>
#include <random>
#include <utility>
#include <vector>
#include "core/ring.hpp"
#include "compiler/suf_collect.hpp"
#include "mpc/beaver.hpp"
#include "mpc/masked_wire.hpp"
#include "mpc/net.hpp"
#include "pfss/backend_cleartext.hpp"
#include "pfss/pfss.hpp"
#include "pfss/program_desc.hpp"

namespace gates {

// Parameters: fractional bits f >= 1
struct ReluARSParams {
  int frac_bits = 1;
};

// Per-party key material for one ReluARS gate instance.
struct ReluARSGateKey {
  ReluARSParams params;

  // mask shares
  mpc::AddShare<core::Z2n<64>> r_in_share;
  mpc::AddShare<core::Z2n<64>> r_hi_share;   // (r >> f)
  mpc::AddShare<core::Z2n<64>> r_out_share;  // output mask
  mpc::AddShare<core::Z2n<64>> rho_share;    // MSB(r) as 0/1 additive share

  // PFSS keys
  pfss::Key pred_lt_r;       // k_lt_r_64
  pfss::Key pred_lt_r_plus;  // k_lt_rplus_64
  pfss::Key pred_lt_rlow;    // k_lt_rlow_f

  // Beaver triples for arithmetic-bit operations
  mpc::BeaverTripleA<core::Z2n<64>> tri_c1c2;   // xor of c1,c2
  mpc::BeaverTripleA<core::Z2n<64>> tri_xorrho; // xor*rho
  mpc::BeaverTripleA<core::Z2n<64>> tri_wbase;  // w * base
};

struct ReluARSGateKeys {
  ReluARSGateKey party0;
  ReluARSGateKey party1;
};

// Helper: arithmetic bit XOR using Beaver triple: a ^ b = a + b - 2ab
inline mpc::AddShare<core::Z2n<64>> bit_xor_arith(
    int party,
    net::Chan& ch,
    mpc::AddShare<core::Z2n<64>> a,
    mpc::AddShare<core::Z2n<64>> b,
    const mpc::BeaverTripleA<core::Z2n<64>>& tri) {
  auto ab = mpc::mul_share(party, ch, a, b, tri);
  core::Z2n<64> two_ab = core::Z2n<64>(ab.s.v + ab.s.v);
  return {core::Z2n<64>(a.s.v + b.s.v - two_ab.v)};
}

// Helper: arithmetic bit NOT: (1 - x) with constant 1 owned by party 0.
inline mpc::AddShare<core::Z2n<64>> bit_not_arith(int party, mpc::AddShare<core::Z2n<64>> x) {
  return {core::Z2n<64>((party == 0 ? 1ull : 0ull) - x.s.v)};
}

// Dealer/offline key generation for one ReluARS gate.
template<typename PredBackend>
inline ReluARSGateKeys dealer_make_reluars_keys(
    PredBackend& pred_backend,
    const pfss::PublicParams& pp_pred,
    const ReluARSParams& params,
    std::mt19937_64& rng) {
  ReluARSGateKeys out;
  int f = params.frac_bits;

  auto sample_mask = [&](uint64_t val, mpc::AddShare<core::Z2n<64>>& s0,
                         mpc::AddShare<core::Z2n<64>>& s1) {
    auto [a, b] = compiler::split_u64(rng, val);
    s0 = {core::Z2n<64>(a)};
    s1 = {core::Z2n<64>(b)};
  };

  uint64_t r = rng();
  uint64_t r_hi = (f == 64) ? 0 : (r >> f);
  uint64_t r_low = (f == 64) ? r : (r & ((uint64_t(1) << f) - 1));
  uint64_t rho = (r >> 63) & 1ull;

  sample_mask(r, out.party0.r_in_share, out.party1.r_in_share);
  sample_mask(r_hi, out.party0.r_hi_share, out.party1.r_hi_share);
  sample_mask(rho, out.party0.rho_share, out.party1.rho_share);

  uint64_t r_out = rng();
  sample_mask(r_out, out.party0.r_out_share, out.party1.r_out_share);

  // Predicates: each key handles one predicate bit.
  auto make_pred_key = [&](uint64_t threshold, int bits) -> std::pair<pfss::Key, pfss::Key> {
    pfss_desc::PredBitDesc desc;
    desc.k_bits = bits;
    if (threshold != 0) {
      desc.ranges.push_back({0ull, threshold});
    } else {
      // empty range => always 0
    }
    pfss::ProgramDesc pd;
    pd.kind = "predicates";
    pd.dealer_only_desc = pfss_desc::serialize_pred_bits({desc});
    return pred_backend.prog_gen(pp_pred, pd);
  };

  auto [k0_r, k1_r] = make_pred_key(r, 64);
  auto [k0_rp, k1_rp] = make_pred_key(r + (uint64_t(1) << 63), 64);
  auto [k0_rl, k1_rl] = make_pred_key(r_low, f);

  out.party0.pred_lt_r = std::move(k0_r);
  out.party1.pred_lt_r = std::move(k1_r);
  out.party0.pred_lt_r_plus = std::move(k0_rp);
  out.party1.pred_lt_r_plus = std::move(k1_rp);
  out.party0.pred_lt_rlow = std::move(k0_rl);
  out.party1.pred_lt_rlow = std::move(k1_rl);

  // Beaver triples
  auto t1 = mpc::dealer_make_tripleA<core::Z2n<64>>(rng);
  auto t2 = mpc::dealer_make_tripleA<core::Z2n<64>>(rng);
  auto t3 = mpc::dealer_make_tripleA<core::Z2n<64>>(rng);
  out.party0.tri_c1c2 = t1.first;
  out.party1.tri_c1c2 = t1.second;
  out.party0.tri_xorrho = t2.first;
  out.party1.tri_xorrho = t2.second;
  out.party0.tri_wbase = t3.first;
  out.party1.tri_wbase = t3.second;

  out.party0.params = params;
  out.party1.params = params;
  return out;
}

// Online evaluation. Returns masked y_share and helper bits (w,t,d) as arithmetic bit shares.
inline void eval_reluars_gate(
    const ReluARSGateKey& k,
    int party,
    net::Chan& ch,
    const pfss::Backend<pfss::PredPayload>& pred_backend,
    const pfss::PublicParams& pp_pred,
    uint64_t x_hat_public,
    mpc::AddShare<core::Z2n<64>>& out_y_hat_share,
    mpc::AddShare<core::Z2n<64>>& out_w,
    mpc::AddShare<core::Z2n<64>>& out_t,
    mpc::AddShare<core::Z2n<64>>& out_d) {
  int f = k.params.frac_bits;
  uint64_t off = (f == 0) ? 0ull : (uint64_t(1) << (f - 1));
  uint64_t z_hat = x_hat_public + off;

  auto eval_single_pred = [&](const pfss::Key& kk, uint64_t input) -> mpc::AddShare<core::Z2n<64>> {
    auto payload = pred_backend.eval(party, pp_pred, kk, input);
    if (payload.empty()) return {core::Z2n<64>(0)};
    return {core::Z2n<64>(payload[0] & 1ull)};
  };

  auto c1 = eval_single_pred(k.pred_lt_r, x_hat_public);
  auto c2 = eval_single_pred(k.pred_lt_r_plus, x_hat_public);
  auto xorb = bit_xor_arith(party, ch, c1, c2, k.tri_c1c2);

  // w = xor + rho - 2*xor*rho
  auto xor_rho = mpc::mul_share(party, ch, xorb, k.rho_share, k.tri_xorrho);
  core::Z2n<64> two_xor_rho(xor_rho.s.v + xor_rho.s.v);
  out_w = {core::Z2n<64>(xorb.s.v + k.rho_share.s.v - two_xor_rho.v)};

  out_d = eval_single_pred(k.pred_lt_r, z_hat);
  uint64_t z_low = (f == 64) ? z_hat : (z_hat & ((uint64_t(1) << f) - 1));
  out_t = eval_single_pred(k.pred_lt_rlow, z_low);

  // base = (z_hat >> f) - r_hi - t + d * 2^(64-f)
  uint64_t z_shift = (f == 0) ? z_hat : (z_hat >> f);
  mpc::AddShare<core::Z2n<64>> base = {core::Z2n<64>((party == 0 ? z_shift : 0ull))};
  base.s -= k.r_hi_share.s;
  base.s -= out_t.s;
  if (f > 0 && f < 64) {
    uint64_t big = uint64_t(1) << (64 - f);
    base.s += core::Z2n<64>(out_d.s.v * big);
  }

  // y = w * base
  auto y_share = mpc::mul_share(party, ch, base, out_w, k.tri_wbase);
  y_share.s += k.r_out_share.s;
  out_y_hat_share = y_share;
}

}  // namespace gates
