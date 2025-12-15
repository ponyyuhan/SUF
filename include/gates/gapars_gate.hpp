#pragma once

#include "gates/ars_faithful_gate.hpp"

namespace gates {

// GapARS: optimized ARS under a proven "gap" condition (SIGMA-style).
//
// When the cleartext value v lies in:
//   [0, 2^(n-2)) ∪ [2^n - 2^(n-2), 2^n)
// (i.e. as signed, |v| < 2^(n-2)), ARS can be reduced to a logical right shift
// on x = v + 2^(n-2), plus a constant correction. This enables a cheaper
// protocol in the PFSS/Composite setting; here we implement the same semantics
// in the toy MPC harness used by benches/tests.
using GapArsParams = ArsParams;
using GapArsKey = ArsKey;
using GapArsKeys = ArsKeys;

inline GapArsKeys dealer_make_gapars_keys(const GapArsParams& params,
                                          size_t n,
                                          std::mt19937_64& rng) {
  return dealer_make_ars_keys(params, n, rng);
}

inline std::vector<mpc::AddShare<core::Z2n<64>>> eval_gapars(
    const GapArsKey& k,
    int party,
    net::Chan& ch,
    const std::vector<mpc::AddShare<core::Z2n<64>>>& xs) {
  std::vector<mpc::AddShare<core::Z2n<64>>> out(xs.size());
  const int f = k.params.frac_bits;
  const uint64_t bias_in = uint64_t(1) << 62;
  const uint64_t bias_out = (f >= 0 && f <= 62) ? (uint64_t(1) << (62 - f)) : 0ull;
  const uint64_t low_mask = (f <= 0) ? 0ull : ((f >= 64) ? ~uint64_t(0) : ((uint64_t(1) << f) - 1));
  const uint64_t modulus = (f <= 0 || f >= 64) ? 0ull : (uint64_t(1) << (64 - f));
  for (size_t i = 0; i < xs.size(); ++i) {
    // Shift input into the GapLRS domain: x = v + 2^(n-2) with n=64.
    uint64_t x_share = xs[i].s.v;
    if (party == 0) x_share = x_share + bias_in;

    // Open hatx = x + r_in (mod 2^64).
    uint64_t masked = x_share + k.r_share[i];
    uint64_t other = 0;
    if (party == 0) {
      ch.send_u64(masked);
      other = ch.recv_u64();
    } else {
      other = ch.recv_u64();
      ch.send_u64(masked);
    }
    uint64_t hatx = masked + other;

    const uint64_t r = k.r_value[i];
    const uint64_t r_lo = (f <= 0) ? 0ull : (r & low_mask);
    const uint64_t r_hi = (f >= 64) ? 0ull : (r >> f);

    // carry = 1[x0 < r0] where x0 = hatx mod 2^f and r0 = r mod 2^f.
    uint64_t carry = (f <= 0) ? 0ull : (((hatx & low_mask) < r_lo) ? 1ull : 0ull);

    // wrap = 1[hatx < r] using the MSB-to-wrap optimization (valid when x < 2^(n-1)).
    uint64_t msb_hatx = (hatx >> 63) & 1ull;
    uint64_t msb_r = (r >> 63) & 1ull;
    uint64_t wrap = ((msb_hatx == 0) && (msb_r == 1)) ? 1ull : 0ull;

    uint64_t y = (party == 0) ? ((f >= 64) ? 0ull : (hatx >> f)) : 0ull;
    y -= k.r_high_share[i];
    if (party == 0 && carry) y -= 1ull;
    if (party == 0 && modulus != 0 && wrap) y += modulus;
    // Convert back to ARS domain.
    if (party == 0) y -= bias_out;
    y += k.out_mask[i];
    out[i] = {core::Z2n<64>(y)};
  }
  return out;
}

}  // namespace gates
