#pragma once

#include "proto/common.hpp"
#include "proto/pfss_interval_lut_ext.hpp"
#include "proto/beaver_mul64.hpp"
#include "proto/bit_ring_ops.hpp"
#include "proto/pfss_utils.hpp"
#include <vector>
#include <stdexcept>

namespace proto {

struct GeluIntervalLutPartyKey {
  int d = 3;

  u64 r_in_share = 0;
  u64 r_out_share = 0;

  u64 wrap_sign_share = 0; // additive share of wrap bit
  FssKey dcf_hat_lt_r;
  FssKey dcf_hat_lt_r_plus_2p63;

  FssKey lut_key;  // interval LUT returning (d+1) words

  std::vector<BeaverTriple64Share> triples64;
};

inline GeluOut eval_gelu_interval_lut_one(int party,
                                          const PfssIntervalLutExt& fss,
                                          IChannel& ch,
                                          const GeluIntervalLutPartyKey& K,
                                          u64 hatx_public) {
  BeaverMul64 mul{party, ch, K.triples64, 0};
  BitRingOps B{party, mul};

  u64 x = (party == 0) ? sub_mod(hatx_public, K.r_in_share)
                       : sub_mod(0ull, K.r_in_share);

  u64 a = eval_u64_share_from_dcf(fss, 64, K.dcf_hat_lt_r, hatx_public);
  u64 b = eval_u64_share_from_dcf(fss, 64, K.dcf_hat_lt_r_plus_2p63, hatx_public);
  u64 na = B.NOT(a);
  u64 u = B.AND(b, na);
  u64 wrap_or = B.OR(na, b);
  u64 w = B.SEL(K.wrap_sign_share, u, wrap_or); // wrap ? wrap_or : u

  const u64 TWO63 = (1ull << 63);
  u64 hatx_bias = add_mod(hatx_public, TWO63);

  // Default fallback: use base eval_dcf (bytes-out) if interval API not implemented.
  std::vector<u8> bytes = fss.eval_dcf(64, K.lut_key, fss.u64_to_bits_msb(hatx_bias, 64));
  auto coeff = unpack_u64_vec_le(bytes);
  if ((int)coeff.size() < K.d + 1) throw std::runtime_error("lut coeff size mismatch");

  u64 acc = coeff[K.d];
  for (int k = K.d - 1; k >= 0; k--) acc = add_mod(mul.mul(acc, x), coeff[k]);
  u64 delta = acc;

  u64 xplus = mul.mul(w, x);
  u64 y = add_mod(xplus, delta);
  u64 haty = add_mod(y, K.r_out_share);
  return GeluOut{w, y, haty};
}

}  // namespace proto
