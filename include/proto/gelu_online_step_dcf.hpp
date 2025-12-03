#pragma once

#include "proto/common.hpp"
#include "proto/pfss_backend.hpp"
#include "proto/beaver_mul64.hpp"
#include "proto/bit_ring_ops.hpp"
#include "proto/pfss_utils.hpp"
#include <vector>
#include <stdexcept>

namespace proto {

struct StepCutVec {
  FssKey dcf_key;               // outputs (d+1) words as bytes
  std::vector<u64> delta_vec;   // public Δ_j for >=cut trick (length d+1)
};

struct GeluStepDCFPartyKey {
  int d = 3;

  u64 r_in_share = 0;
  u64 r_out_share = 0;

  bool wrap_sign = false;
  FssKey dcf_hat_lt_r;
  FssKey dcf_hat_lt_r_plus_2p63;

  std::vector<u64> base_coeff;   // length d+1
  std::vector<StepCutVec> cuts;  // per-cut

  std::vector<BeaverTriple64Share> triples64;
};

struct GeluOut {
  u64 w = 0;
  u64 y_share = 0;
  u64 haty_share = 0;
};

inline GeluOut eval_gelu_step_dcf_one(int party,
                                      const PfssBackend& fss,
                                      IChannel& ch,
                                      const GeluStepDCFPartyKey& K,
                                      u64 hatx_public) {
  BeaverMul64 mul{party, ch, K.triples64, 0};
  BitRingOps B{party, mul};

  u64 x = (party == 0) ? sub_mod(hatx_public, K.r_in_share)
                       : sub_mod(0ull, K.r_in_share);

  u64 a = eval_u64_share_from_dcf(fss, 64, K.dcf_hat_lt_r, hatx_public);
  u64 b = eval_u64_share_from_dcf(fss, 64, K.dcf_hat_lt_r_plus_2p63, hatx_public);
  u64 na = B.NOT(a);
  u64 u = B.AND(b, na);
  u64 w = K.wrap_sign ? B.OR(na, b) : u;

  const u64 TWO63 = (1ull << 63);
  u64 hatx_bias = add_mod(hatx_public, TWO63);

  std::vector<u64> coeff(K.d + 1, 0);
  for (int i = 0; i <= K.d; i++) coeff[i] = (party == 0) ? K.base_coeff[i] : 0ull;

  for (const auto& cut : K.cuts) {
    auto outv = eval_vec_u64_from_dcf(fss, 64, cut.dcf_key, hatx_bias, K.d + 1);
    for (int i = 0; i <= K.d; i++) {
      u64 addc = (party == 0) ? cut.delta_vec[i] : 0ull;
      u64 term = sub_mod(addc, outv[i]);
      coeff[i] = add_mod(coeff[i], term);
    }
  }

  u64 acc = coeff[K.d];
  for (int k = K.d - 1; k >= 0; k--) acc = add_mod(mul.mul(acc, x), coeff[k]);
  u64 delta = acc;

  u64 xplus = mul.mul(w, x);
  u64 y = add_mod(xplus, delta);
  u64 haty = add_mod(y, K.r_out_share);
  return GeluOut{w, y, haty};
}

}  // namespace proto
