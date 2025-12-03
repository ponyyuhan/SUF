#pragma once

#include "proto/common.hpp"
#include "proto/pfss_backend.hpp"
#include "proto/beaver_mul64.hpp"
#include "proto/bit_ring_ops.hpp"
#include "proto/pfss_utils.hpp"
#include "proto/reluars_dealer.hpp"
#include <array>
#include <stdexcept>

namespace proto {

struct ReluARSParamsOnline {
  int f = 12;
  std::array<u64, 8> delta = {};  // Δ[idx] where idx=(w<<2)|(t<<1)|d
};

struct ReluARSPartyKeyOnline {
  int f = 12;
  u64 r_in_share = 0;
  u64 r_hi_share = 0;     // share of r_in >> f (dealer-provided)
  u64 r_out_share = 0;

  u64 wrap_sign_share = 0; // additive share of wrap bit

  FssKey dcf_hat_lt_r;
  FssKey dcf_hat_lt_r_plus_2p63;
  FssKey dcf_low_lt_r_low;
  FssKey dcf_low_lt_r_low_plus1;

  ReluARSParamsOnline params;
  std::vector<BeaverTriple64Share> triples64;
};

struct ReluARSOut {
  u64 w = 0, t = 0, d = 0;
  u64 y_share = 0;
  u64 haty_share = 0;
};

inline ReluARSOut eval_reluars_one(int party,
                                  const PfssBackend& fss,
                                  IChannel& ch,
                                  const ReluARSPartyKeyOnline& K,
                                  u64 hatx_public);

// Tape reader helper: reconstruct key then eval.
template<typename TapeR>
inline ReluARSOut eval_reluars_from_tape(int party,
                                         const PfssBackend& fss,
                                         IChannel& ch,
                                         TapeR& tr,
                                         const ReluARSParamsOnline& params,
                                         u64 hatx_public) {
  ReluARSPartyKeyOnline k;
  k.f = params.f;
  k.params = params;
  k.wrap_sign_share = tr.read_u64();
  k.r_in_share = tr.read_u64();
  k.r_hi_share = tr.read_u64();
  k.r_out_share = tr.read_u64();
  k.dcf_hat_lt_r.bytes = tr.read_bytes();
  k.dcf_hat_lt_r_plus_2p63.bytes = tr.read_bytes();
  k.dcf_low_lt_r_low.bytes = tr.read_bytes();
  k.dcf_low_lt_r_low_plus1.bytes = tr.read_bytes();
  k.triples64 = tr.template read_triple64_vec<BeaverTriple64Share>();
  if (!k.triples64.empty() && k.triples64.size() < reluars_triples64_needed()) {
    throw std::runtime_error("reluars_from_tape: insufficient triples");
  }
  return eval_reluars_one(party, fss, ch, k, hatx_public);
}

inline ReluARSOut eval_reluars_one(int party,
                                  const PfssBackend& fss,
                                  IChannel& ch,
                                  const ReluARSPartyKeyOnline& K,
                                  u64 hatx_public) {
  BeaverMul64 mul{party, ch, K.triples64, 0};
  BitRingOps B{party, mul};

  const int f = K.f;
  const u64 off = (f == 0) ? 0ull : (1ull << (f - 1));

  // helper bits from DCFs
  u64 a = eval_u64_share_from_dcf(fss, 64, K.dcf_hat_lt_r, hatx_public);                 // 1[hatx < r]
  u64 b = eval_u64_share_from_dcf(fss, 64, K.dcf_hat_lt_r_plus_2p63, hatx_public);       // 1[hatx < r+2^63]
  u64 na = B.NOT(a);
  u64 u = B.AND(b, na);
  u64 wrap_or = B.OR(na, b);
  u64 w = B.SEL(K.wrap_sign_share, wrap_or, u);

  u64 hatz = hatx_public + off;
  u64 hatz_low = (f == 64) ? hatz : (hatz & ((1ull << f) - 1));
  u64 t = eval_u64_share_from_dcf(fss, f, K.dcf_low_lt_r_low, hatz_low);
  u64 s = eval_u64_share_from_dcf(fss, f, K.dcf_low_lt_r_low_plus1, hatz_low);
  u64 d = B.AND(s, B.NOT(t));

  // truncation core
  u64 H = (f == 64) ? 0 : (hatz >> f);  // public
  u64 q = sub_mod(sub_mod((party == 0) ? H : 0ull, K.r_hi_share), t);

  // y = w * q
  u64 y = mul.mul(w, q);

  // correction LUT
  u64 delta = lut8_select(B, w, t, d, K.params.delta);
  y = add_mod(y, delta);

  u64 haty = add_mod(y, K.r_out_share);
  return ReluARSOut{w, t, d, y, haty};
}

}  // namespace proto
