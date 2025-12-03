#pragma once

#include "proto/beaver.hpp"
#include "proto/pfss_backend.hpp"
#include <span>

namespace proto {

struct ReluARSParams {
  int f = 12;                 // fractional bits for ARS/trunc
  std::array<u64, 8> delta{}; // correction LUT Δ[w,t,d] placeholder
};

// DCF program handles and mask shares for one party.
struct ReluARSPartyKey {
  u64 r_in_share = 0;
  u64 r_hi_share = 0;    // (r_in >> f) share
  u64 r_out_share = 0;

  u64 wrap_sign_share = 0; // additive share of wrap bit
  bool wrap_half = false;

  FssKey dcf_hat_lt_r;
  FssKey dcf_hat_lt_r_plus_2p63;
  FssKey dcf_low_lt_r_low;
  FssKey dcf_low_lt_r_low_plus1;

  std::vector<BeaverTriple64Share> triples64;
  std::vector<BeaverTripleBitShare> triplesBit;

  ReluARSParams params;
};

struct ReluARSDealerOut {
  ReluARSPartyKey k0;
  ReluARSPartyKey k1;
};

inline constexpr size_t reluars_triples64_needed() { return 12; }

class ReluARSDealer {
public:
  // Dealer generates everything offline.
  static ReluARSDealerOut keygen(const ReluARSParams& p, PfssBackend& fss, BeaverDealer& dealer) {
    ReluARSDealerOut out;
    out.k0.params = p;
    out.k1.params = p;

    // 1) Sample masks
    u64 r_in = dealer.rng.rand_u64();
    u64 r_out = dealer.rng.rand_u64();

    auto [r_in0, r_in1] = dealer.split_add(r_in);
    u64 r_hi = (p.f >= 64) ? 0 : (r_in >> p.f);
    auto [r_hi0, r_hi1] = dealer.split_add(r_hi);
    auto [r_out0, r_out1] = dealer.split_add(r_out);

    out.k0.r_in_share = r_in0;
    out.k1.r_in_share = r_in1;
    out.k0.r_out_share = r_out0;
    out.k1.r_out_share = r_out1;
    out.k0.r_hi_share = r_hi0;
    out.k1.r_hi_share = r_hi1;

    // 2) Program DCFs for helper bits on public hatx.
    const u64 TWO63 = (u64(1) << 63);
    u64 thr1 = r_in;           // r
    u64 thr2 = r_in + TWO63;   // r + 2^63 (wraps automatically)
    bool wrap = (thr2 < thr1);
    u64 wrap0 = dealer.rng.rand_bit();
    if (!wrap) wrap0 = 0;
    u64 wrap1 = wrap ? (1ull - wrap0) : 0ull;
    out.k0.wrap_sign_share = wrap0;
    out.k1.wrap_sign_share = wrap1;

    auto one_byte = std::vector<u8>{1u};

    auto thr1_bits = fss.u64_to_bits_msb(thr1, 64);
    auto thr2_bits = fss.u64_to_bits_msb(thr2, 64);

    auto kp1 = fss.gen_dcf(64, thr1_bits, one_byte);
    auto kp2 = fss.gen_dcf(64, thr2_bits, one_byte);

    out.k0.dcf_hat_lt_r = kp1.k0;
    out.k1.dcf_hat_lt_r = kp1.k1;
    out.k0.dcf_hat_lt_r_plus_2p63 = kp2.k0;
    out.k1.dcf_hat_lt_r_plus_2p63 = kp2.k1;

    // Trunc carry bit: t = 1[(hatx mod 2^f) < (r_in mod 2^f)]
    u64 r_low = mask_low(r_in, p.f);
    auto rlow_bits = fss.u64_to_bits_msb(r_low, p.f);
    auto kpt = fss.gen_dcf(p.f, rlow_bits, one_byte);
    out.k0.dcf_low_lt_r_low = kpt.k0;
    out.k1.dcf_low_lt_r_low = kpt.k1;

    // Placeholder d: equality test on low bits via two comparisons
    u64 r_low_plus1 = (p.f == 64) ? (r_low + 1) : ((r_low + 1) & ((u64(1) << p.f) - 1));
    auto rlow1_bits = fss.u64_to_bits_msb(r_low_plus1, p.f);
    auto kpd = fss.gen_dcf(p.f, rlow1_bits, one_byte);
    out.k0.dcf_low_lt_r_low_plus1 = kpd.k0;
    out.k1.dcf_low_lt_r_low_plus1 = kpd.k1;

    // 3) Beaver triples (conservative counts)
    const int need_triples64 = static_cast<int>(reluars_triples64_needed());
    const int need_triplesBit = 32;

    out.k0.triples64.reserve(need_triples64);
    out.k1.triples64.reserve(need_triples64);
    for (int i = 0; i < need_triples64; i++) {
      auto [t0, t1] = dealer.gen_triple64();
      out.k0.triples64.push_back(t0);
      out.k1.triples64.push_back(t1);
    }

    out.k0.triplesBit.reserve(need_triplesBit);
    out.k1.triplesBit.reserve(need_triplesBit);
    for (int i = 0; i < need_triplesBit; i++) {
      auto [t0, t1] = dealer.gen_triple_bit();
      out.k0.triplesBit.push_back(t0);
      out.k1.triplesBit.push_back(t1);
    }

    return out;
  }
};

// Tape order (per instance, per party):
// [wrap_flag][r_in][r_hi][r_out][k_hat_lt_r][k_hat_lt_r2][k_low_lt_r_low][k_low_lt_r_low+1][triples64 vec]
template<typename TapeW>
inline void reluars_write_tape(const ReluARSPartyKey& k, TapeW& tw) {
  tw.write_u64(k.wrap_sign_share);
  tw.write_u64(k.r_in_share);
  tw.write_u64(k.r_hi_share);
  tw.write_u64(k.r_out_share);
  tw.write_bytes(k.dcf_hat_lt_r.bytes);
  tw.write_bytes(k.dcf_hat_lt_r_plus_2p63.bytes);
  tw.write_bytes(k.dcf_low_lt_r_low.bytes);
  tw.write_bytes(k.dcf_low_lt_r_low_plus1.bytes);
  tw.template write_triple64_vec<BeaverTriple64Share>(
      std::span<const BeaverTriple64Share>(k.triples64.data(), k.triples64.size()));
}

}  // namespace proto
