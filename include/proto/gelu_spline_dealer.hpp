#pragma once

#include "proto/beaver.hpp"
#include "proto/pfss_backend.hpp"
#include <algorithm>
#include <vector>
#include <span>

namespace proto {

struct GeluSplineParams {
  int f = 12;
  int d = 3;         // spline degree
  u64 T = 0;         // clipping bound in fixed-point (positive) as uint64 (interpreted signed)

  // Central spline boundaries in signed space: a0=-T < a1 < ... < am=T
  std::vector<int64_t> a;                 // size m+1

  // For each central interval i in [0..m-1], polynomial coeffs for δ(x): length d+1
  std::vector<std::vector<u64>> coeffs;   // size m, each length (d+1)
};

struct DcfVecProgramPartyKey {
  FssKey dcf_key;
};

struct StepCut {
  u64 start;                 // cutpoint in hatx_bias domain
  std::vector<u64> delta;    // Δ vector (public constant)
  DcfVecProgramPartyKey party0;
  DcfVecProgramPartyKey party1;
};

struct GeluSplinePartyKey {
  u64 r_in_share = 0;
  u64 r_out_share = 0;

  u64 wrap_sign_share = 0; // additive share of wrap bit
  FssKey dcf_hat_lt_r;
  FssKey dcf_hat_lt_r_plus_2p63;

  std::vector<u64> base_coeff;  // length d+1
  std::vector<StepCut> cuts;

  std::vector<BeaverTriple64Share> triples64;
  std::vector<BeaverTripleBitShare> triplesBit;

  GeluSplineParams params;
};

struct GeluSplineDealerOut {
  GeluSplinePartyKey k0;
  GeluSplinePartyKey k1;
};

// Write one instance to tape (step-DCF variant) in deterministic order:
// [wrap_flag][r_in_share][r_out_share][k_hat_lt_r][k_hat_lt_r2][base_coeff vec][num_cuts cut_keys+delta][triples...]
template<typename TapeW>
inline void gelu_stepdcf_write_tape(const GeluSplinePartyKey& k, TapeW& tw) {
  tw.write_u64(k.wrap_sign_share);
  tw.write_u64(k.r_in_share);
  tw.write_u64(k.r_out_share);
  tw.write_bytes(k.dcf_hat_lt_r.bytes);
  tw.write_bytes(k.dcf_hat_lt_r_plus_2p63.bytes);
  tw.write_u64_vec(k.base_coeff);
  tw.write_u64(static_cast<uint64_t>(k.cuts.size()));
  for (const auto& c : k.cuts) {
    tw.write_bytes(c.party0.dcf_key.bytes.empty() ? c.party1.dcf_key.bytes : c.party0.dcf_key.bytes);
    tw.write_u64_vec(c.delta);
  }
  tw.template write_triple64_vec<BeaverTriple64Share>(
      std::span<const BeaverTriple64Share>(k.triples64.data(), k.triples64.size()));
}

namespace gelu_internal {

inline u64 bias(int64_t x) { return static_cast<u64>(x) + (u64(1) << 63); }

// A single non-wrapping segment [start,end) with payload v
struct Segment {
  u64 start;
  u64 end;  // start < end, non-wrapping
  std::vector<u64> v;
};

inline std::vector<Segment> rotate_and_split_segments(
    const std::vector<u64>& boundaries_bias,
    const std::vector<std::vector<u64>>& payloads,
    u64 r_in) {
  if (boundaries_bias.size() < 2) throw std::runtime_error("need >=2 boundaries");
  if (payloads.size() + 1 != boundaries_bias.size()) throw std::runtime_error("payloads/boundaries mismatch");

  std::vector<Segment> segs;
  segs.reserve(payloads.size() * 2);

  for (size_t i = 0; i < payloads.size(); i++) {
    u64 a = boundaries_bias[i];
    u64 b = boundaries_bias[i + 1];

    u64 s = a + r_in;
    u64 e = b + r_in;

    if (s < e) {
      segs.push_back(Segment{s, e, payloads[i]});
    } else if (s > e) {
      segs.push_back(Segment{s, static_cast<u64>(0), payloads[i]});
      segs.push_back(Segment{0, e, payloads[i]});
    } else {
      throw std::runtime_error("degenerate rotated segment");
    }
  }

  std::sort(segs.begin(), segs.end(), [](const Segment& x, const Segment& y) {
    return x.start < y.start;
  });

  return segs;
}

}  // namespace gelu_internal

class GeluSplineDealer {
public:
  static GeluSplineDealerOut keygen(const GeluSplineParams& p, PfssBackend& fss, BeaverDealer& dealer) {
    if (static_cast<int>(p.a.size()) < 2) throw std::runtime_error("need a0..am boundaries");
    if (static_cast<int>(p.coeffs.size()) != static_cast<int>(p.a.size()) - 1)
      throw std::runtime_error("coeffs size mismatch");
    for (auto& cv : p.coeffs) {
      if (static_cast<int>(cv.size()) != p.d + 1) throw std::runtime_error("each coeff vector must be d+1");
    }

    GeluSplineDealerOut out;
    out.k0.params = p;
    out.k1.params = p;

    // 1) masks
    u64 r_in = dealer.rng.rand_u64();
    u64 r_out = dealer.rng.rand_u64();
    auto [r_in0, r_in1] = dealer.split_add(r_in);
    auto [r_out0, r_out1] = dealer.split_add(r_out);

    out.k0.r_in_share = r_in0;
    out.k1.r_in_share = r_in1;
    out.k0.r_out_share = r_out0;
    out.k1.r_out_share = r_out1;

    // sign comparisons for x^+
    const u64 TWO63 = (u64(1) << 63);
    u64 thr1 = r_in;
    u64 thr2 = r_in + TWO63;
    bool wrap = (thr2 < thr1);
    auto [wrap0, wrap1] = dealer.split_add(wrap ? 1ull : 0ull);
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

    // 2) Build piecewise δ(x) coefficient vectors in biased domain.
    std::vector<u64> boundaries;
    boundaries.reserve(p.a.size() + 2);
    boundaries.push_back(0);
    for (size_t i = 0; i < p.a.size(); i++) boundaries.push_back(gelu_internal::bias(p.a[i]));
    for (size_t i = 1; i < boundaries.size(); i++) {
      if (boundaries[i] <= boundaries[i - 1]) throw std::runtime_error("boundaries must be strictly increasing in biased space");
    }
    auto zero_vec = std::vector<u64>(p.d + 1, 0);

    std::vector<std::vector<u64>> payloads;
    payloads.push_back(zero_vec);           // left tail
    for (const auto& cv : p.coeffs) payloads.push_back(cv);  // central
    payloads.push_back(zero_vec);           // right tail
    boundaries.push_back(0);                // conceptual end 2^64
    if (payloads.size() + 1 != boundaries.size()) throw std::runtime_error("payloads/boundaries internal mismatch");

    auto segs = gelu_internal::rotate_and_split_segments(boundaries, payloads, r_in);
    if (segs.empty() || segs[0].start != 0) {
      throw std::runtime_error("expected segment starting at 0 after rotate/split");
    }

    // base v0 = segs[0].v
    const std::vector<u64> v0 = segs[0].v;
    out.k0.base_coeff = v0;
    out.k1.base_coeff = v0;

    out.k0.cuts.clear();
    out.k1.cuts.clear();
    for (size_t j = 1; j < segs.size(); j++) {
      const u64 start = segs[j].start;
      std::vector<u64> delta(v0.size());
      for (size_t t = 0; t < delta.size(); t++) delta[t] = sub_mod(segs[j].v[t], segs[j - 1].v[t]);

      auto start_bits = fss.u64_to_bits_msb(start, 64);
      auto payload_bytes = pack_u64_vec_le(delta);
      auto kp = fss.gen_dcf(64, start_bits, payload_bytes);

      StepCut cut;
      cut.start = start;
      cut.delta = delta;
      cut.party0.dcf_key = kp.k0;
      cut.party1.dcf_key = kp.k1;
      out.k0.cuts.push_back(cut);
      out.k1.cuts.push_back(cut);
      out.k0.cuts.back().party1.dcf_key.bytes.clear();
      out.k1.cuts.back().party0.dcf_key.bytes.clear();
    }

    // 5) Beaver triples
    const int need_triples64 = p.d + 8;
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

}  // namespace proto
