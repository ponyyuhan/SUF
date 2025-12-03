#pragma once

#include "proto/pfss_interval_lut_ext.hpp"
#include <stdexcept>
#include <vector>

namespace proto {

struct PackedLtKeyPair { FssKey k0, k1; };

class SigmaFastBackend : public PfssIntervalLutExt {
public:
  struct Params {
    int lambda_bytes = 16;
    bool xor_bitmask = true; // if true, packed bits are XOR-shares
  };

  explicit SigmaFastBackend(Params p = Params{}) : params_(p) {}

  // Base PfssBackend interface stubs
  DcfKeyPair gen_dcf(int, const std::vector<u8>&, const std::vector<u8>&) override {
    throw std::runtime_error("SigmaFastBackend::gen_dcf: use packed/interval APIs");
  }
  std::vector<u8> eval_dcf(int, const FssKey&, const std::vector<u8>&) const override {
    throw std::runtime_error("SigmaFastBackend::eval_dcf: use packed/interval APIs");
  }
  std::vector<u8> u64_to_bits_msb(u64 x, int in_bits) const override {
    std::vector<u8> bits(in_bits);
    for (int i = 0; i < in_bits; i++) bits[i] = static_cast<u8>((x >> (in_bits - 1 - i)) & 1u);
    return bits;
  }

  // Packed multi-threshold compare (CDPF-style)
  PackedLtKeyPair gen_packed_lt(int in_bits, const std::vector<u64>& thresholds) {
    if (in_bits != 64) throw std::runtime_error("SigmaFastBackend stub supports only 64-bit");
    // Stub: deterministic IDs; party0 holds packed bits, party1 zeros.
    u64 id = next_id_++;
    packed_[id] = thresholds;
    PackedLtKeyPair kp;
    kp.k0.bytes = proto::pack_u64_le(id << 1);
    kp.k1.bytes = proto::pack_u64_le((id << 1) | 1ull);
    return kp;
  }

  // Evaluate packed compare bundle for many inputs: outs_bitmask is [N][out_words] u64
  void eval_packed_lt_many(size_t key_bytes,
                           const uint8_t* keys_flat,
                           const std::vector<u64>& xs_u64,
                           int in_bits,
                           int out_words,
                           u64* outs_bitmask) const {
    if (in_bits != 64) throw std::runtime_error("SigmaFastBackend stub supports only 64-bit");
    if (key_bytes < 8) throw std::runtime_error("SigmaFastBackend: key size too small");
    for (size_t i = 0; i < xs_u64.size(); i++) {
      u64 kid = proto::unpack_u64_le(keys_flat + i * key_bytes);
      u64 id = kid >> 1;
      int party = static_cast<int>(kid & 1ull);
      auto it = packed_.find(id);
      if (it == packed_.end()) throw std::runtime_error("SigmaFastBackend: unknown key");
      const auto& thr = it->second;
      // out_words must cover thresholds bits; pack into u64 words (xor-share)
      std::vector<u64> words(static_cast<size_t>(out_words), 0);
      for (size_t t = 0; t < thr.size(); t++) {
        bool bit = xs_u64[i] < thr[t];
        size_t word_idx = t / 64;
        size_t bit_idx = t % 64;
        if (word_idx >= words.size()) break;
        if (party == 0) {
          if (bit) words[word_idx] |= (uint64_t(1) << bit_idx);
        }
      }
      for (int w = 0; w < out_words; w++) {
        outs_bitmask[i * out_words + w] = words[static_cast<size_t>(w)];
      }
    }
  }

  // Interval LUT (vector payload) API
  IntervalLutKeyPair gen_interval_lut(const IntervalLutDesc& desc) override {
    u64 id = next_id_++;
    intervals_[id] = desc;
    IntervalLutKeyPair kp;
    kp.k0.bytes = proto::pack_u64_le(id << 1);
    kp.k1.bytes = proto::pack_u64_le((id << 1) | 1ull);
    return kp;
  }

  void eval_interval_lut_many_u64(size_t key_bytes,
                                  const uint8_t* keys_flat,
                                  const std::vector<u64>& xs_u64,
                                  int out_words,
                                  u64* outs_flat) const override {
    if (key_bytes < 8) throw std::runtime_error("SigmaFastBackend: key size too small");
    for (size_t i = 0; i < xs_u64.size(); i++) {
      u64 kid = proto::unpack_u64_le(keys_flat + i * key_bytes);
      u64 id = kid >> 1;
      int party = static_cast<int>(kid & 1ull);
      auto it = intervals_.find(id);
      if (it == intervals_.end()) throw std::runtime_error("SigmaFastBackend: unknown interval key");
      const auto& d = it->second;
      std::vector<u64> out(static_cast<size_t>(out_words), 0);
      if (party == 0) {
        size_t idx = d.cutpoints.size() - 1;
        for (size_t j = 0; j + 1 < d.cutpoints.size(); j++) {
          if (xs_u64[i] >= d.cutpoints[j] && xs_u64[i] < d.cutpoints[j + 1]) { idx = j; break; }
        }
        for (int j = 0; j < out_words; j++) out[static_cast<size_t>(j)] = d.payload_flat[idx * d.out_words + j];
      }
      for (int j = 0; j < out_words; j++) outs_flat[i * out_words + j] = out[static_cast<size_t>(j)];
    }
  }

private:
  Params params_;
  mutable u64 next_id_ = 1;
  mutable std::unordered_map<u64, std::vector<u64>> packed_;
  mutable std::unordered_map<u64, IntervalLutDesc> intervals_;
};

}  // namespace proto
