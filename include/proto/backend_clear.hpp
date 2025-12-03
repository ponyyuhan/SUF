#pragma once

#include "proto/pfss_interval_lut_ext.hpp"
#include <unordered_map>
#include <random>
#include <stdexcept>

namespace proto {

// Cleartext backend implementing PfssIntervalLutExt for testing.
class ClearBackend final : public PfssIntervalLutExt {
public:
  ClearBackend() {
    std::random_device rd;
    seed_ = (static_cast<u64>(rd()) << 32) ^ static_cast<u64>(rd());
  }

  DcfKeyPair gen_dcf(int in_bits,
                     const std::vector<u8>& alpha_bits,
                     const std::vector<u8>& payload_bytes) override {
    u64 id = next_id_++;
    DcfDesc d;
    d.in_bits = in_bits;
    d.alpha_bits = alpha_bits;
    // Deterministic sharing: party0 gets payload, party1 gets zeros.
    d.payload = payload_bytes;
    d.key_bytes = static_cast<size_t>(payload_bytes.size());
    dcf_[id] = d;
    DcfKeyPair kp;
    kp.k0.bytes = pack_u64_le(id);
    kp.k1.bytes = pack_u64_le(id);
    return kp;
  }

  std::vector<u8> eval_dcf(int in_bits,
                           const FssKey& kb,
                           const std::vector<u8>& x_bits) const override {
    if (static_cast<int>(x_bits.size()) != in_bits) throw std::runtime_error("eval_dcf: x_bits mismatch");
    u64 id = unpack_u64_le(kb.bytes.data());
    auto it = dcf_.find(id);
    if (it == dcf_.end()) throw std::runtime_error("eval_dcf: unknown key id");
    const auto& d = it->second;
    if (d.in_bits != in_bits) throw std::runtime_error("eval_dcf: in_bits mismatch");

    bool lt = false;
    for (int i = 0; i < in_bits; i++) {
      u8 xb = x_bits[static_cast<size_t>(i)] & 1u;
      u8 ab = d.alpha_bits[static_cast<size_t>(i)] & 1u;
      if (xb < ab) { lt = true; break; }
      if (xb > ab) { lt = false; break; }
    }
    if (lt) return d.payload;
    return std::vector<u8>(d.payload.size(), 0u);
  }

  std::vector<u8> u64_to_bits_msb(u64 x, int in_bits) const override {
    std::vector<u8> bits(in_bits);
    for (int i = 0; i < in_bits; i++) {
      int shift = (in_bits - 1 - i);
      bits[i] = static_cast<u8>((x >> shift) & 1u);
    }
    return bits;
  }

  IntervalLutKeyPair gen_interval_lut(const IntervalLutDesc& desc) override {
    u64 id = next_id_++;
    IntervalDesc d;
    d.desc = desc;
    d.key_bytes = sizeof(u64);
    interval_[id] = d;
    IntervalLutKeyPair kp;
    kp.k0.bytes = pack_u64_le(id);
    kp.k1.bytes = pack_u64_le(id);
    return kp;
  }

  void eval_interval_lut_many_u64(size_t key_bytes,
                                  const uint8_t* keys_flat,
                                  const std::vector<u64>& xs_u64,
                                  int out_words,
                                  u64* outs_flat) const override {
    for (size_t i = 0; i < xs_u64.size(); i++) {
      FssKey k;
      k.bytes.assign(keys_flat + i * key_bytes, keys_flat + (i + 1) * key_bytes);
      auto res = eval_interval_single(k, xs_u64[i], out_words);
      for (int j = 0; j < out_words; j++) {
        outs_flat[i * out_words + j] = res[j];
      }
    }
  }

private:
  struct DcfDesc {
    int in_bits;
    size_t key_bytes;
    std::vector<u8> alpha_bits;
    std::vector<u8> payload;
  };
  struct IntervalDesc {
    IntervalLutDesc desc;
    size_t key_bytes;
  };

  std::vector<u64> eval_interval_single(const FssKey& kb, u64 x, int out_words) const {
    u64 id = unpack_u64_le(kb.bytes.data());
    auto it = interval_.find(id);
    if (it == interval_.end()) throw std::runtime_error("eval_interval: unknown key");
    const auto& desc = it->second.desc;
    const auto& cp = desc.cutpoints;
    size_t intervals = cp.size();
    size_t idx = intervals - 1;
    for (size_t i = 0; i + 1 < cp.size(); i++) {
      if (x >= cp[i] && x < cp[i + 1]) { idx = i; break; }
    }
    if (idx >= intervals) idx = intervals - 1;
    std::vector<u64> out(static_cast<size_t>(out_words), 0);
    for (int j = 0; j < out_words; j++) {
      out[j] = desc.payload_flat[idx * desc.out_words + j];
    }
    return out;
  }

  mutable std::unordered_map<u64, DcfDesc> dcf_;
  mutable std::unordered_map<u64, IntervalDesc> interval_;
  u64 next_id_ = 1;
  u64 seed_ = 0;
};

}  // namespace proto
