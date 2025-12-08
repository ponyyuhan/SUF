#pragma once

#include "proto/pfss_interval_lut_ext.hpp"
#include <unordered_map>
#include <random>
#include <stdexcept>
#include <cstring>
#include <memory>

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
    d.payload0 = payload_bytes;
    d.payload1.assign(payload_bytes.size(), 0u);
    d.key_bytes = static_cast<size_t>(payload_bytes.size());
    dcf_[id] = d;
    DcfKeyPair kp;
    u64 kid0 = (id << 1);
    u64 kid1 = (id << 1) | 1ull;
    kp.k0.bytes = pack_u64_le(kid0);
    kp.k1.bytes = pack_u64_le(kid1);
    return kp;
  }

  std::vector<u8> eval_dcf(int in_bits,
                           const FssKey& kb,
                           const std::vector<u8>& x_bits) const override {
    if (static_cast<int>(x_bits.size()) != in_bits) throw std::runtime_error("eval_dcf: x_bits mismatch");
    auto [id, party] = decode_key(kb);
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
    if (lt) {
      return (party == 0) ? d.payload0 : d.payload1;
    }
    return std::vector<u8>(d.payload0.size(), 0u);
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
    d.payload0 = desc.payload_flat;
    d.payload1.assign(desc.payload_flat.size(), 0u);
    interval_[id] = d;
    IntervalLutKeyPair kp;
    u64 kid0 = (id << 1);
    u64 kid1 = (id << 1) | 1ull;
    kp.k0.bytes = pack_u64_le(kid0);
    kp.k1.bytes = pack_u64_le(kid1);
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

  // Convenience helper used in CUDA-gated tests.
  std::vector<uint8_t> eval_interval_lut(const IntervalLutDesc& desc,
                                         const IntervalLutKeyPair& key,
                                         const std::vector<uint64_t>& xs) const {
    if (xs.empty()) return {};
    std::vector<uint8_t> out(xs.size() * static_cast<size_t>(desc.out_words) * sizeof(uint64_t));
    auto* out_words = reinterpret_cast<uint64_t*>(out.data());
    std::vector<uint8_t> key_bytes(key.k0.bytes);
    // replicate the same key for every x
    for (size_t i = 1; i < xs.size(); i++) {
      key_bytes.insert(key_bytes.end(), key.k0.bytes.begin(), key.k0.bytes.end());
    }
    eval_interval_lut_many_u64(key.k0.bytes.size(), key_bytes.data(), xs, desc.out_words, out_words);
    return out;
  }

  static std::unique_ptr<PfssBackendBatch> make() { return std::make_unique<ClearBackend>(); }

 private:
  struct DcfDesc {
    int in_bits;
    size_t key_bytes;
    std::vector<u8> alpha_bits;
    std::vector<u8> payload0;
    std::vector<u8> payload1;
  };
  struct IntervalDesc {
    IntervalLutDesc desc;
    size_t key_bytes;
    std::vector<u64> payload0;
    std::vector<u64> payload1;
  };

  std::vector<u64> eval_interval_single(const FssKey& kb, u64 x, int out_words) const {
    auto [id, party] = decode_key(kb);
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
      out[j] = (party == 0 ? it->second.payload0 : it->second.payload1)[idx * desc.out_words + j];
    }
    return out;
  }

  std::pair<u64, int> decode_key(const FssKey& kb) const {
    if (kb.bytes.size() < 8) throw std::runtime_error("backend_clear: key truncated");
    u64 kid = unpack_u64_le(kb.bytes.data());
    u64 id = kid >> 1;
    int party = static_cast<int>(kid & 1ull);
    return {id, party};
  }

  mutable std::unordered_map<u64, DcfDesc> dcf_;
  mutable std::unordered_map<u64, IntervalDesc> interval_;
  u64 next_id_ = 1;
  u64 seed_ = 0;
};

}  // namespace proto
