#pragma once

#include "proto/pfss_interval_lut_ext.hpp"
#include <unordered_map>
#include <random>
#include <stdexcept>
#include <array>
#include <cstring>

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
    d.payload_share = split_payload_bytes(payload_bytes, id);
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
    if (lt) return d.payload_share[static_cast<size_t>(party)];
    return std::vector<u8>(d.payload_share[static_cast<size_t>(party)].size(), 0u);
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
    d.payload_share = split_payload_words(desc.payload_flat, id);
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

private:
  struct DcfDesc {
    int in_bits;
    size_t key_bytes;
    std::vector<u8> alpha_bits;
    std::array<std::vector<u8>, 2> payload_share;
  };
  struct IntervalDesc {
    IntervalLutDesc desc;
    size_t key_bytes;
    std::array<std::vector<u64>, 2> payload_share;
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
      out[j] = it->second.payload_share[static_cast<size_t>(party)][idx * desc.out_words + j];
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

  std::array<std::vector<u8>, 2> split_payload_bytes(const std::vector<u8>& payload, u64 salt) const {
    std::array<std::vector<u8>, 2> s;
    s[0].resize(payload.size());
    s[1].resize(payload.size());
    std::mt19937_64 rng_local(seed_ ^ (salt + 0xBEEF));
    if (payload.size() % 8 == 0) {
      for (size_t off = 0; off < payload.size(); off += 8) {
        u64 w = unpack_u64_le(payload.data() + off);
        u64 s1 = rng_local();
        u64 s0 = sub_mod(w, s1);
        std::memcpy(s[0].data() + off, &s0, 8);
        std::memcpy(s[1].data() + off, &s1, 8);
      }
    } else {
      for (size_t i = 0; i < payload.size(); i++) {
        uint8_t r = static_cast<uint8_t>(rng_local() & 0xFFu);
        s[1][i] = r;
        s[0][i] = static_cast<uint8_t>(payload[i] - r);
      }
    }
    return s;
  }

  std::array<std::vector<u64>, 2> split_payload_words(const std::vector<u64>& payload, u64 salt) const {
    std::array<std::vector<u64>, 2> s;
    s[0].resize(payload.size());
    s[1].resize(payload.size());
    std::mt19937_64 rng_local(seed_ ^ (salt + 0x1234));
    for (size_t i = 0; i < payload.size(); i++) {
      u64 r = rng_local();
      s[1][i] = r;
      s[0][i] = sub_mod(payload[i], r);
    }
    return s;
  }

  mutable std::unordered_map<u64, DcfDesc> dcf_;
  mutable std::unordered_map<u64, IntervalDesc> interval_;
  u64 next_id_ = 1;
  u64 seed_ = 0;
};

}  // namespace proto
