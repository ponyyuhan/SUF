#pragma once

#include "proto/common.hpp"
#include "proto/pfss_interval_lut_ext.hpp"
#include <cstring>
#include <stdexcept>

namespace proto {

// Deterministic reference backend: predicates return XOR-shared 1-byte payload,
// coefficients return additive payload (party0 holds payload, party1 zero).
class ReferenceBackend : public PfssIntervalLutExt {
public:
  std::vector<u8> u64_to_bits_msb(u64 x, int in_bits) const override {
    std::vector<u8> bits(in_bits);
    for (int i = 0; i < in_bits; i++) {
      int shift = (in_bits - 1 - i);
      bits[i] = static_cast<u8>((x >> shift) & 1u);
    }
    return bits;
  }

  DcfKeyPair gen_dcf(int in_bits,
                     const std::vector<u8>& alpha_bits,
                     const std::vector<u8>& payload_bytes) override {
    if (in_bits <= 0 || in_bits > 64) throw std::runtime_error("reference_backend: bad in_bits");
    if (static_cast<int>(alpha_bits.size()) != in_bits) throw std::runtime_error("reference_backend: alpha size mismatch");
    if (payload_bytes.empty()) throw std::runtime_error("reference_backend: empty payload");
    DcfKeyPair kp;
    kp.k0.bytes = pack_key(in_bits, /*party=*/0, alpha_bits, payload_bytes);
    kp.k1.bytes = pack_key(in_bits, /*party=*/1, alpha_bits, payload_bytes);
    return kp;
  }

  std::vector<u8> eval_dcf(int in_bits,
                           const FssKey& kb,
                           const std::vector<u8>& x_bits) const override {
    int key_bits = 0;
    int party = 0;
    std::vector<u8> alpha_bits;
    std::vector<u8> payload;
    unpack_key(kb.bytes, key_bits, party, alpha_bits, payload);
    if (key_bits != in_bits) throw std::runtime_error("reference_backend: in_bits mismatch");
    if (x_bits.size() != static_cast<size_t>(in_bits)) throw std::runtime_error("reference_backend: x_bits mismatch");
    bool lt = false;
    for (int i = 0; i < in_bits; i++) {
      u8 xb = x_bits[static_cast<size_t>(i)] & 1u;
      u8 ab = alpha_bits[static_cast<size_t>(i)] & 1u;
      if (xb < ab) { lt = true; break; }
      if (xb > ab) { lt = false; break; }
    }
    std::vector<u8> out(payload.size(), 0);
    if (lt) {
      if (payload.size() == 1) {
        // XOR semantics for predicate bits: payload[0] is 0/1
        if (party == 0) out[0] = payload[0] & 1u;
        else out[0] = 0u;
      } else {
        // additive payload words
        if (party == 0) out = payload;
      }
    }
    return out;
  }

  IntervalLutKeyPair gen_interval_lut(const IntervalLutDesc& desc) override {
    if (desc.in_bits <= 0 || desc.in_bits > 64) throw std::runtime_error("reference_backend: bad interval in_bits");
    if (desc.out_words <= 0) throw std::runtime_error("reference_backend: bad interval out_words");
    if (desc.cutpoints.size() < 2) throw std::runtime_error("reference_backend: interval cutpoints too small");
    const size_t intervals = desc.cutpoints.size() - 1;
    if (desc.payload_flat.size() != intervals * static_cast<size_t>(desc.out_words)) {
      throw std::runtime_error("reference_backend: interval payload size mismatch");
    }
    IntervalLutKeyPair kp;
    kp.k0.bytes = pack_interval_key(/*party=*/0, desc);
    // Party 1 doesn't need payload; it always returns zeros.
    kp.k1.bytes = pack_interval_key(/*party=*/1, IntervalLutDesc{desc.in_bits, desc.out_words, desc.cutpoints, {}});
    return kp;
  }

  void eval_interval_lut_many_u64(size_t key_bytes,
                                  const uint8_t* keys_flat,
                                  const std::vector<u64>& xs_u64,
                                  int out_words,
                                  u64* outs_flat) const override {
    if (!keys_flat) throw std::runtime_error("reference_backend: interval keys null");
    if (!outs_flat) throw std::runtime_error("reference_backend: interval outs null");
    if (xs_u64.empty()) return;
    if (key_bytes == 0) throw std::runtime_error("reference_backend: interval key_bytes=0");

    // All keys in `keys_flat` are typically identical (broadcasted by the caller); parse once.
    int key_in_bits = 0;
    int party = 0;
    IntervalLutDesc desc;
    unpack_interval_key(keys_flat, key_bytes, key_in_bits, party, desc);
    if (key_in_bits != desc.in_bits) throw std::runtime_error("reference_backend: interval in_bits mismatch");
    if (out_words != desc.out_words) throw std::runtime_error("reference_backend: interval out_words mismatch");
    if (desc.cutpoints.size() < 2) {
      std::memset(outs_flat, 0, xs_u64.size() * static_cast<size_t>(out_words) * sizeof(u64));
      return;
    }
    const size_t intervals = desc.cutpoints.size() - 1;
    if (party != 0) {
      // Additive shares: party 1 holds zeros.
      std::memset(outs_flat, 0, xs_u64.size() * static_cast<size_t>(out_words) * sizeof(u64));
      return;
    }
    if (desc.payload_flat.size() != intervals * static_cast<size_t>(out_words)) {
      throw std::runtime_error("reference_backend: interval payload missing");
    }

    for (size_t i = 0; i < xs_u64.size(); i++) {
      const u64 x = xs_u64[i];
      size_t idx = 0;
      // Find the first cutpoint > x, then take previous interval.
      // cutpoints are increasing in masked domain (last cutpoint is exclusive upper bound).
      auto it = std::upper_bound(desc.cutpoints.begin(), desc.cutpoints.end(), x);
      if (it == desc.cutpoints.begin()) idx = 0;
      else {
        size_t pos = static_cast<size_t>(it - desc.cutpoints.begin());
        idx = (pos == 0) ? 0 : (pos - 1);
        if (idx >= intervals) idx = intervals - 1;
      }
      for (int j = 0; j < out_words; j++) {
        outs_flat[i * static_cast<size_t>(out_words) + static_cast<size_t>(j)] =
            desc.payload_flat[idx * static_cast<size_t>(out_words) + static_cast<size_t>(j)];
      }
    }
  }

private:
  static std::vector<u8> pack_key(int in_bits, int party,
                                  const std::vector<u8>& alpha_bits,
                                  const std::vector<u8>& payload) {
    std::vector<u8> out;
    out.reserve(2 + alpha_bits.size() + payload.size());
    out.push_back(static_cast<u8>(in_bits));
    out.push_back(static_cast<u8>(party & 1));
    out.insert(out.end(), alpha_bits.begin(), alpha_bits.end());
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
  }

  static void unpack_key(const std::vector<u8>& key,
                         int& in_bits,
                         int& party,
                         std::vector<u8>& alpha_bits,
                         std::vector<u8>& payload) {
    if (key.size() < 2) throw std::runtime_error("reference_backend: key too short");
    in_bits = static_cast<int>(key[0]);
    party = static_cast<int>(key[1] & 1u);
    if (in_bits <= 0 || key.size() < 2u + static_cast<size_t>(in_bits)) {
      throw std::runtime_error("reference_backend: key corrupted");
    }
    alpha_bits.assign(key.begin() + 2, key.begin() + 2 + in_bits);
    payload.assign(key.begin() + 2 + in_bits, key.end());
  }

  static std::vector<u8> pack_interval_key(int party, const IntervalLutDesc& desc) {
    // Format:
    //   "ILUT" (4 bytes)
    //   u8 party
    //   u8 in_bits
    //   u16 reserved
    //   u32 out_words
    //   u32 n_cutpoints
    //   u64 cutpoints[n_cutpoints]
    //   u64 payload_flat[(n_cutpoints-1)*out_words]  (may be empty for party1)
    std::vector<u8> out;
    out.insert(out.end(), {'I', 'L', 'U', 'T'});
    out.push_back(static_cast<u8>(party & 1));
    out.push_back(static_cast<u8>(desc.in_bits));
    out.push_back(0u);
    out.push_back(0u);
    auto append_u32 = [&](uint32_t v) {
      out.push_back(static_cast<u8>(v & 0xFFu));
      out.push_back(static_cast<u8>((v >> 8) & 0xFFu));
      out.push_back(static_cast<u8>((v >> 16) & 0xFFu));
      out.push_back(static_cast<u8>((v >> 24) & 0xFFu));
    };
    append_u32(static_cast<uint32_t>(desc.out_words));
    append_u32(static_cast<uint32_t>(desc.cutpoints.size()));
    for (u64 cp : desc.cutpoints) {
      auto bytes = proto::pack_u64_le(cp);
      out.insert(out.end(), bytes.begin(), bytes.end());
    }
    for (u64 w : desc.payload_flat) {
      auto bytes = proto::pack_u64_le(w);
      out.insert(out.end(), bytes.begin(), bytes.end());
    }
    return out;
  }

  static void unpack_interval_key(const uint8_t* key,
                                  size_t key_bytes,
                                  int& in_bits,
                                  int& party,
                                  IntervalLutDesc& desc) {
    if (key_bytes < 4 + 2 + 2 + 4 + 4) throw std::runtime_error("reference_backend: interval key too short");
    if (!(key[0] == 'I' && key[1] == 'L' && key[2] == 'U' && key[3] == 'T')) {
      throw std::runtime_error("reference_backend: interval key bad magic");
    }
    party = static_cast<int>(key[4] & 1u);
    in_bits = static_cast<int>(key[5]);
    auto read_u32 = [&](size_t off) -> uint32_t {
      return static_cast<uint32_t>(key[off]) |
             (static_cast<uint32_t>(key[off + 1]) << 8) |
             (static_cast<uint32_t>(key[off + 2]) << 16) |
             (static_cast<uint32_t>(key[off + 3]) << 24);
    };
    uint32_t out_words = read_u32(8);
    uint32_t n_cutpoints = read_u32(12);
    const size_t header = 16;
    size_t need_cp = header + static_cast<size_t>(n_cutpoints) * 8;
    if (key_bytes < need_cp) throw std::runtime_error("reference_backend: interval key truncated cutpoints");
    desc.in_bits = in_bits;
    desc.out_words = static_cast<int>(out_words);
    desc.cutpoints.resize(n_cutpoints);
    const uint8_t* p = key + header;
    for (uint32_t i = 0; i < n_cutpoints; i++) {
      desc.cutpoints[i] = proto::unpack_u64_le(p);
      p += 8;
    }
    size_t remaining = key_bytes - need_cp;
    if (remaining == 0) {
      desc.payload_flat.clear();
      return;
    }
    if (remaining % 8 != 0) throw std::runtime_error("reference_backend: interval payload misaligned");
    const size_t n_words = remaining / 8;
    desc.payload_flat.resize(n_words);
    for (size_t i = 0; i < n_words; i++) {
      desc.payload_flat[i] = proto::unpack_u64_le(p);
      p += 8;
    }
  }
};

}  // namespace proto
