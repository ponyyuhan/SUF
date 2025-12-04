#pragma once

#include "proto/pfss_backend_batch.hpp"
#include <cstring>
#include <stdexcept>

namespace proto {

// Deterministic reference backend: predicates return XOR-shared 1-byte payload,
// coefficients return additive payload (party0 holds payload, party1 zero).
class ReferenceBackend : public PfssBackendBatch {
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
};

}  // namespace proto
