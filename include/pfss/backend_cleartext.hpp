#pragma once

#include <cstring>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>
#include "pfss/pfss.hpp"
#include "pfss/program_desc.hpp"

namespace pfss {

// Payload types for cleartext demo
using PredPayload = std::vector<uint64_t>;   // packed bits (XOR shares)
using CoeffPayload = std::vector<uint64_t>;  // coeff words (additive shares)

inline uint64_t read_u64_key(const Key& k, size_t idx) {
  if (k.bytes.size() < (idx + 1) * sizeof(uint64_t)) return 0;
  uint64_t v = 0;
  std::memcpy(&v, k.bytes.data() + idx * sizeof(uint64_t), sizeof(uint64_t));
  return v;
}

inline Key make_key_blob(uint64_t id, uint64_t seed) {
  Key k;
  k.bytes.resize(sizeof(uint64_t) * 2);
  std::memcpy(k.bytes.data(), &id, sizeof(uint64_t));
  std::memcpy(k.bytes.data() + sizeof(uint64_t), &seed, sizeof(uint64_t));
  return k;
}

struct CleartextBackendPred final : Backend<PredPayload> {
  struct Stored {
    std::vector<pfss_desc::PredBitDesc> bits;
    uint64_t seed;
  };
  mutable std::unordered_map<uint64_t, Stored> programs;  // key_id -> desc
  uint64_t next_id = 1;

  PublicParams setup(int lambda_bits) override { return {"CLEAR_PRED", lambda_bits}; }

  std::pair<Key, Key> prog_gen(const PublicParams&, const ProgramDesc& desc) override {
    if (!desc.kind.empty() && desc.kind != "predicates") {
      throw std::runtime_error("CleartextBackendPred: unexpected program kind");
    }
    auto bits = pfss_desc::deserialize_pred_bits(desc.dealer_only_desc);
    uint64_t id = next_id++;
    std::random_device rd;
    uint64_t seed = (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());
    programs.emplace(id, Stored{std::move(bits), seed});
    Key k0 = make_key_blob(id, seed);
    Key k1 = k0;
    return {k0, k1};
  }

  PredPayload eval(int party,
                   const PublicParams&,
                   const Key& key,
                   uint64_t x_hat) const override {
    uint64_t id = read_u64_key(key, 0);
    uint64_t seed = read_u64_key(key, 1);
    auto it = programs.find(id);
    if (it == programs.end()) return {};
    const auto& bits = it->second.bits;

    // Compute predicate bits in clear.
    std::vector<uint64_t> packed((bits.size() + 63) / 64, 0);
    for (size_t idx = 0; idx < bits.size(); ++idx) {
      const auto& b = bits[idx];
      uint64_t mask = (b.k_bits == 64) ? ~0ull : ((1ull << b.k_bits) - 1);
      uint64_t v = x_hat & mask;
      bool in = false;
      for (auto [L, U] : b.ranges) {
        if (L <= U) {
          if (v >= L && v < U) { in = true; break; }
        } else {
          if (v >= L || v < U) { in = true; break; }
        }
      }
      if (in) packed[idx / 64] |= (uint64_t(1) << (idx % 64));
    }

    // Deterministic pseudo-random share for party 0, XOR with payload for party 1.
    std::mt19937_64 prng(seed ^ x_hat);
    PredPayload out(packed.size(), 0);
    for (size_t i = 0; i < packed.size(); ++i) {
      uint64_t r = prng();
      if (party == 0) out[i] = r;
      else out[i] = packed[i] ^ r;
    }
    return out;
  }
};

struct CleartextBackendCoeff final : Backend<CoeffPayload> {
  struct Stored {
    pfss_desc::PiecewiseVectorDesc desc;
    uint64_t seed;
  };
  mutable std::unordered_map<uint64_t, Stored> programs;
  uint64_t next_id = 1;

  PublicParams setup(int lambda_bits) override { return {"CLEAR_COEFF", lambda_bits}; }

  std::pair<Key, Key> prog_gen(const PublicParams&, const ProgramDesc& desc) override {
    if (!desc.kind.empty() && desc.kind != "coeff_lut") {
      throw std::runtime_error("CleartextBackendCoeff: unexpected program kind");
    }
    auto pw = pfss_desc::deserialize_piecewise(desc.dealer_only_desc);
    uint64_t id = next_id++;
    std::random_device rd;
    uint64_t seed = (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());
    programs.emplace(id, Stored{std::move(pw), seed});
    Key k0 = make_key_blob(id, seed);
    Key k1 = k0;
    return {k0, k1};
  }

  CoeffPayload eval(int party,
                    const PublicParams&,
                    const Key& key,
                    uint64_t x_hat) const override {
    uint64_t id = read_u64_key(key, 0);
    uint64_t seed = read_u64_key(key, 1);
    auto it = programs.find(id);
    if (it == programs.end()) return {};
    const auto& desc = it->second.desc;

    // Locate piece.
    uint64_t v = x_hat;
    const std::vector<uint64_t>* payload_ptr = nullptr;
    for (const auto& p : desc.pieces) {
      if (p.L <= p.U) {
        if (v >= p.L && v < p.U) { payload_ptr = &p.payload; break; }
      } else {
        if (v >= p.L || v < p.U) { payload_ptr = &p.payload; break; }
      }
    }
    if (!payload_ptr) return {};
    const auto& payload = *payload_ptr;

    std::mt19937_64 prng(seed ^ x_hat);
    CoeffPayload out(payload.size(), 0);
    for (size_t i = 0; i < payload.size(); ++i) {
      uint64_t r = prng();
      if (party == 0) out[i] = r;
      else out[i] = payload[i] - r;
    }
    return out;
  }
};

}  // namespace pfss
