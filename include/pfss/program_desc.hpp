#pragma once

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>
#include "core/serialization.hpp"

namespace pfss_desc {

// A “predicate bit” is membership in a rotated interval, in either full n bits or low f bits.
struct PredBitDesc {
  int k_bits;  // n or f
  // up to two [L,U) ranges
  std::vector<std::pair<uint64_t, uint64_t>> ranges;
};

// Piecewise constant vector payload: disjoint intervals in Z_{2^n}
struct PiecewiseVectorDesc {
  int n_bits;
  struct Piece {
    uint64_t L;
    uint64_t U;
    std::vector<uint64_t> payload;
  };
  std::vector<Piece> pieces;  // must partition domain (or cover all with default)
};

// --- Simple serialization helpers (little-endian, forward compatible) ---

inline std::vector<uint8_t> serialize_pred_bits(const std::vector<PredBitDesc>& bits) {
  std::vector<uint8_t> out;
  core::append_u32(out, static_cast<uint32_t>(bits.size()));
  for (const auto& b : bits) {
    core::append_u32(out, static_cast<uint32_t>(b.k_bits));
    core::append_u32(out, static_cast<uint32_t>(b.ranges.size()));
    for (auto [L, U] : b.ranges) {
      core::append_u64(out, L);
      core::append_u64(out, U);
    }
  }
  return out;
}

inline std::vector<PredBitDesc> deserialize_pred_bits(const std::vector<uint8_t>& bytes) {
  size_t off = 0;
  uint32_t count = core::read_u32(bytes, off);
  std::vector<PredBitDesc> out;
  out.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    int k_bits = static_cast<int>(core::read_u32(bytes, off));
    uint32_t rcount = core::read_u32(bytes, off);
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    ranges.reserve(rcount);
    for (uint32_t j = 0; j < rcount; ++j) {
      uint64_t L = core::read_u64(bytes, off);
      uint64_t U = core::read_u64(bytes, off);
      ranges.push_back({L, U});
    }
    out.push_back({k_bits, std::move(ranges)});
  }
  if (off != bytes.size()) throw std::runtime_error("deserialize_pred_bits: trailing bytes");
  return out;
}

inline std::vector<uint8_t> serialize_piecewise(const PiecewiseVectorDesc& desc) {
  std::vector<uint8_t> out;
  core::append_u32(out, static_cast<uint32_t>(desc.n_bits));
  core::append_u32(out, static_cast<uint32_t>(desc.pieces.size()));
  for (const auto& p : desc.pieces) {
    core::append_u64(out, p.L);
    core::append_u64(out, p.U);
    core::append_u32(out, static_cast<uint32_t>(p.payload.size()));
    for (auto v : p.payload) core::append_u64(out, v);
  }
  return out;
}

inline PiecewiseVectorDesc deserialize_piecewise(const std::vector<uint8_t>& bytes) {
  size_t off = 0;
  PiecewiseVectorDesc desc;
  desc.n_bits = static_cast<int>(core::read_u32(bytes, off));
  uint32_t count = core::read_u32(bytes, off);
  desc.pieces.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    uint64_t L = core::read_u64(bytes, off);
    uint64_t U = core::read_u64(bytes, off);
    uint32_t plen = core::read_u32(bytes, off);
    std::vector<uint64_t> payload;
    payload.reserve(plen);
    for (uint32_t j = 0; j < plen; ++j) payload.push_back(core::read_u64(bytes, off));
    desc.pieces.push_back({L, U, std::move(payload)});
  }
  if (off != bytes.size()) throw std::runtime_error("deserialize_piecewise: trailing bytes");
  return desc;
}

}  // namespace pfss_desc
