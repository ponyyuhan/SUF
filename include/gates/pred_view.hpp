#pragma once

#include <cstddef>
#include <cstdint>

namespace gates {

// View into packed XOR predicate masks
struct PredViewPacked {
  const uint64_t* words = nullptr;
  size_t nwords = 0;
  inline uint64_t get(size_t idx) const {
    size_t w = idx >> 6;
    size_t b = idx & 63;
    if (w >= nwords) return 0;
    return (words[w] >> b) & 1ull;
  }
};

// View into flat XOR predicate bits (per-bit storage)
struct PredViewFlat {
  const uint64_t* bits = nullptr;
  inline uint64_t get(size_t idx) const { return (bits ? (bits[idx] & 1ull) : 0ull); }
};

}  // namespace gates
