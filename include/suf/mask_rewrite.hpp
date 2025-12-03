#pragma once

#include <cstdint>

namespace suf {

struct RotCmp64Recipe {
  uint64_t theta0; // r
  uint64_t theta1; // r + beta (mod 2^64)
  uint8_t wrap;    // 0/1 (must be secret-shared in real protocol)
};

inline RotCmp64Recipe rewrite_lt_u64(uint64_t r, uint64_t beta) {
  uint64_t theta0 = r;
  uint64_t theta1 = theta0 + beta;
  uint8_t wrap = (theta1 < theta0) ? 1u : 0u;
  return RotCmp64Recipe{theta0, theta1, wrap};
}

struct RotLowRecipe {
  int f;
  uint64_t theta0; // r_low
  uint64_t theta1; // r_low + gamma (mod 2^f)
  uint8_t wrap;
};

inline RotLowRecipe rewrite_ltlow(uint64_t r, int f, uint64_t gamma) {
  uint64_t mask = (f >= 64) ? ~uint64_t(0) : ((uint64_t(1) << f) - 1);
  uint64_t theta0 = r & mask;
  uint64_t theta1 = (theta0 + gamma) & mask;
  uint8_t wrap = (theta1 < theta0) ? 1u : 0u;
  return RotLowRecipe{f, theta0, theta1, wrap};
}

inline RotCmp64Recipe rewrite_msb_add(uint64_t r, uint64_t c) {
  // Interval length = 2^63 starting at start = r - c
  uint64_t start = r - c;
  uint64_t theta0 = start;
  uint64_t theta1 = theta0 + (uint64_t(1) << 63);
  uint8_t wrap = (theta1 < theta0) ? 1u : 0u;
  return RotCmp64Recipe{theta0, theta1, wrap};
}

}  // namespace suf
