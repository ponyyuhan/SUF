#pragma once

#include <cstdint>

namespace suf {

// Placeholder for fixed-point helpers; extend with scaling utilities as needed.
struct FixedParams {
  int n_bits = 64;
  int frac_bits = 0;
};

inline uint64_t scale_to_ring(double x, int frac_bits) {
  double scaled = x * static_cast<double>(uint64_t(1) << frac_bits);
  return static_cast<uint64_t>(scaled);
}

}  // namespace suf
