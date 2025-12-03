#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace compiler {

// Represents a subset of Z_{2^k} as 1 or 2 half-open intervals [L,U).
struct RotInterval {
  int k_bits;
  std::vector<std::pair<uint64_t, uint64_t>> ranges;  // each is [L,U) in [0,2^k)

  // Evaluate in clear (for testing only!)
  bool contains(uint64_t x) const {
    uint64_t mask = (k_bits == 64) ? ~0ull : ((1ull << k_bits) - 1);
    x &= mask;
    for (auto [L, U] : ranges) {
      if (L <= U) {
        if (L <= x && x < U) return true;
      } else {
        if (x >= L || x < U) return true;
      }
    }
    return false;
  }
};

// Image of [0,beta) under +r mod 2^k
inline RotInterval rotate_prefix(int k_bits, uint64_t r, uint64_t beta) {
  uint64_t mod = (k_bits == 64) ? 0 : (1ull << k_bits);
  auto norm = [&](uint64_t x) { return (k_bits == 64) ? x : (x % mod); };

  RotInterval out{k_bits, {}};
  r = norm(r);
  beta = norm(beta);

  uint64_t end = norm(r + beta);
  if (beta == 0) return out;

  if (k_bits == 64) {
    uint64_t r_plus_beta = r + beta;
    bool wrap = (r_plus_beta < r);
    if (!wrap) out.ranges.push_back({r, r_plus_beta});
    else {
      out.ranges.push_back({0, r_plus_beta});
      out.ranges.push_back({r, ~0ull});
    }
    return out;
  }

  if (r + beta < mod) out.ranges.push_back({r, r + beta});
  else {
    out.ranges.push_back({0, (r + beta) - mod});
    out.ranges.push_back({r, mod});
  }
  (void)end;  // silence unused in non-64-bit branch
  return out;
}

// Rotate low-bit comparison: u = x mod 2^f < gamma, with s = (u + delta) mod 2^f
inline RotInterval rotate_lowbits(int f_bits, uint64_t delta, uint64_t gamma) {
  return rotate_prefix(f_bits, delta, gamma);
}

}  // namespace compiler
