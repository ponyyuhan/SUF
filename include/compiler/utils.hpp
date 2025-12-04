#pragma once

#include <cstdint>
#include <random>
#include <utility>

namespace compiler {

inline std::pair<uint64_t, uint64_t> split_u64(std::mt19937_64& rng, uint64_t x) {
  uint64_t a = rng();
  uint64_t b = x - a;
  return {a, b};
}

}  // namespace compiler
