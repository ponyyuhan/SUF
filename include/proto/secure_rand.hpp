#pragma once

#include "proto/common.hpp"
#include <random>

namespace proto {

// Simple prototype RNG; replace with OS RNG for production.
struct SecureRand {
  std::random_device rd;

  u64 rand_u64() {
    u64 x = 0;
    for (int i = 0; i < 4; i++) {
      x = (x << 16) ^ static_cast<u64>(rd() & 0xFFFFu);
    }
    return x;
  }

  u8 rand_bit() { return static_cast<u8>(rand_u64() & 1u); }
};

}  // namespace proto
