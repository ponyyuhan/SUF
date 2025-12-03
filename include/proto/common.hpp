#pragma once

#include "core/serialization.hpp"
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

namespace proto {

using u8 = uint8_t;
using u64 = uint64_t;
using u128 = unsigned __int128;

using core::pack_u64_le;
using core::pack_u64_vec_le;
using core::unpack_u64_le;
using core::unpack_u64_vec_le;

inline u64 add_mod(u64 a, u64 b) { return a + b; }  // wraps mod 2^64
inline u64 sub_mod(u64 a, u64 b) { return a - b; }  // wraps mod 2^64
inline u64 mul_mod(u64 a, u64 b) { return static_cast<u64>(static_cast<u128>(a) * static_cast<u128>(b)); }

inline u64 rot_add(u64 x, u64 r) { return x + r; }

inline u64 mask_low(u64 x, int f) {
  if (f <= 0) return 0;
  if (f >= 64) return x;
  return x & ((u64(1) << f) - 1);
}

inline u64 one_hot_bit(u8 b) { return static_cast<u64>(b & 1); }

}  // namespace proto
