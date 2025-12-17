#pragma once

#include "core/serialization.hpp"
#include <array>
#include <cstdint>
#include <limits>
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

namespace detail {
// Runtime-selected ring bitwidth for arithmetic operations.
// Default: 64-bit ring (mask = all-ones).
inline int g_ring_bits = 64;
inline u64 g_ring_mask = ~u64(0);
}  // namespace detail

inline void set_ring_bits(int bits) {
  int b = bits;
  if (b < 1) b = 1;
  if (b > 64) b = 64;
  detail::g_ring_bits = b;
  if (b == 64) {
    detail::g_ring_mask = ~u64(0);
  } else {
    detail::g_ring_mask = (u64(1) << b) - 1;
  }
}

inline int ring_bits() { return detail::g_ring_bits; }
inline u64 ring_mask() { return detail::g_ring_mask; }

inline u64 norm_mod(u64 x) { return x & detail::g_ring_mask; }

inline int64_t to_signed(u64 v) {
  const int bits = detail::g_ring_bits;
  if (bits == 64) return static_cast<int64_t>(v);
  const u64 mask = detail::g_ring_mask;
  v &= mask;
  const u64 sign_bit = u64(1) << (bits - 1);
  if (v & sign_bit) v |= ~mask;
  return static_cast<int64_t>(v);
}

inline u64 from_signed(int64_t v) { return norm_mod(static_cast<u64>(v)); }

inline u64 add_mod(u64 a, u64 b) { return norm_mod(a + b); }
inline u64 sub_mod(u64 a, u64 b) { return norm_mod(a - b); }
inline u64 mul_mod(u64 a, u64 b) {
  return norm_mod(static_cast<u64>(static_cast<u128>(a) * static_cast<u128>(b)));
}

inline u64 rot_add(u64 x, u64 r) { return add_mod(x, r); }

inline u64 mask_low(u64 x, int f) {
  if (f <= 0) return 0;
  if (f >= 64) return x;
  return x & ((u64(1) << f) - 1);
}

inline u64 one_hot_bit(u8 b) { return static_cast<u64>(b & 1); }

}  // namespace proto
