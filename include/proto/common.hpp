#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

namespace proto {

using u8 = uint8_t;
using u64 = uint64_t;
using u128 = unsigned __int128;

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

// Little-endian packing helpers
inline std::vector<u8> pack_u64_le(u64 x) {
  std::vector<u8> out(8);
  std::memcpy(out.data(), &x, 8);
  return out;
}

inline u64 unpack_u64_le(const u8* p) {
  u64 x;
  std::memcpy(&x, p, 8);
  return x;
}

inline std::vector<u8> pack_u64_vec_le(const std::vector<u64>& ws) {
  std::vector<u8> out(ws.size() * 8);
  for (size_t i = 0; i < ws.size(); i++) {
    std::memcpy(out.data() + 8 * i, &ws[i], 8);
  }
  return out;
}

inline std::vector<u64> unpack_u64_vec_le(const std::vector<u8>& bytes) {
  if (bytes.size() % 8 != 0) throw std::runtime_error("unpack_u64_vec_le: size not multiple of 8");
  size_t n = bytes.size() / 8;
  std::vector<u64> ws(n);
  for (size_t i = 0; i < n; i++) ws[i] = unpack_u64_le(bytes.data() + 8 * i);
  return ws;
}

}  // namespace proto
