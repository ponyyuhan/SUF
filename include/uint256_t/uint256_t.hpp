#pragma once

#include <cstdint>
#include <cstring>

// Compatibility shim for libdpf's `#include "uint256_t/uint256_t.hpp"`.
//
// The upstream libdpf repo expects a `uint256_t` type in the global namespace.
// We provide a minimal, trivially-copyable 256-bit unsigned integer that is
// sufficient to compile libdpf in this repo's build (used by grotto adapter).
//
// libdpf also expects a `uint128_t` POD type that supports:
// - brace-init from two 64-bit limbs: `uint128_t{lo, hi}`
// - basic +,-,* (used by its SIMD helpers)
// We implement it as a thin wrapper around `unsigned __int128` so it remains
// trivially copyable and `memcpy`-compatible with 16-byte vectors.
struct uint128_t {
  unsigned __int128 v = 0;

  constexpr uint128_t() = default;
  // Allow implicit construction from small integer literals (libdpf's
  // numeric_limits specializations return `0` in several places).
  constexpr uint128_t(uint64_t x) : v(static_cast<unsigned __int128>(x)) {}
  constexpr explicit uint128_t(unsigned __int128 x) : v(x) {}
  constexpr uint128_t(uint64_t lo, uint64_t hi)
      : v((static_cast<unsigned __int128>(hi) << 64) | static_cast<unsigned __int128>(lo)) {}

  constexpr uint64_t lo() const { return static_cast<uint64_t>(v); }
  constexpr uint64_t hi() const { return static_cast<uint64_t>(v >> 64); }

  constexpr uint128_t& operator+=(uint128_t o) {
    v += o.v;
    return *this;
  }
  constexpr uint128_t& operator-=(uint128_t o) {
    v -= o.v;
    return *this;
  }
  constexpr uint128_t& operator*=(uint128_t o) {
    v *= o.v;
    return *this;
  }

  friend constexpr uint128_t operator+(uint128_t a, uint128_t b) { return uint128_t(a.v + b.v); }
  friend constexpr uint128_t operator-(uint128_t a, uint128_t b) { return uint128_t(a.v - b.v); }
  friend constexpr uint128_t operator*(uint128_t a, uint128_t b) { return uint128_t(a.v * b.v); }
  friend constexpr uint128_t operator~(uint128_t a) { return uint128_t(~a.v); }
  friend constexpr bool operator==(uint128_t a, uint128_t b) { return a.v == b.v; }
  friend constexpr bool operator!=(uint128_t a, uint128_t b) { return a.v != b.v; }
};

// Representation: 4x64-bit limbs, little-endian (w0 is least significant).
struct uint256_t {
  uint64_t w[4]{0, 0, 0, 0};

  constexpr uint256_t() = default;
  constexpr uint256_t(uint64_t v) : w{v, 0, 0, 0} {}

  // libdpf's dpf/utils.hpp numeric_limits specialization expects this ctor
  // signature: uint256_t{ uint128_t{...}, uint128_t{...} }.
  template <typename UInt128T>
  constexpr uint256_t(const UInt128T& hi128, const UInt128T& lo128) {
    uint64_t hi_words[2]{0, 0};
    uint64_t lo_words[2]{0, 0};
    std::memcpy(hi_words, &hi128, sizeof(hi_words));
    std::memcpy(lo_words, &lo128, sizeof(lo_words));
    w[0] = lo_words[0];
    w[1] = lo_words[1];
    w[2] = hi_words[0];
    w[3] = hi_words[1];
  }

  constexpr uint256_t& operator+=(const uint256_t& o) {
    unsigned __int128 c = 0;
    for (int i = 0; i < 4; ++i) {
      unsigned __int128 s = static_cast<unsigned __int128>(w[i]) +
                            static_cast<unsigned __int128>(o.w[i]) + c;
      w[i] = static_cast<uint64_t>(s);
      c = s >> 64;
    }
    return *this;
  }
  constexpr uint256_t& operator-=(const uint256_t& o) {
    unsigned __int128 b = 0;
    for (int i = 0; i < 4; ++i) {
      unsigned __int128 a = static_cast<unsigned __int128>(w[i]);
      unsigned __int128 sub = static_cast<unsigned __int128>(o.w[i]) + b;
      if (a >= sub) {
        w[i] = static_cast<uint64_t>(a - sub);
        b = 0;
      } else {
        w[i] = static_cast<uint64_t>((static_cast<unsigned __int128>(1) << 64) + a - sub);
        b = 1;
      }
    }
    return *this;
  }

  constexpr uint256_t& operator+=(uint64_t v) {
    uint256_t t(v);
    return (*this += t);
  }
  constexpr uint256_t& operator*=(uint64_t m) {
    unsigned __int128 c = 0;
    for (int i = 0; i < 4; ++i) {
      unsigned __int128 p = static_cast<unsigned __int128>(w[i]) * m + c;
      w[i] = static_cast<uint64_t>(p);
      c = p >> 64;
    }
    return *this;
  }

  friend constexpr uint256_t operator+(uint256_t a, const uint256_t& b) { return a += b; }
  friend constexpr uint256_t operator-(uint256_t a, const uint256_t& b) { return a -= b; }
  friend constexpr uint256_t operator~(uint256_t a) {
    for (auto& x : a.w) x = ~x;
    return a;
  }
  friend constexpr uint256_t operator|(uint256_t a, const uint256_t& b) {
    for (int i = 0; i < 4; ++i) a.w[i] |= b.w[i];
    return a;
  }
  friend constexpr uint256_t operator&(uint256_t a, const uint256_t& b) {
    for (int i = 0; i < 4; ++i) a.w[i] &= b.w[i];
    return a;
  }
  friend constexpr uint256_t operator^(uint256_t a, const uint256_t& b) {
    for (int i = 0; i < 4; ++i) a.w[i] ^= b.w[i];
    return a;
  }

  friend constexpr bool operator==(const uint256_t& a, const uint256_t& b) {
    return a.w[0] == b.w[0] && a.w[1] == b.w[1] && a.w[2] == b.w[2] && a.w[3] == b.w[3];
  }
  friend constexpr bool operator!=(const uint256_t& a, const uint256_t& b) { return !(a == b); }
  friend constexpr bool operator<(const uint256_t& a, const uint256_t& b) {
    for (int i = 3; i >= 0; --i) {
      if (a.w[i] < b.w[i]) return true;
      if (a.w[i] > b.w[i]) return false;
    }
    return false;
  }
  friend constexpr bool operator>(const uint256_t& a, const uint256_t& b) { return b < a; }
  friend constexpr bool operator<=(const uint256_t& a, const uint256_t& b) { return !(b < a); }
  friend constexpr bool operator>=(const uint256_t& a, const uint256_t& b) { return !(a < b); }
};
