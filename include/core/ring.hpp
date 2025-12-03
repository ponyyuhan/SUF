#pragma once

#include <cstdint>
#include <type_traits>

namespace core {

// Ring Z_{2^n}. If Bits=64, uint64 overflow already wraps correctly.
template<int Bits>
struct Z2n {
  static_assert(Bits >= 1 && Bits <= 64, "Bits must be in [1,64]");
  using word = uint64_t;

  word v;

  static constexpr word mask() {
    if constexpr (Bits == 64) {
      return ~word(0);
    } else {
      return (word(1) << Bits) - 1;
    }
  }

  static inline word norm(word x) {
    if constexpr (Bits == 64) return x;
    return x & mask();
  }

  Z2n() : v(0) {}
  explicit Z2n(word x) : v(norm(x)) {}

  friend inline Z2n operator+(Z2n a, Z2n b) { return Z2n(norm(a.v + b.v)); }
  friend inline Z2n operator-(Z2n a, Z2n b) { return Z2n(norm(a.v - b.v)); }
  friend inline Z2n operator*(Z2n a, Z2n b) { return Z2n(norm(a.v * b.v)); }

  Z2n& operator+=(Z2n o) { v = norm(v + o.v); return *this; }
  Z2n& operator-=(Z2n o) { v = norm(v - o.v); return *this; }
  Z2n& operator*=(Z2n o) { v = norm(v * o.v); return *this; }
};

}  // namespace core
