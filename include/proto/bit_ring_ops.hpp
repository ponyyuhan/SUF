#pragma once

#include "proto/common.hpp"
#include "proto/beaver_mul64.hpp"
#include <array>

namespace proto {

// Bit shares are u64 in {0,1} secret-shared additively mod 2^64.
struct BitRingOps {
  int party;
  BeaverMul64& mul;

  inline u64 ONE() const { return (party == 0) ? 1ULL : 0ULL; }  // additive share of 1

  inline u64 NOT(u64 a) const { return sub_mod(ONE(), a); }

  inline u64 AND(u64 a, u64 b) { return mul.mul(a, b); }

  inline u64 XOR(u64 a, u64 b) {
    u64 ab = AND(a, b);
    return sub_mod(add_mod(a, b), add_mod(ab, ab));
  }

  inline u64 OR(u64 a, u64 b) {
    u64 ab = AND(a, b);
    return sub_mod(add_mod(a, b), ab);
  }

  // Select: returns (bit ? x1 : x0) with 1 multiplication
  inline u64 SEL(u64 bit, u64 x0, u64 x1) {
    u64 diff = sub_mod(x1, x0);
    u64 t = mul.mul(bit, diff);
    return add_mod(x0, t);
  }

  inline u64 C(u64 c) const { return (party == 0) ? c : 0ULL; }
};

// 8-entry LUT select using 7 selects; idx = (w<<2)|(t<<1)|d
inline u64 lut8_select(BitRingOps& B,
                       u64 w, u64 t, u64 d,
                       const std::array<u64,8>& table_const) {
  u64 v00 = B.SEL(d, B.C(table_const[0]), B.C(table_const[1])); // w=0,t=0
  u64 v01 = B.SEL(d, B.C(table_const[2]), B.C(table_const[3])); // w=0,t=1
  u64 v10 = B.SEL(d, B.C(table_const[4]), B.C(table_const[5])); // w=1,t=0
  u64 v11 = B.SEL(d, B.C(table_const[6]), B.C(table_const[7])); // w=1,t=1

  u64 vw0 = B.SEL(t, v00, v01);
  u64 vw1 = B.SEL(t, v10, v11);

  return B.SEL(w, vw0, vw1);
}

}  // namespace proto
