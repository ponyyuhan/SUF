#pragma once

#include "proto/common.hpp"
#include "proto/secure_rand.hpp"
#include <utility>

namespace proto {

struct BeaverTriple64Share {
  u64 a, b, c;  // additive share in Z_2^64
};
struct BeaverTripleBitShare {
  u8 a, b, c;  // XOR share in GF(2)
};

struct BeaverDealer {
  SecureRand rng;

  // Sample (a,b,c=a*b) in Z_2^64, split into 2 additive shares.
  std::pair<BeaverTriple64Share, BeaverTriple64Share> gen_triple64() {
    u64 a = rng.rand_u64();
    u64 b = rng.rand_u64();
    u64 c = mul_mod(a, b);

    u64 a0 = rng.rand_u64();
    u64 a1 = sub_mod(a, a0);
    u64 b0 = rng.rand_u64();
    u64 b1 = sub_mod(b, b0);
    u64 c0 = rng.rand_u64();
    u64 c1 = sub_mod(c, c0);

    return {BeaverTriple64Share{a0, b0, c0}, BeaverTriple64Share{a1, b1, c1}};
  }

  // Sample (a,b,c=a&b) in GF(2), split into 2 XOR shares.
  std::pair<BeaverTripleBitShare, BeaverTripleBitShare> gen_triple_bit() {
    u8 a = rng.rand_bit();
    u8 b = rng.rand_bit();
    u8 c = static_cast<u8>(a & b);

    u8 a0 = rng.rand_bit();
    u8 a1 = static_cast<u8>(a ^ a0);
    u8 b0 = rng.rand_bit();
    u8 b1 = static_cast<u8>(b ^ b0);
    u8 c0 = rng.rand_bit();
    u8 c1 = static_cast<u8>(c ^ c0);

    return {BeaverTripleBitShare{a0, b0, c0}, BeaverTripleBitShare{a1, b1, c1}};
  }

  // Split a ring element into 2 additive shares
  std::pair<u64, u64> split_add(u64 x) {
    u64 x0 = rng.rand_u64();
    u64 x1 = sub_mod(x, x0);
    return {x0, x1};
  }

  // Split a bit into 2 XOR shares
  std::pair<u8, u8> split_xor(u8 b) {
    u8 b0 = rng.rand_bit();
    u8 b1 = static_cast<u8>(b ^ b0);
    return {b0, b1};
  }
};

}  // namespace proto
