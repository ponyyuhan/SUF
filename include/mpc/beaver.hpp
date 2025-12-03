#pragma once

#include "mpc/shares.hpp"
#include <random>
#include <utility>

namespace mpc {

// Arithmetic Beaver triple: a,b uniform; c=a*b in ring. Shared additively.
template<typename RingT>
struct BeaverTripleA {
  AddShare<RingT> a, b, c;
};

// Boolean Beaver triple for AND over GF(2): u,v uniform bits; w=u&v. Shared by XOR.
struct BeaverTripleB {
  XorShare u, v, w;
};

// Dealer-side generation (in preprocessing model).
template<typename RingT, class URBG>
inline std::pair<BeaverTripleA<RingT>, BeaverTripleA<RingT>>
dealer_make_tripleA(URBG& g) {
  using W = typename RingT::word;
  std::uniform_int_distribution<W> dist;
  RingT a(dist(g)), b(dist(g)), c(a * b);

  RingT a0(dist(g)), b0(dist(g)), c0(dist(g));
  RingT a1 = a - a0, b1 = b - b0, c1 = c - c0;

  BeaverTripleA<RingT> t0{{a0},{b0},{c0}};
  BeaverTripleA<RingT> t1{{a1},{b1},{c1}};
  return {t0, t1};
}

template<class URBG>
inline std::pair<BeaverTripleB, BeaverTripleB>
dealer_make_tripleB(URBG& g) {
  std::uniform_int_distribution<int> dist(0, 1);
  uint8_t u = static_cast<uint8_t>(dist(g));
  uint8_t v = static_cast<uint8_t>(dist(g));
  uint8_t w = static_cast<uint8_t>(u & v);

  uint8_t u0 = static_cast<uint8_t>(dist(g));
  uint8_t v0 = static_cast<uint8_t>(dist(g));
  uint8_t w0 = static_cast<uint8_t>(dist(g));

  uint8_t u1 = static_cast<uint8_t>(u ^ u0);
  uint8_t v1 = static_cast<uint8_t>(v ^ v0);
  uint8_t w1 = static_cast<uint8_t>(w ^ w0);

  return {BeaverTripleB{{u0},{v0},{w0}}, BeaverTripleB{{u1},{v1},{w1}}};
}

}  // namespace mpc
