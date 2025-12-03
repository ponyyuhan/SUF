#pragma once

#include <vector>
#include "core/ring.hpp"
#include "mpc/shares.hpp"
#include "mpc/arithmetic_mpc.hpp"
#include "mpc/net.hpp"

namespace suf {

// coeffs[k] corresponds to x^k
template<typename RingT>
struct Poly {
  std::vector<RingT> coeffs;  // size d+1

  int degree() const { return static_cast<int>(coeffs.size()) - 1; }
};

// Horner evaluation on additive shares: consumes (deg) Beaver triples.
template<typename RingT>
inline mpc::AddShare<RingT> eval_poly_horner_shared(
    int party,
    net::Chan& ch,
    const Poly<RingT>& p,
    mpc::AddShare<RingT> x,
    const std::vector<mpc::BeaverTripleA<RingT>>& triples  // >= degree
) {
  if (p.coeffs.empty()) return {RingT(0)};
  // result = a_d
  mpc::AddShare<RingT> acc{p.coeffs.back()};
  for (int i = static_cast<int>(p.coeffs.size()) - 2; i >= 0; --i) {
    acc = mpc::mul_share(party, ch, acc, x, triples[static_cast<size_t>(i)]);
    acc.s += p.coeffs[static_cast<size_t>(i)];
  }
  return acc;
}

}  // namespace suf
