#pragma once

#include "mpc/beaver.hpp"
#include "mpc/net.hpp"

namespace mpc {

// One Beaver-based multiplication on additive shares.
template<typename RingT>
inline AddShare<RingT> mul_share(
    int party,
    ::net::Chan& ch,
    AddShare<RingT> x,
    AddShare<RingT> y,
    BeaverTripleA<RingT> t) {
  RingT d_local = x.s - t.a.s;
  RingT e_local = y.s - t.b.s;

  ch.send_u64(d_local.v);
  uint64_t d_other = ch.recv_u64();
  ch.send_u64(e_local.v);
  uint64_t e_other = ch.recv_u64();

  RingT d = RingT(d_local.v + d_other);
  RingT e = RingT(e_local.v + e_other);

  RingT z = t.c.s + RingT(d.v) * t.b.s + RingT(e.v) * t.a.s;
  if (party == 0) z += RingT(d.v) * RingT(e.v);
  return {z};
}

}  // namespace mpc
