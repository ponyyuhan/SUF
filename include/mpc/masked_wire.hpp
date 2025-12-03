#pragma once

#include "mpc/shares.hpp"
#include "mpc/net.hpp"

namespace mpc {

template<typename RingT>
struct MaskedWire {
  RingT x_hat_public;      // public masked value
  AddShare<RingT> r_share; // each party stores its share of r_in
};

// shares -> masked (Protocol 3.3): requires one broadcast each
template<typename RingT>
inline MaskedWire<RingT> shares_to_masked(
    net::Chan& ch,
    AddShare<RingT> x_share,
    AddShare<RingT> r_share) {
  RingT xh_local = x_share.s + r_share.s;
  ch.send_u64(xh_local.v);
  uint64_t xh_other = ch.recv_u64();
  RingT x_hat = RingT(xh_local.v + xh_other);
  return {x_hat, r_share};
}

// masked -> shares (Protocol 3.4): no communication
template<typename RingT>
inline AddShare<RingT> masked_to_shares(int party, MaskedWire<RingT> w) {
  if (party == 0) return {w.x_hat_public - w.r_share.s};
  return {RingT(0) - w.r_share.s};
}

}  // namespace mpc
