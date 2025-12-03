#pragma once

#include "mpc/beaver.hpp"
#include "mpc/net.hpp"

namespace mpc {

inline XorShare bxor(XorShare a, XorShare b) { return {static_cast<uint8_t>(a.b ^ b.b)}; }
inline XorShare bnot(XorShare a) { return {static_cast<uint8_t>(a.b ^ 1)}; }

// AND via Beaver triple over GF(2) with open(d), open(e)
inline XorShare band_share(
    int party,
    net::Chan& ch,
    XorShare x,
    XorShare y,
    BeaverTripleB t) {
  uint8_t d_local = static_cast<uint8_t>(x.b ^ t.u.b);
  uint8_t e_local = static_cast<uint8_t>(y.b ^ t.v.b);

  ch.send_u64(d_local);
  uint64_t d_other = ch.recv_u64();
  ch.send_u64(e_local);
  uint64_t e_other = ch.recv_u64();

  uint8_t d = static_cast<uint8_t>(d_local ^ static_cast<uint8_t>(d_other));
  uint8_t e = static_cast<uint8_t>(e_local ^ static_cast<uint8_t>(e_other));

  uint8_t z = static_cast<uint8_t>(t.w.b ^ (d & t.v.b) ^ (e & t.u.b));
  if (party == 0) z = static_cast<uint8_t>(z ^ static_cast<uint8_t>(d & e));
  return {z};
}

}  // namespace mpc
