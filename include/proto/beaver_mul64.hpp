#pragma once

#include "proto/common.hpp"
#include "proto/channel.hpp"
#include "proto/beaver.hpp"
#include <vector>
#include <stdexcept>
#include <span>

namespace proto {

struct BeaverMul64 {
  int party;  // 0 or 1
  IChannel& ch;
  const std::vector<BeaverTriple64Share>& triples;
  size_t idx = 0;

  // One multiplication (1 round: open e,f)
  u64 mul(u64 x, u64 y) {
    if (idx >= triples.size()) throw std::runtime_error("BeaverMul64: out of triples");
    const auto& t = triples[idx++];
    u64 e_share = sub_mod(x, t.a);
    u64 f_share = sub_mod(y, t.b);

    u64 other_e = 0, other_f = 0;
    ch.send_bytes(&e_share, sizeof(u64));
    ch.send_bytes(&f_share, sizeof(u64));
    ch.recv_bytes(&other_e, sizeof(u64));
    ch.recv_bytes(&other_f, sizeof(u64));

    u64 e = add_mod(e_share, other_e);
    u64 f = add_mod(f_share, other_f);

    u64 z = t.c;
    z = add_mod(z, mul_mod(e, t.b));
    z = add_mod(z, mul_mod(f, t.a));
    if (party == 0) z = add_mod(z, mul_mod(e, f));
    return z;
  }

  // Batch multiplication (one round regardless of batch size)
  void mul_batch(const std::vector<u64>& x,
                 const std::vector<u64>& y,
                 std::vector<u64>& out) {
    const size_t n = x.size();
    if (y.size() != n) throw std::runtime_error("mul_batch size mismatch");
    if (idx + n > triples.size()) throw std::runtime_error("mul_batch out of triples");
    out.resize(n);

    std::vector<u64> e_share(n), f_share(n), other_e(n), other_f(n);
    for (size_t i = 0; i < n; i++) {
      const auto& t = triples[idx + i];
      e_share[i] = sub_mod(x[i], t.a);
      f_share[i] = sub_mod(y[i], t.b);
    }

    exchange_u64_vec(ch, e_share, other_e);
    exchange_u64_vec(ch, f_share, other_f);

    for (size_t i = 0; i < n; i++) {
      const auto& t = triples[idx + i];
      u64 e = add_mod(e_share[i], other_e[i]);
      u64 f = add_mod(f_share[i], other_f[i]);

      u64 z = t.c;
      z = add_mod(z, mul_mod(e, t.b));
      z = add_mod(z, mul_mod(f, t.a));
      if (party == 0) z = add_mod(z, mul_mod(e, f));
      out[i] = z;
    }
    idx += n;
  }
};

// Batched wrapper API (preferred)
class BeaverMul64Batch {
public:
  BeaverMul64Batch(int party_, IChannel& ch_, std::span<const BeaverTriple64Share> triples_)
      : party(party_), ch(ch_), triples(triples_), idx(0) {}

  void mul(std::span<const u64> x, std::span<const u64> y, std::span<u64> out) {
    if (x.size() != y.size() || out.size() != x.size()) throw std::runtime_error("BeaverMul64Batch size mismatch");
    if (idx + x.size() > triples.size()) throw std::runtime_error("BeaverMul64Batch out of triples");

    std::vector<u64> e_share(x.size()), f_share(x.size()), other_e(x.size()), other_f(x.size());
    for (size_t i = 0; i < x.size(); i++) {
      const auto& t = triples[idx + i];
      e_share[i] = sub_mod(x[i], t.a);
      f_share[i] = sub_mod(y[i], t.b);
    }
    exchange_u64_vec(ch, e_share, other_e);
    exchange_u64_vec(ch, f_share, other_f);

    for (size_t i = 0; i < x.size(); i++) {
      const auto& t = triples[idx + i];
      u64 e = add_mod(e_share[i], other_e[i]);
      u64 f = add_mod(f_share[i], other_f[i]);
      u64 z = t.c;
      z = add_mod(z, mul_mod(e, t.b));
      z = add_mod(z, mul_mod(f, t.a));
      if (party == 0) z = add_mod(z, mul_mod(e, f));
      out[i] = z;
    }
    idx += x.size();
  }

  size_t triples_used() const { return idx; }

private:
  int party;
  IChannel& ch;
  std::span<const BeaverTriple64Share> triples;
  size_t idx;
};

}  // namespace proto
