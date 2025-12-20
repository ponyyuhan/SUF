#pragma once

#include "proto/common.hpp"
#include "proto/channel.hpp"
#include "proto/beaver.hpp"
#include <array>
#include <vector>
#include <stdexcept>
namespace proto {

struct BeaverMul64 {
  int party;  // 0 or 1
  IChannel& ch;
  const std::vector<BeaverTriple64Share>& triples;
  size_t idx = 0;
  // Scratch buffers to avoid per-call allocations in hot batched paths.
  std::vector<u64> ef_share_buf{};
  std::vector<u64> ef_other_buf{};

  // One multiplication (1 round: open e,f)
  u64 mul(u64 x, u64 y) {
    if (idx >= triples.size()) {
      throw std::runtime_error("BeaverMul64: out of triples at " + std::to_string(idx) +
                               " of " + std::to_string(triples.size()));
    }
    const auto& t = triples[idx++];
    u64 e_share = sub_mod(x, t.a);
    u64 f_share = sub_mod(y, t.b);

    std::array<u64, 2> local{e_share, f_share};
    std::array<u64, 2> other{0, 0};
    ch.send_bytes(local.data(), local.size() * sizeof(u64));
    ch.recv_bytes(other.data(), other.size() * sizeof(u64));

    u64 e = add_mod(e_share, other[0]);
    u64 f = add_mod(f_share, other[1]);

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
    if (idx + n > triples.size()) {
      throw std::runtime_error("mul_batch out of triples at " + std::to_string(idx + n) +
                               " of " + std::to_string(triples.size()));
    }
    out.resize(n);

    // Exchange (e,f) in one contiguous message to reduce channel overhead.
    // Use thread-local scratch so repeated short-lived BeaverMul64 instances
    // (common in batched composite evaluation) don't churn allocations.
    thread_local std::vector<u64> ef_share_tls;
    thread_local std::vector<u64> ef_other_tls;
    ef_share_tls.resize(2 * n);
    ef_other_tls.resize(2 * n);
#ifdef _OPENMP
#pragma omp parallel for if (n >= (1ull << 16)) schedule(static)
#endif
    for (long long ii = 0; ii < static_cast<long long>(n); ++ii) {
      const size_t i = static_cast<size_t>(ii);
      const auto& t = triples[idx + i];
      ef_share_tls[i] = sub_mod(x[i], t.a);
      ef_share_tls[n + i] = sub_mod(y[i], t.b);
    }

    ch.send_bytes(ef_share_tls.data(), ef_share_tls.size() * sizeof(u64));
    ch.recv_bytes(ef_other_tls.data(), ef_other_tls.size() * sizeof(u64));

#ifdef _OPENMP
#pragma omp parallel for if (n >= (1ull << 16)) schedule(static)
#endif
    for (long long ii = 0; ii < static_cast<long long>(n); ++ii) {
      const size_t i = static_cast<size_t>(ii);
      const auto& t = triples[idx + i];
      u64 e = add_mod(ef_share_tls[i], ef_other_tls[i]);
      u64 f = add_mod(ef_share_tls[n + i], ef_other_tls[n + i]);

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
  BeaverMul64Batch(int party_, IChannel& ch_, const std::vector<BeaverTriple64Share>& triples_)
      : party(party_), ch(ch_), triples(triples_), idx(0) {}

  void mul(const std::vector<u64>& x, const std::vector<u64>& y, std::vector<u64>& out) {
    if (x.size() != y.size()) throw std::runtime_error("BeaverMul64Batch size mismatch");
    if (idx + x.size() > triples.size()) throw std::runtime_error("BeaverMul64Batch out of triples");

    const size_t n = x.size();
    ef_share_buf_.resize(2 * n);
    ef_other_buf_.resize(2 * n);
    for (size_t i = 0; i < x.size(); i++) {
      const auto& t = triples[idx + i];
      ef_share_buf_[i] = sub_mod(x[i], t.a);
      ef_share_buf_[n + i] = sub_mod(y[i], t.b);
    }
    ch.send_bytes(ef_share_buf_.data(), ef_share_buf_.size() * sizeof(u64));
    ch.recv_bytes(ef_other_buf_.data(), ef_other_buf_.size() * sizeof(u64));

    out.resize(x.size());
    for (size_t i = 0; i < x.size(); i++) {
      const auto& t = triples[idx + i];
      u64 e = add_mod(ef_share_buf_[i], ef_other_buf_[i]);
      u64 f = add_mod(ef_share_buf_[n + i], ef_other_buf_[n + i]);
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
  const std::vector<BeaverTriple64Share>& triples;
  size_t idx;
  std::vector<u64> ef_share_buf_{};
  std::vector<u64> ef_other_buf_{};
};

}  // namespace proto
