#pragma once

#include "proto/common.hpp"
#include "proto/channel.hpp"
#include "proto/beaver.hpp"
#include <array>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <string>
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
  std::vector<uint8_t> ef_bytes_send{};
  std::vector<uint8_t> ef_bytes_recv{};

  static bool pack_enabled() {
    const char* env = std::getenv("SUF_BEAVER_PACK_EFFBITS");
    if (!env) return true;
    std::string s(env);
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return !(s == "0" || s == "false" || s == "off" || s == "no");
  }

  static inline int bytes_per_val(int ring_bits) {
    if (ring_bits <= 0) return 8;
    if (ring_bits >= 64) return 8;
    return (ring_bits + 7) / 8;
  }

  static inline void pack_words_to_bytes(const u64* in,
                                        size_t elems,
                                        int bytes_each,
                                        u64 mask,
                                        std::vector<uint8_t>& out) {
    out.resize(elems * static_cast<size_t>(bytes_each));
#ifdef _OPENMP
#pragma omp parallel for if (elems >= (1ull << 16)) schedule(static)
#endif
    for (long long ii = 0; ii < static_cast<long long>(elems); ++ii) {
      const size_t i = static_cast<size_t>(ii);
      u64 v = in[i] & mask;
      std::memcpy(out.data() + i * static_cast<size_t>(bytes_each), &v, static_cast<size_t>(bytes_each));
    }
  }

  static inline void unpack_words_from_bytes(const uint8_t* in,
                                            size_t elems,
                                            int bytes_each,
                                            u64 mask,
                                            std::vector<u64>& out) {
    out.resize(elems);
#ifdef _OPENMP
#pragma omp parallel for if (elems >= (1ull << 16)) schedule(static)
#endif
    for (long long ii = 0; ii < static_cast<long long>(elems); ++ii) {
      const size_t i = static_cast<size_t>(ii);
      u64 v = 0;
      std::memcpy(&v, in + i * static_cast<size_t>(bytes_each), static_cast<size_t>(bytes_each));
      out[i] = v & mask;
    }
  }

  // One multiplication (1 round: open e,f)
  u64 mul(u64 x, u64 y) {
    if (idx >= triples.size()) {
      throw std::runtime_error("BeaverMul64: out of triples at " + std::to_string(idx) +
                               " of " + std::to_string(triples.size()));
    }
    const auto& t = triples[idx++];
    u64 e_share = sub_mod(x, t.a);
    u64 f_share = sub_mod(y, t.b);

    const int bits = proto::ring_bits();
    const bool do_pack = pack_enabled() && bits > 0 && bits < 64;
    std::array<u64, 2> other{0, 0};
    if (!do_pack) {
      std::array<u64, 2> local{e_share, f_share};
      ch.send_bytes(local.data(), local.size() * sizeof(u64));
      ch.recv_bytes(other.data(), other.size() * sizeof(u64));
    } else {
      const size_t elems = 2;
      const int b = bytes_per_val(bits);
      const u64 mask = proto::ring_mask();
      u64 tmp[2]{e_share, f_share};
      pack_words_to_bytes(tmp, elems, b, mask, ef_bytes_send);
      ef_bytes_recv.resize(ef_bytes_send.size());
      ch.send_bytes(ef_bytes_send.data(), ef_bytes_send.size());
      ch.recv_bytes(ef_bytes_recv.data(), ef_bytes_recv.size());
      unpack_words_from_bytes(ef_bytes_recv.data(), elems, b, mask, ef_other_buf);
      other[0] = ef_other_buf[0];
      other[1] = ef_other_buf[1];
    }

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
    // Use per-instance scratch buffers; these are safe under OpenMP because each
    // loop iteration writes to disjoint indices.
    ef_share_buf.resize(2 * n);
    ef_other_buf.resize(2 * n);
#ifdef _OPENMP
#pragma omp parallel for if (n >= (1ull << 16)) schedule(static)
#endif
    for (long long ii = 0; ii < static_cast<long long>(n); ++ii) {
      const size_t i = static_cast<size_t>(ii);
      const auto& t = triples[idx + i];
      ef_share_buf[i] = sub_mod(x[i], t.a);
      ef_share_buf[n + i] = sub_mod(y[i], t.b);
    }

    const int bits = proto::ring_bits();
    const bool do_pack = pack_enabled() && bits > 0 && bits < 64;
    if (!do_pack) {
      ch.send_bytes(ef_share_buf.data(), ef_share_buf.size() * sizeof(u64));
      ch.recv_bytes(ef_other_buf.data(), ef_other_buf.size() * sizeof(u64));
    } else {
      const size_t elems = ef_share_buf.size();
      const int b = bytes_per_val(bits);
      const u64 mask = proto::ring_mask();
      pack_words_to_bytes(ef_share_buf.data(), elems, b, mask, ef_bytes_send);
      ef_bytes_recv.resize(ef_bytes_send.size());
      ch.send_bytes(ef_bytes_send.data(), ef_bytes_send.size());
      ch.recv_bytes(ef_bytes_recv.data(), ef_bytes_recv.size());
      unpack_words_from_bytes(ef_bytes_recv.data(), elems, b, mask, ef_other_buf);
    }

#ifdef _OPENMP
#pragma omp parallel for if (n >= (1ull << 16)) schedule(static)
#endif
    for (long long ii = 0; ii < static_cast<long long>(n); ++ii) {
      const size_t i = static_cast<size_t>(ii);
      const auto& t = triples[idx + i];
      u64 e = add_mod(ef_share_buf[i], ef_other_buf[i]);
      u64 f = add_mod(ef_share_buf[n + i], ef_other_buf[n + i]);

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
    const int bits = proto::ring_bits();
    const bool do_pack = BeaverMul64::pack_enabled() && bits > 0 && bits < 64;
    if (!do_pack) {
      ch.send_bytes(ef_share_buf_.data(), ef_share_buf_.size() * sizeof(u64));
      ch.recv_bytes(ef_other_buf_.data(), ef_other_buf_.size() * sizeof(u64));
    } else {
      const size_t elems = ef_share_buf_.size();
      const int b = BeaverMul64::bytes_per_val(bits);
      const u64 mask = proto::ring_mask();
      BeaverMul64::pack_words_to_bytes(ef_share_buf_.data(), elems, b, mask, ef_bytes_send_);
      ef_bytes_recv_.resize(ef_bytes_send_.size());
      ch.send_bytes(ef_bytes_send_.data(), ef_bytes_send_.size());
      ch.recv_bytes(ef_bytes_recv_.data(), ef_bytes_recv_.size());
      BeaverMul64::unpack_words_from_bytes(ef_bytes_recv_.data(), elems, b, mask, ef_other_buf_);
    }

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
  std::vector<uint8_t> ef_bytes_send_{};
  std::vector<uint8_t> ef_bytes_recv_{};
};

}  // namespace proto
