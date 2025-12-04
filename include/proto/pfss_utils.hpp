#pragma once

#include "proto/common.hpp"
#include "proto/pfss_backend.hpp"
#include "proto/beaver_mul64.hpp"
#include <stdexcept>
#include <cstring>

namespace proto {

inline u64 eval_u64_share_from_dcf(const PfssBackend& fss, int in_bits,
                                  const FssKey& kb, u64 x) {
  auto out = fss.eval_dcf(in_bits, kb, fss.u64_to_bits_msb(x, in_bits));
  if (out.size() < 8) return 0;
  u64 val = 0;
  std::memcpy(&val, out.data(), std::min<size_t>(8, out.size()));
  return val;
}

inline std::vector<u64> eval_vec_u64_from_dcf(const PfssBackend& fss, int in_bits,
                                              const FssKey& kb, u64 x, int out_words) {
  auto out = fss.eval_dcf(in_bits, kb, fss.u64_to_bits_msb(x, in_bits));
  std::vector<u64> v(static_cast<size_t>(out_words), 0);
  size_t want = static_cast<size_t>(out_words) * 8;
  size_t copy = std::min(want, out.size());
  std::memcpy(v.data(), out.data(), copy);
  return v;
}

inline uint64_t eval_bit_share_from_dcf(const PfssBackend& fss, int in_bits,
                                        const FssKey& kb, u64 x) {
  auto out = fss.eval_dcf(in_bits, kb, fss.u64_to_bits_msb(x, in_bits));
  if (out.empty()) return 0;
  return static_cast<uint64_t>(out[0] & 1u);
}

// Evaluate predicate share under explicit semantics; return an XOR-domain bit share.
inline uint64_t eval_pred_bit_share(const PfssBackend& fss,
                                    int in_bits,
                                    const PredKeyMeta& meta,
                                    const FssKey& kb,
                                    u64 x) {
  auto out = fss.eval_dcf(in_bits, kb, fss.u64_to_bits_msb(x, in_bits));
  if (meta.sem == ShareSemantics::XorBytes) {
    if (out.size() < static_cast<size_t>(meta.out_bytes)) {
      throw std::runtime_error("eval_pred_bit_share: payload truncated");
    }
    if (out.empty()) return 0;
    return static_cast<uint64_t>(out[0] & 1u);
  }
  if (out.size() < 8) throw std::runtime_error("eval_pred_bit_share: additive payload too short");
  uint64_t v = 0;
  std::memcpy(&v, out.data(), std::min<size_t>(8, out.size()));
  return v & 1ull;
}

// XOR-shared bit -> additive u64 share using one Beaver multiplication.
inline uint64_t b2a_bit(uint64_t bx, int party, BeaverMul64& mul) {
  uint64_t a_share = (party == 0) ? (bx & 1ull) : 0ull;
  uint64_t b_share = (party == 1) ? (bx & 1ull) : 0ull;
  uint64_t prod = mul.mul(a_share, b_share);
  uint64_t two_prod = add_mod(prod, prod);
  return sub_mod(add_mod(a_share, b_share), two_prod);
}

// Batched XOR->additive conversion; consumes one triple per element.
inline void b2a_bits_batch(const std::vector<uint64_t>& bits_xor,
                           int party,
                           BeaverMul64& mul,
                           std::vector<uint64_t>& out) {
  const size_t n = bits_xor.size();
  std::vector<uint64_t> xs(n), ys(n);
  for (size_t i = 0; i < n; i++) {
    uint64_t b = bits_xor[i] & 1ull;
    xs[i] = (party == 0) ? b : 0ull;
    ys[i] = (party == 1) ? b : 0ull;
  }
  std::vector<uint64_t> prod;
  mul.mul_batch(xs, ys, prod);
  out.resize(n);
  for (size_t i = 0; i < n; i++) {
    uint64_t two_prod = add_mod(prod[i], prod[i]);
    out[i] = sub_mod(add_mod(xs[i], ys[i]), two_prod);
  }
}

}  // namespace proto
