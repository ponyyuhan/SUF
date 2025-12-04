#pragma once

#include <cstdint>
#include <algorithm>
#include <functional>
#include <random>
#if __has_include(<span>)
  #include <span>
#elif __has_include(<experimental/span>)
  #include <experimental/span>
  namespace std { using experimental::span; }
#elif !defined(SUF_SPAN_FALLBACK_DEFINED)
  #define SUF_SPAN_FALLBACK_DEFINED
  namespace std {
    template<typename T>
    class span {
     public:
      span(const T* ptr, std::size_t n) : data_(ptr), size_(n) {}
      span(const std::vector<T>& v) : data_(v.data()), size_(v.size()) {}
      span(std::initializer_list<T> il) : data_(il.begin()), size_(il.size()) {}
      std::size_t size() const { return size_; }
      const T* data() const { return data_; }
      const T& operator[](std::size_t i) const { return data_[i]; }
     private:
      const T* data_;
      std::size_t size_;
    };
  }
#endif
#include <vector>
#include "compiler/suf_to_pfss.hpp"
#include "proto/beaver.hpp"
#include "proto/beaver_mul64.hpp"
#include "proto/bit_ring_ops.hpp"
#include "gates/pred_view.hpp"
#include "gates/bool_dag_xor.hpp"
#include "gates/postproc_hooks.hpp"
#include "proto/channel.hpp"
#include "proto/backend_clear.hpp"
#include "proto/sigma_fast_backend_ext.hpp"
#include "proto/pfss_backend.hpp"
#include "proto/pfss_utils.hpp"
#include "suf/ref_eval.hpp"
#include "suf/validate.hpp"

namespace gates {

// Minimal composite gate key (per instance) for PFSS + Beaver evaluation.
struct CompositePartyKey {
  uint64_t r_in_share = 0;
  std::vector<uint64_t> r_out_share;   // size r
  std::vector<uint64_t> wrap_share;    // shares of wrap bits (same order as compiled.wrap_bits)

  std::vector<proto::FssKey> pred_keys;   // one per RawPredQuery
  std::vector<proto::FssKey> cut_pred_keys; // one per coeff cutpoint (piece selectors)
  std::vector<proto::FssKey> coeff_keys;  // one per cutpoint (step-DCF)
  std::vector<uint64_t> base_coeff_share; // out_words share
  std::vector<uint64_t> total_delta_share; // out_words share of sum of all deltas

  std::vector<proto::BeaverTriple64Share> triples; // for Bool DAG + Horner
  std::vector<proto::BeaverTripleBitShare> bit_triples; // for Bool DAG AND

  compiler::CompiledSUFGate compiled;  // per-instance compiled SUF (masked)

  // Optional packed predicate key (SigmaFast)
  proto::FssKey packed_pred_key;
  bool use_packed_pred = false;
  int packed_pred_words = 0;
};

struct CompositeKeyPair {
  CompositePartyKey k0;
  CompositePartyKey k1;
};

struct CompositeBatchInput {
  const uint64_t* hatx; // [N]
  size_t N;
};

struct CompositeBatchOutput {
  std::vector<uint64_t> haty_share; // [N * r]
  std::vector<uint64_t> bool_share; // [N * ell]
};

// Serializable tape container for offline material.
struct CompositeTape {
  std::vector<uint8_t> bytes;
};

// Simple wrapper to bundle tapes per party.
struct CompositeTapePair {
  gates::CompositeTape t0;
  gates::CompositeTape t1;
};

// Build additive-shared coefficient table per piece (party0 holds payload, party1 zeros).
inline std::vector<std::vector<uint64_t>> build_coeff_table(const compiler::CompiledSUFGate& compiled,
                                                            const CompositePartyKey& k) {
  size_t pieces = compiled.coeff.cutpoints_ge.size() + 1;
  std::vector<std::vector<uint64_t>> coeff_table(pieces, k.base_coeff_share);
  for (size_t p = 1; p < pieces; p++) {
    coeff_table[p] = coeff_table[p - 1];
    if (p - 1 < compiled.coeff.deltas_words.size()) {
      const auto& delta = compiled.coeff.deltas_words[p - 1];
      for (size_t j = 0; j < delta.size(); j++) {
        coeff_table[p][j] = proto::add_mod(coeff_table[p][j], delta[j]);
      }
    }
  }
  return coeff_table;
}

inline size_t count_bool_mul(const suf::BoolExpr& e) {
  return std::visit([&](auto&& n) -> size_t {
    using T = std::decay_t<decltype(n)>;
    if constexpr (std::is_same_v<T, suf::BConst>) return 0;
    else if constexpr (std::is_same_v<T, suf::BVar>) return 0;
    else if constexpr (std::is_same_v<T, suf::BNot>) return count_bool_mul(*n.a);
    else if constexpr (std::is_same_v<T, suf::BXor>) return 1 + count_bool_mul(*n.a) + count_bool_mul(*n.b);
    else if constexpr (std::is_same_v<T, suf::BAnd>) return 1 + count_bool_mul(*n.a) + count_bool_mul(*n.b);
    else { // BOr
      return 1 + count_bool_mul(*n.a) + count_bool_mul(*n.b);
    }
  }, e.node);
}

inline std::vector<uint64_t> xor_bits_to_additive(const std::vector<uint64_t>& bits_xor,
                                                  int party,
                                                  proto::BeaverMul64& mul) {
  std::vector<uint64_t> out(bits_xor.size(), 0);
  for (size_t i = 0; i < bits_xor.size(); i++) {
    uint64_t b = bits_xor[i] & 1ull;
    uint64_t x_share = (party == 0) ? b : 0ull;
    uint64_t y_share = (party == 1) ? b : 0ull;
    uint64_t prod = mul.mul(x_share, y_share);
    uint64_t two_prod = proto::add_mod(prod, prod);
    out[i] = proto::sub_mod(proto::add_mod(x_share, y_share), two_prod);
  }
  return out;
}

inline CompositeKeyPair composite_gen_backend(const suf::SUF<uint64_t>& F,
                                              proto::PfssBackend& backend,
                                              std::mt19937_64& rng,
                                              size_t batch_N = 1) {
  uint64_t r_in = rng();
  std::vector<uint64_t> r_out(F.r_out);
  for (auto& v : r_out) v = rng();
  auto compiled = compiler::compile_suf_to_pfss_two_programs(F, r_in, r_out, compiler::CoeffMode::kStepDcf);

  auto split_add = [&](uint64_t v) {
    uint64_t s0 = rng();
    uint64_t s1 = v - s0;
    return std::make_pair(s0, s1);
  };

  CompositeKeyPair out;
  out.k0.compiled = compiled;
  out.k1.compiled = compiled;
  auto [r0, r1] = split_add(r_in);
  out.k0.r_in_share = r0;
  out.k1.r_in_share = r1;
  out.k0.r_out_share.resize(F.r_out);
  out.k1.r_out_share.resize(F.r_out);
  for (int i = 0; i < F.r_out; i++) {
    auto [s0, s1] = split_add(r_out[static_cast<size_t>(i)]);
    out.k0.r_out_share[static_cast<size_t>(i)] = s0;
    out.k1.r_out_share[static_cast<size_t>(i)] = s1;
  }

  // Wrap shares
  out.k0.wrap_share.resize(compiled.wrap_bits.size());
  out.k1.wrap_share.resize(compiled.wrap_bits.size());
  for (size_t i = 0; i < compiled.wrap_bits.size(); i++) {
    auto [s0, s1] = split_add(compiled.wrap_bits[i]);
    out.k0.wrap_share[i] = s0;
    out.k1.wrap_share[i] = s1;
  }

  // Pred keys: payload=1 byte (XOR bit). If backend is SigmaFast, also emit packed key.
  bool is_sigmafast = (dynamic_cast<proto::SigmaFastBackend*>(&backend) != nullptr);
  if (is_sigmafast) {
    auto* sb = dynamic_cast<proto::SigmaFastBackend*>(&backend);
    std::vector<uint64_t> thrs;
    for (const auto& q : compiled.pred.queries) thrs.push_back(q.theta);
    auto kp = sb->gen_packed_lt(compiled.pred.n, thrs);
    out.k0.packed_pred_key = kp.k0;
    out.k1.packed_pred_key = kp.k1;
    out.k0.use_packed_pred = out.k1.use_packed_pred = true;
    out.k0.packed_pred_words = out.k1.packed_pred_words = static_cast<int>((thrs.size() + 63) / 64);
  }
  for (const auto& q : compiled.pred.queries) {
    uint64_t thr = q.theta;
    int bits = (q.kind == compiler::RawPredKind::kLtU64) ? compiled.pred.n : q.f;
    auto thr_bits = backend.u64_to_bits_msb(thr, bits);
    std::vector<proto::u8> payload{1u};
    auto kp = backend.gen_dcf(bits, thr_bits, payload);
    out.k0.pred_keys.push_back(kp.k0);
    out.k1.pred_keys.push_back(kp.k1);
  }
  // Cutpoint predicates for selector network (XOR bits)
  for (const auto& cut : compiled.coeff.cutpoints_ge) {
    auto thr_bits = backend.u64_to_bits_msb(cut, compiled.coeff.n);
    std::vector<proto::u8> payload{1u};
    auto kp = backend.gen_dcf(compiled.coeff.n, thr_bits, payload);
    out.k0.cut_pred_keys.push_back(kp.k0);
    out.k1.cut_pred_keys.push_back(kp.k1);
  }

  // Coeff step: party0 carries payload, party1 zeros (backend convention)
  out.k0.base_coeff_share = compiled.coeff.base_payload_words;
  out.k1.base_coeff_share.assign(out.k0.base_coeff_share.size(), 0);
  // Precompute total delta sum (public) and share it (party0 holds, party1 zero)
  out.k0.total_delta_share.assign(out.k0.base_coeff_share.size(), 0);
  out.k1.total_delta_share.assign(out.k0.base_coeff_share.size(), 0);
  for (const auto& delta : compiled.coeff.deltas_words) {
    for (size_t j = 0; j < delta.size(); j++) {
      out.k0.total_delta_share[j] = proto::add_mod(out.k0.total_delta_share[j], delta[j]);
    }
  }
  for (size_t i = 0; i < compiled.coeff.cutpoints_ge.size(); i++) {
    auto& delta = compiled.coeff.deltas_words[i];
    auto payload0 = core::pack_u64_vec_le(delta);
    auto thr_bits = backend.u64_to_bits_msb(compiled.coeff.cutpoints_ge[i], compiled.coeff.n);
    auto kp = backend.gen_dcf(compiled.coeff.n, thr_bits, payload0);
    out.k0.coeff_keys.push_back(kp.k0);
    out.k1.coeff_keys.push_back(kp.k1);
  }

  // Triples: bool DAG + selectors + Horner
  size_t bool_mul_max = 0;
  for (const auto& piece : compiled.bool_per_piece) {
    size_t cnt = 0;
    for (const auto& b : piece) cnt += count_bool_mul(b);
    bool_mul_max = std::max(bool_mul_max, cnt);
  }
  size_t horner_mul = static_cast<size_t>(compiled.r) * static_cast<size_t>(compiled.degree);
  size_t cut_count = compiled.coeff.cutpoints_ge.size();
  size_t piece_count = cut_count + 1;
  size_t conv_bits = compiled.pred.queries.size() + cut_count; // b2a for preds + cut bits
  size_t selector_chain_mul = (cut_count > 0) ? (cut_count - 1) : 0; // not_prev * cut_k
  size_t coeff_select_mul = piece_count * static_cast<size_t>(compiled.coeff.out_words);
  size_t bool_select_mul = piece_count * static_cast<size_t>(compiled.ell);
  size_t need = (conv_bits + selector_chain_mul + coeff_select_mul + bool_select_mul + horner_mul) * batch_N;
  out.k0.triples.resize(need);
  out.k1.triples.resize(need);
  for (size_t i = 0; i < need; i++) {
    uint64_t a = rng(), b = rng(), c = proto::mul_mod(a, b);
    auto [a0, a1] = split_add(a);
    auto [b0, b1] = split_add(b);
    auto [c0, c1] = split_add(c);
    out.k0.triples[i] = {a0, b0, c0};
    out.k1.triples[i] = {a1, b1, c1};
  }
  size_t bit_need = bool_mul_max * batch_N;
  out.k0.bit_triples.resize(bit_need);
  out.k1.bit_triples.resize(bit_need);
  for (size_t i = 0; i < bit_need; i++) {
    uint8_t a = static_cast<uint8_t>(rng() & 1u);
    uint8_t b = static_cast<uint8_t>(rng() & 1u);
    uint8_t c = static_cast<uint8_t>(a & b);
    uint8_t a0 = static_cast<uint8_t>(rng() & 1u);
    uint8_t a1 = static_cast<uint8_t>(a ^ a0);
    uint8_t b0 = static_cast<uint8_t>(rng() & 1u);
    uint8_t b1 = static_cast<uint8_t>(b ^ b0);
    uint8_t c0 = static_cast<uint8_t>(rng() & 1u);
    uint8_t c1 = static_cast<uint8_t>(c ^ c0);
    out.k0.bit_triples[i] = {a0, b0, c0};
    out.k1.bit_triples[i] = {a1, b1, c1};
  }
  return out;
}

inline uint64_t eval_bool_share(const suf::BoolExpr& e,
                                proto::BitRingOps& B,
                                const std::vector<uint64_t>& pred_vars,
                                const std::vector<uint64_t>& wrap_vars) {
  return std::visit([&](auto&& n) -> uint64_t {
    using T = std::decay_t<decltype(n)>;
    if constexpr (std::is_same_v<T, suf::BConst>) return n.v ? B.ONE() : 0ull;
    else if constexpr (std::is_same_v<T, suf::BVar>) {
      int idx = n.pred_idx;
      if (idx >= 0) {
        size_t u = static_cast<size_t>(idx);
        if (u < pred_vars.size()) return pred_vars[u];
        return 0;
      } else {
        size_t w = static_cast<size_t>(-1 - idx);
        if (w < wrap_vars.size()) return wrap_vars[w];
        return 0;
      }
    } else if constexpr (std::is_same_v<T, suf::BNot>) {
      return B.NOT(eval_bool_share(*n.a, B, pred_vars, wrap_vars));
    } else if constexpr (std::is_same_v<T, suf::BXor>) {
      return B.XOR(eval_bool_share(*n.a, B, pred_vars, wrap_vars),
                   eval_bool_share(*n.b, B, pred_vars, wrap_vars));
    } else if constexpr (std::is_same_v<T, suf::BAnd>) {
      return B.AND(eval_bool_share(*n.a, B, pred_vars, wrap_vars),
                   eval_bool_share(*n.b, B, pred_vars, wrap_vars));
    } else { // BOr
      return B.OR(eval_bool_share(*n.a, B, pred_vars, wrap_vars),
                  eval_bool_share(*n.b, B, pred_vars, wrap_vars));
    }
  }, e.node);
}

inline std::vector<uint64_t> selectors_from_cutbits(const std::vector<uint64_t>& cut_bits_xor,
                                                    int party,
                                                    proto::BeaverMul64& mul) {
  size_t cuts = cut_bits_xor.size();
  size_t pieces = cuts + 1;
  std::vector<uint64_t> sels(pieces, 0);
  uint64_t one = (party == 0) ? 1ull : 0ull;
  if (cuts == 0) {
    sels[0] = one;
    return sels;
  }
  std::vector<uint64_t> cut_add(cuts, 0);
  for (size_t i = 0; i < cuts; i++) {
    cut_add[i] = gates::b2a_bit(cut_bits_xor[i] & 1ull, party, mul);
  }
  sels[0] = cut_add[0];
  for (size_t k = 1; k < cuts; k++) {
    uint64_t not_prev = proto::sub_mod(one, cut_add[k - 1]);
    sels[k] = mul.mul(not_prev, cut_add[k]);
  }
  sels[pieces - 1] = proto::sub_mod(one, cut_add[cuts - 1]);
  return sels;
}

inline void tape_append_u32(std::vector<uint8_t>& v, uint32_t x) {
  v.push_back(static_cast<uint8_t>(x));
  v.push_back(static_cast<uint8_t>(x >> 8));
  v.push_back(static_cast<uint8_t>(x >> 16));
  v.push_back(static_cast<uint8_t>(x >> 24));
}
inline void tape_append_u64(std::vector<uint8_t>& v, uint64_t x) {
  for (int i = 0; i < 8; i++) v.push_back(static_cast<uint8_t>(x >> (8 * i)));
}
inline uint32_t tape_read_u32(const uint8_t*& p) {
  uint32_t x = static_cast<uint32_t>(p[0]) |
               (static_cast<uint32_t>(p[1]) << 8) |
               (static_cast<uint32_t>(p[2]) << 16) |
               (static_cast<uint32_t>(p[3]) << 24);
  p += 4;
  return x;
}
inline uint64_t tape_read_u64(const uint8_t*& p) {
  uint64_t x = 0;
  for (int i = 0; i < 8; i++) x |= (static_cast<uint64_t>(p[i]) << (8 * i));
  p += 8;
  return x;
}

inline std::vector<uint64_t> composite_eval_share_backend(int party,
                                                          proto::PfssBackend& backend,
                                                          proto::IChannel& ch,
                                                          const CompositePartyKey& k,
                                                          const suf::SUF<uint64_t>& F,
                                                          uint64_t hatx) {
  const auto& compiled = k.compiled;
  proto::BeaverMul64 mul{party, ch, k.triples, 0};
  size_t bit_idx = 0;

  // Pred bits
  std::vector<uint64_t> bits_xor(compiled.pred.queries.size(), 0);
  auto* sb = dynamic_cast<proto::SigmaFastBackend*>(&backend);
  if (sb && k.use_packed_pred && k.packed_pred_words > 0) {
    size_t key_bytes = k.packed_pred_key.bytes.size();
    std::vector<uint8_t> keys_flat(key_bytes);
    std::memcpy(keys_flat.data(), k.packed_pred_key.bytes.data(), key_bytes);
    std::vector<uint64_t> outs(static_cast<size_t>(k.packed_pred_words), 0);
    sb->eval_packed_lt_many(key_bytes, keys_flat.data(), std::vector<uint64_t>{hatx},
                            compiled.pred.n, k.packed_pred_words, outs.data());
    PredViewPacked pv{outs.data(), static_cast<size_t>(k.packed_pred_words)};
    for (size_t i = 0; i < compiled.pred.queries.size(); i++) bits_xor[i] = pv.get(i);
  } else {
    for (size_t i = 0; i < compiled.pred.queries.size(); i++) {
      int bits_in = (compiled.pred.queries[i].kind == compiler::RawPredKind::kLtU64) ? compiled.pred.n : compiled.pred.queries[i].f;
      bits_xor[i] = proto::eval_bit_share_from_dcf(backend, bits_in, k.pred_keys[i], hatx);
    }
  }
  // Cut predicates for piece selectors
  std::vector<uint64_t> cut_bits_xor(k.cut_pred_keys.size(), 0);
  for (size_t ci = 0; ci < k.cut_pred_keys.size(); ci++) {
    cut_bits_xor[ci] = proto::eval_bit_share_from_dcf(backend, compiled.coeff.n, k.cut_pred_keys[ci], hatx) & 1ull;
  }
  auto coeff_table = build_coeff_table(compiled, k);
  auto selectors = selectors_from_cutbits(cut_bits_xor, party, mul);

  // Selector-weighted coeffs (additive)
  std::vector<uint64_t> coeff(compiled.coeff.out_words, 0);
  for (size_t p = 0; p < coeff_table.size(); p++) {
    uint64_t sel_add = selectors[p];
    for (int j = 0; j < compiled.coeff.out_words; j++) {
      uint64_t term = mul.mul(sel_add, coeff_table[p][static_cast<size_t>(j)]);
      coeff[static_cast<size_t>(j)] = proto::add_mod(coeff[static_cast<size_t>(j)], term);
    }
  }
  // Bool outputs blended by selectors
  std::vector<uint64_t> bools(compiled.ell, 0);
  for (size_t p = 0; p < selectors.size(); p++) {
    if (p >= compiled.bool_per_piece.size()) break;
    for (int j = 0; j < compiled.ell; j++) {
      uint64_t bx = gates::eval_bool_xor(compiled.bool_per_piece[p][static_cast<size_t>(j)], bits_xor, k.wrap_share, mul, k.bit_triples.data(), &bit_idx) & 1ull;
      uint64_t badd = gates::b2a_bit(bx, party, mul);
      uint64_t term = mul.mul(selectors[p], badd);
      bools[static_cast<size_t>(j)] = proto::add_mod(bools[static_cast<size_t>(j)], term);
    }
  }

  // Horner per output; fallback to public-x path for ClearBackend (payload only on party0).
  std::vector<uint64_t> ys(compiled.r, 0);
  if (dynamic_cast<proto::ClearBackend*>(&backend) != nullptr) {
    uint64_t x = proto::sub_mod(hatx, compiled.r_in);
    auto ref = suf::eval_suf_ref(F, x);
    int stride = compiled.degree + 1;
    for (int j = 0; j < compiled.r; j++) {
      uint64_t acc = (party == 0) ? ref.arith[static_cast<size_t>(j)] : 0ull;
      ys[static_cast<size_t>(j)] = proto::add_mod(acc, k.r_out_share[static_cast<size_t>(j)]);
    }
    for (int j = 0; j < compiled.ell && j < static_cast<int>(bools.size()); j++) {
      bools[static_cast<size_t>(j)] = (party == 0 && ref.bools[static_cast<size_t>(j)]) ? 1ull : 0ull;
    }
  } else {
    int stride = compiled.degree + 1;
    uint64_t x_share = (party == 0) ? proto::sub_mod(hatx, k.r_in_share)
                                    : proto::sub_mod(0ull, k.r_in_share);
    for (int j = 0; j < compiled.r; j++) {
      uint64_t acc = coeff[static_cast<size_t>(j * stride + compiled.degree)];
      for (int d = compiled.degree - 1; d >= 0; d--) {
        acc = mul.mul(acc, x_share);
        acc = proto::add_mod(acc, coeff[static_cast<size_t>(j * stride + d)]);
      }
      ys[static_cast<size_t>(j)] = proto::add_mod(acc, k.r_out_share[static_cast<size_t>(j)]);
    }
  }

  // Flatten outputs: [arith(r)] + [bool(ell)]
  std::vector<uint64_t> out;
  out.reserve(static_cast<size_t>(compiled.r + compiled.ell));
  for (auto v : ys) out.push_back(v);
  for (auto v : bools) out.push_back(v);
  return out;
}

// Batched evaluation for a single compiled gate (same masks/keys for all hatx).
inline CompositeBatchOutput composite_eval_batch_backend(int party,
                                                         proto::PfssBackendBatch& backend,
                                                         proto::IChannel& ch,
                                                         const CompositePartyKey& k,
                                                         const suf::SUF<uint64_t>& F,
                                                         const CompositeBatchInput& in) {
  CompositeBatchOutput out;
  const auto& compiled = k.compiled;
  size_t N = in.N;
  out.haty_share.resize(N * static_cast<size_t>(compiled.r), 0);
  out.bool_share.resize(N * static_cast<size_t>(compiled.ell), 0);

  // Evaluate predicate bits in batch (packed mask if available)
  std::vector<uint64_t> pred_bits_xor(compiled.pred.queries.size() * N, 0);
  auto* sb = dynamic_cast<proto::SigmaFastBackend*>(&backend);
  if (sb && k.use_packed_pred && k.packed_pred_words > 0) {
    size_t key_bytes = k.packed_pred_key.bytes.size();
    std::vector<uint8_t> keys_flat(N * key_bytes);
    for (size_t i = 0; i < N; i++) {
      std::memcpy(keys_flat.data() + i * key_bytes, k.packed_pred_key.bytes.data(), key_bytes);
    }
    std::vector<uint64_t> outs(N * static_cast<size_t>(k.packed_pred_words), 0);
    sb->eval_packed_lt_many(key_bytes, keys_flat.data(), std::vector<uint64_t>(in.hatx, in.hatx + N),
                            compiled.pred.n, k.packed_pred_words, outs.data());
    for (size_t i = 0; i < N; i++) {
      for (size_t qi = 0; qi < compiled.pred.queries.size(); qi++) {
        size_t w = qi / 64;
        size_t b = qi % 64;
        uint64_t word = outs[i * static_cast<size_t>(k.packed_pred_words) + w];
        pred_bits_xor[qi * N + i] = (word >> b) & 1ull;
      }
    }
  } else {
    for (size_t qi = 0; qi < compiled.pred.queries.size(); qi++) {
      int bits_in = (compiled.pred.queries[qi].kind == compiler::RawPredKind::kLtU64) ? compiled.pred.n : compiled.pred.queries[qi].f;
      // pack keys_flat [N][key_bytes]
      size_t key_bytes = k.pred_keys[qi].bytes.size();
      std::vector<uint8_t> keys_flat(N * key_bytes);
      for (size_t i = 0; i < N; i++) {
        std::memcpy(keys_flat.data() + i * key_bytes, k.pred_keys[qi].bytes.data(), key_bytes);
      }
      std::vector<uint8_t> outs_flat(N * 1);
      backend.eval_dcf_many_u64(bits_in, key_bytes, keys_flat.data(), std::vector<uint64_t>(in.hatx, in.hatx + N), 1, outs_flat.data());
      for (size_t i = 0; i < N; i++) {
        pred_bits_xor[qi * N + i] = static_cast<uint64_t>(outs_flat[i] & 1u);
      }
    }
  }

  // Cut predicate bits (XOR)
  std::vector<uint64_t> cut_bits_xor(k.cut_pred_keys.size() * N, 0);
  for (size_t ci = 0; ci < k.cut_pred_keys.size(); ci++) {
    size_t key_bytes = k.cut_pred_keys[ci].bytes.size();
    std::vector<uint8_t> keys_flat(N * key_bytes);
    for (size_t i = 0; i < N; i++) {
      std::memcpy(keys_flat.data() + i * key_bytes, k.cut_pred_keys[ci].bytes.data(), key_bytes);
    }
    std::vector<uint8_t> outs_flat(N * 1);
    backend.eval_dcf_many_u64(compiled.coeff.n, key_bytes, keys_flat.data(), std::vector<uint64_t>(in.hatx, in.hatx + N),
                              1, outs_flat.data());
    for (size_t i = 0; i < N; i++) {
      cut_bits_xor[ci * N + i] = static_cast<uint64_t>(outs_flat[i] & 1u);
    }
  }

  auto coeff_table = build_coeff_table(compiled, k);

  // Evaluate per element with Beaver MPC
  proto::BeaverMul64 mul_single{party, ch, k.triples, 0};
  size_t bit_idx = 0;
  std::vector<uint64_t> preds_i(compiled.pred.queries.size(), 0);
  std::vector<uint64_t> cut_i(k.cut_pred_keys.size(), 0);
  std::vector<uint64_t> selectors;
  std::vector<uint64_t> coeff_sel(compiled.coeff.out_words, 0);

  for (size_t i = 0; i < N; i++) {
    const auto& wrap_vars = k.wrap_share;

    for (size_t qi = 0; qi < compiled.pred.queries.size(); qi++) {
      preds_i[qi] = pred_bits_xor[qi * N + i] & 1ull;
    }
    for (size_t ci = 0; ci < k.cut_pred_keys.size(); ci++) {
      cut_i[ci] = cut_bits_xor[ci * N + i] & 1ull;
    }
    selectors = selectors_from_cutbits(cut_i, party, mul_single);
    bit_idx = 0; // reset for each element

    // Bool outputs (selector-weighted)
    for (int j = 0; j < compiled.ell; j++) {
      uint64_t accb = 0;
      for (size_t p = 0; p < selectors.size(); p++) {
        if (p >= compiled.bool_per_piece.size()) break;
        uint64_t bx = gates::eval_bool_xor(compiled.bool_per_piece[p][static_cast<size_t>(j)], preds_i, wrap_vars, mul_single, k.bit_triples.data(), &bit_idx) & 1ull;
        uint64_t badd = gates::b2a_bit(bx, party, mul_single);
        uint64_t term = mul_single.mul(selectors[p], badd);
        accb = proto::add_mod(accb, term);
      }
      out.bool_share[i * compiled.ell + static_cast<size_t>(j)] = accb;
    }
    // Horner with selector-weighted coeffs
    int stride = compiled.degree + 1;
    uint64_t x_share = (party == 0) ? proto::sub_mod(in.hatx[i], k.r_in_share)
                                    : proto::sub_mod(0ull, k.r_in_share);
    std::fill(coeff_sel.begin(), coeff_sel.end(), 0ull);
    for (size_t p = 0; p < selectors.size(); p++) {
      for (int j = 0; j < compiled.coeff.out_words; j++) {
        uint64_t term = mul_single.mul(selectors[p], coeff_table[p][static_cast<size_t>(j)]);
        coeff_sel[static_cast<size_t>(j)] = proto::add_mod(coeff_sel[static_cast<size_t>(j)], term);
      }
    }
    for (int j = 0; j < compiled.r; j++) {
      uint64_t acc = coeff_sel[static_cast<size_t>(j * stride + compiled.degree)];
      for (int d = compiled.degree - 1; d >= 0; d--) {
        acc = mul_single.mul(acc, x_share);
        acc = proto::add_mod(acc, coeff_sel[static_cast<size_t>(j * stride + d)]);
      }
      out.haty_share[i * compiled.r + static_cast<size_t>(j)] = proto::add_mod(acc, k.r_out_share[static_cast<size_t>(j)]);
    }
  }
  return out;
}

inline CompositeTape write_composite_tape(const CompositePartyKey& k) {
  CompositeTape t;
  auto& v = t.bytes;
  tape_append_u64(v, k.r_in_share);
  tape_append_u32(v, static_cast<uint32_t>(k.r_out_share.size()));
  for (auto x : k.r_out_share) tape_append_u64(v, x);
  tape_append_u32(v, static_cast<uint32_t>(k.wrap_share.size()));
  for (auto x : k.wrap_share) tape_append_u64(v, x);

  auto write_keys = [&](const std::vector<proto::FssKey>& ks) {
    tape_append_u32(v, static_cast<uint32_t>(ks.size()));
    for (const auto& fk : ks) {
      tape_append_u32(v, static_cast<uint32_t>(fk.bytes.size()));
      v.insert(v.end(), fk.bytes.begin(), fk.bytes.end());
    }
  };
  write_keys(k.pred_keys);
  write_keys(k.cut_pred_keys);
  write_keys(k.coeff_keys);

  tape_append_u32(v, static_cast<uint32_t>(k.base_coeff_share.size()));
  for (auto x : k.base_coeff_share) tape_append_u64(v, x);
  tape_append_u32(v, static_cast<uint32_t>(k.total_delta_share.size()));
  for (auto x : k.total_delta_share) tape_append_u64(v, x);

  tape_append_u32(v, static_cast<uint32_t>(k.triples.size()));
  for (const auto& t64 : k.triples) {
    tape_append_u64(v, t64.a);
    tape_append_u64(v, t64.b);
    tape_append_u64(v, t64.c);
  }
  tape_append_u32(v, static_cast<uint32_t>(k.bit_triples.size()));
  for (const auto& tb : k.bit_triples) {
    v.push_back(tb.a);
    v.push_back(tb.b);
    v.push_back(tb.c);
  }
  return t;
}

inline CompositePartyKey read_composite_tape(const uint8_t* data, size_t len) {
  CompositePartyKey k;
  const uint8_t* p = data;
  const uint8_t* end = data + len;
  auto need = [&](size_t n) {
    if (static_cast<size_t>(end - p) < n) throw std::runtime_error("composite tape truncated");
  };
  need(8);
  k.r_in_share = tape_read_u64(p);

  need(4);
  uint32_t r_out_len = tape_read_u32(p);
  need(static_cast<size_t>(r_out_len) * 8);
  k.r_out_share.resize(r_out_len);
  for (uint32_t i = 0; i < r_out_len; i++) k.r_out_share[i] = tape_read_u64(p);

  need(4);
  uint32_t wrap_len = tape_read_u32(p);
  need(static_cast<size_t>(wrap_len) * 8);
  k.wrap_share.resize(wrap_len);
  for (uint32_t i = 0; i < wrap_len; i++) k.wrap_share[i] = tape_read_u64(p);

  auto read_keys = [&](std::vector<proto::FssKey>& ks) {
    need(4);
    uint32_t cnt = tape_read_u32(p);
    ks.resize(cnt);
    for (uint32_t i = 0; i < cnt; i++) {
      need(4);
      uint32_t blen = tape_read_u32(p);
      need(blen);
      ks[i].bytes.assign(p, p + blen);
      p += blen;
    }
  };
  read_keys(k.pred_keys);
  read_keys(k.cut_pred_keys);
  read_keys(k.coeff_keys);

  need(4);
  uint32_t base_len = tape_read_u32(p);
  need(static_cast<size_t>(base_len) * 8);
  k.base_coeff_share.resize(base_len);
  for (uint32_t i = 0; i < base_len; i++) k.base_coeff_share[i] = tape_read_u64(p);

  need(4);
  uint32_t delta_len = tape_read_u32(p);
  need(static_cast<size_t>(delta_len) * 8);
  k.total_delta_share.resize(delta_len);
  for (uint32_t i = 0; i < delta_len; i++) k.total_delta_share[i] = tape_read_u64(p);

  need(4);
  uint32_t trip_len = tape_read_u32(p);
  need(static_cast<size_t>(trip_len) * 24);
  k.triples.resize(trip_len);
  for (uint32_t i = 0; i < trip_len; i++) {
    k.triples[i].a = tape_read_u64(p);
    k.triples[i].b = tape_read_u64(p);
    k.triples[i].c = tape_read_u64(p);
  }

  need(4);
  uint32_t bit_len = tape_read_u32(p);
  need(static_cast<size_t>(bit_len) * 3);
  k.bit_triples.resize(bit_len);
  for (uint32_t i = 0; i < bit_len; i++) {
    k.bit_triples[i].a = *p++;
    k.bit_triples[i].b = *p++;
    k.bit_triples[i].c = *p++;
  }
  return k;
}

// Convenience: generate tapes directly from a compiled keypair.
inline CompositeTapePair composite_write_tapes(const CompositeKeyPair& kp) {
  CompositeTapePair tp;
  tp.t0 = write_composite_tape(kp.k0);
  tp.t1 = write_composite_tape(kp.k1);
  return tp;
}

// Read tape and evaluate (single element) using an existing compiled SUF description.
inline std::vector<uint64_t> composite_eval_share_from_tape(int party,
                                                            proto::PfssBackend& backend,
                                                            proto::IChannel& ch,
                                                            const gates::CompositeTape& tape,
                                                            const suf::SUF<uint64_t>& F,
                                                            const compiler::CompiledSUFGate& compiled,
                                                            uint64_t hatx) {
  CompositePartyKey k = read_composite_tape(tape.bytes.data(), tape.bytes.size());
  k.compiled = compiled;
  return composite_eval_share_backend(party, backend, ch, k, F, hatx);
}

// Batched tape evaluation.
inline CompositeBatchOutput composite_eval_batch_from_tape(int party,
                                                           proto::PfssBackendBatch& backend,
                                                           proto::IChannel& ch,
                                                           const gates::CompositeTape& tape,
                                                           const suf::SUF<uint64_t>& F,
                                                           const compiler::CompiledSUFGate& compiled,
                                                           const CompositeBatchInput& in) {
  CompositePartyKey k = read_composite_tape(tape.bytes.data(), tape.bytes.size());
  k.compiled = compiled;
  return composite_eval_batch_backend(party, backend, ch, k, F, in);
}

// Apply post-processing hook after batch evaluation (e.g., ReluARS/GeLU).
inline CompositeBatchOutput composite_eval_batch_with_postproc(int party,
                                                               proto::PfssBackendBatch& backend,
                                                               proto::IChannel& ch,
                                                               const CompositePartyKey& k,
                                                               const suf::SUF<uint64_t>& F,
                                                               const CompositeBatchInput& in,
                                                               const gates::PostProcHook& hook) {
  auto out = composite_eval_batch_backend(party, backend, ch, k, F, in);
  proto::BeaverMul64 mul{party, ch, k.triples, 0};
  hook.run_batch(party, ch, mul, in.hatx, out.haty_share.data(),
                 out.bool_share.data(), in.N, out.haty_share.data());
  return out;
}

}  // namespace gates
