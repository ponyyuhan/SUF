#pragma once

#include <cstdint>
#include <algorithm>
#include <functional>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <cstddef>
#if !defined(SUF_SPAN_FALLBACK_DEFINED)
  #define SUF_SPAN_FALLBACK_DEFINED
  namespace std {
    template<typename T>
    class span {
     public:
      span() : data_(nullptr), size_(0) {}
      span(const T* ptr, std::size_t n) : data_(ptr), size_(n) {}
      template <typename U, typename = std::enable_if_t<std::is_same_v<std::remove_const_t<T>, U>>>
      span(const std::vector<U>& v) : data_(v.data()), size_(v.size()) {}
      span(std::initializer_list<T> il) : data_(il.begin()), size_(il.size()) {}
      std::size_t size() const { return size_; }
      const T* data() const { return data_; }
      T* data() { return const_cast<T*>(data_); }
      const T& operator[](std::size_t i) const { return data_[i]; }
      T& operator[](std::size_t i) { return const_cast<T&>(data_[i]); }
      span subspan(std::size_t off, std::size_t n) const {
        if (off > size_) return span();
        std::size_t len = (off + n > size_) ? (size_ - off) : n;
        return span(data_ + off, len);
      }
      const T* begin() const { return data_; }
      const T* end() const { return data_ + size_; }
     private:
      const T* data_;
      std::size_t size_;
    };
    template <typename T>
    const T* begin(span<T> s) { return s.data(); }
    template <typename T>
    const T* end(span<T> s) { return s.data() + s.size(); }
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
#include "suf/trunc_suf_builders.hpp"
#include "suf/ref_eval.hpp"
#include "suf/validate.hpp"

namespace gates {

// Minimal composite gate key (per instance) for PFSS + Beaver evaluation.
struct CompositePartyKey {
  uint64_t r_in_share = 0;
  std::vector<uint64_t> r_out_share;   // size r
  std::vector<uint64_t> wrap_share;    // shares of wrap bits (same order as compiled.wrap_bits)

  // Optional postproc masks (ReluARS/GeLU). Stored if layout needs them.
  uint64_t r_hi_share = 0;       // r_in >> f share (ReluARS)
  uint64_t wrap_sign_share = 0;  // wrap bit share for ReluARS
  std::vector<uint64_t> extra_params; // gate-specific constants (e.g., delta LUT flattened)

  proto::PredKeyMeta pred_meta;
  proto::PredKeyMeta cut_pred_meta;
  proto::CoeffKeyMeta coeff_meta;

  std::vector<proto::FssKey> pred_keys;   // one per RawPredQuery
  std::vector<proto::FssKey> cut_pred_keys; // one per coeff cutpoint (piece selectors)
  std::vector<proto::FssKey> coeff_keys;  // one per cutpoint (step-DCF)
  std::vector<uint64_t> base_coeff_share; // out_words share
  std::vector<uint64_t> total_delta_share; // out_words share of sum of all deltas

  std::vector<proto::BeaverTriple64Share> triples; // for Bool DAG + Horner
  std::vector<proto::BeaverTripleBitShare> bit_triples; // for Bool DAG AND

  compiler::CompiledSUFGate compiled;  // per-instance compiled new (masked)

  // Optional packed predicate key (SigmaFast)
  proto::FssKey packed_pred_key;
  bool use_packed_pred = false;
  int packed_pred_words = 0;

  // Optional packed cutpoint predicates (SigmaFast)
  proto::FssKey packed_cut_key;
  bool use_packed_cut = false;
  int packed_cut_words = 0;
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
                                                            const CompositePartyKey& k,
                                                            bool add_deltas) {
  if (k.coeff_meta.sem != proto::ShareSemantics::AddU64) {
    throw std::runtime_error("build_coeff_table: coeff semantics must be additive");
  }
  size_t pieces = compiled.coeff.cutpoints_ge.size() + 1;
  std::vector<std::vector<uint64_t>> coeff_table(pieces, k.base_coeff_share);
  for (size_t p = 1; p < pieces; p++) {
    coeff_table[p] = coeff_table[p - 1];
    if (add_deltas && p - 1 < compiled.coeff.deltas_words.size()) {
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

  // Optional gate-specific extras (e.g., ReluARS masks/params): default zeros.
  out.k0.r_hi_share = 0;
  out.k1.r_hi_share = 0;
  out.k0.wrap_sign_share = 0;
  out.k1.wrap_sign_share = 0;
  out.k0.extra_params = compiled.extra_u64;
  out.k1.extra_params = compiled.extra_u64;

  proto::BitOrder bit_order = backend.bit_order();
  out.k0.pred_meta.bit_order = bit_order;
  out.k1.pred_meta.bit_order = bit_order;
  out.k0.pred_meta.sem = proto::ShareSemantics::XorBytes;
  out.k1.pred_meta.sem = proto::ShareSemantics::XorBytes;
  out.k0.pred_meta.out_bytes = 1;
  out.k1.pred_meta.out_bytes = 1;
  out.k0.cut_pred_meta = out.k0.pred_meta;
  out.k1.cut_pred_meta = out.k1.pred_meta;
  out.k0.coeff_meta.bit_order = bit_order;
  out.k1.coeff_meta.bit_order = bit_order;
  out.k0.coeff_meta.sem = proto::ShareSemantics::AddU64;
  out.k1.coeff_meta.sem = proto::ShareSemantics::AddU64;

  // Pred keys: payload=1 byte (XOR bit). If backend is SigmaFast, also emit packed key.
  bool is_sigmafast = (dynamic_cast<proto::SigmaFastBackend*>(&backend) != nullptr);
  if (is_sigmafast && !compiled.pred.queries.empty()) {
    auto* sb = dynamic_cast<proto::SigmaFastBackend*>(&backend);
    std::vector<uint64_t> thrs;
    for (const auto& q : compiled.pred.queries) thrs.push_back(q.theta);
    auto kp = sb->gen_packed_lt(compiled.pred.n, thrs);
    out.k0.packed_pred_key = kp.k0;
    out.k1.packed_pred_key = kp.k1;
    out.k0.use_packed_pred = out.k1.use_packed_pred = true;
    out.k0.packed_pred_words = out.k1.packed_pred_words = static_cast<int>((thrs.size() + 63) / 64);
    // Packed cutpoints (selector predicates)
    if (!compiled.coeff.cutpoints_ge.empty()) {
      auto kp_cut = sb->gen_packed_lt(compiled.coeff.n, compiled.coeff.cutpoints_ge);
      out.k0.packed_cut_key = kp_cut.k0;
      out.k1.packed_cut_key = kp_cut.k1;
      out.k0.use_packed_cut = out.k1.use_packed_cut = true;
      out.k0.packed_cut_words = out.k1.packed_cut_words = static_cast<int>((compiled.coeff.cutpoints_ge.size() + 63) / 64);
    }
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

// Specialized generator for truncation/ARS gates: fixes r_low used in predicates and
// provides r_hi shares for postproc hooks.
inline CompositeKeyPair composite_gen_trunc_gate(proto::PfssBackend& backend,
                                                 std::mt19937_64& rng,
                                                 int frac_bits,
                                                 compiler::GateKind kind,
                                                 size_t batch_N = 1,
                                                 suf::SUF<uint64_t>* F_out = nullptr) {
  if (kind != compiler::GateKind::FaithfulTR &&
      kind != compiler::GateKind::FaithfulARS &&
      kind != compiler::GateKind::GapARS) {
    throw std::runtime_error("composite_gen_trunc_gate: unsupported GateKind");
  }
  uint64_t r_mask = (frac_bits <= 0) ? 0ull
                                     : (frac_bits >= 64 ? ~uint64_t(0) : ((uint64_t(1) << frac_bits) - 1));
  uint64_t r = rng();
  uint64_t r_low = (frac_bits <= 0) ? 0ull : (r & r_mask);
  uint64_t r_hi = (frac_bits >= 64) ? 0ull : (r >> frac_bits);
  suf::SUF<uint64_t> F;
  if (kind == compiler::GateKind::FaithfulTR) {
    F = suf::build_trunc_faithful_suf(frac_bits, r_low);
  } else if (kind == compiler::GateKind::FaithfulARS) {
    F = suf::build_ars_faithful_suf(frac_bits, r_low);
  } else {
    F = suf::build_gapars_suf(frac_bits, r_low);
  }
  if (F_out) *F_out = F;
  std::vector<uint64_t> r_out = {rng()};
  auto compiled = compiler::compile_suf_to_pfss_two_programs(F, r, r_out, compiler::CoeffMode::kStepDcf, kind);
  compiled.layout.arith_ports = {"y"};
  compiled.layout.bool_ports.clear();
  if (frac_bits > 0) compiled.layout.bool_ports.push_back("carry");
  if (kind != compiler::GateKind::FaithfulTR) compiled.layout.bool_ports.push_back("sign");
  compiled.extra_u64 = {static_cast<uint64_t>(frac_bits), r_low};

  auto split_add = [&](uint64_t v) {
    uint64_t s0 = rng();
    uint64_t s1 = v - s0;
    return std::make_pair(s0, s1);
  };

  CompositeKeyPair out;
  out.k0.compiled = compiled;
  out.k1.compiled = compiled;
  auto [r0, r1] = split_add(r);
  out.k0.r_in_share = r0;
  out.k1.r_in_share = r1;
  out.k0.r_out_share.resize(1);
  out.k1.r_out_share.resize(1);
  auto [rout0, rout1] = split_add(r_out[0]);
  out.k0.r_out_share[0] = rout0;
  out.k1.r_out_share[0] = rout1;

  // Wrap shares
  out.k0.wrap_share.resize(compiled.wrap_bits.size());
  out.k1.wrap_share.resize(compiled.wrap_bits.size());
  for (size_t i = 0; i < compiled.wrap_bits.size(); i++) {
    auto [s0, s1] = split_add(compiled.wrap_bits[i]);
    out.k0.wrap_share[i] = s0;
    out.k1.wrap_share[i] = s1;
  }

  // Provide r_hi shares for postproc hook.
  auto [rhi0, rhi1] = split_add(r_hi);
  out.k0.r_hi_share = rhi0;
  out.k1.r_hi_share = rhi1;
  out.k0.wrap_sign_share = 0;
  out.k1.wrap_sign_share = 0;
  out.k0.extra_params = compiled.extra_u64;
  out.k1.extra_params = compiled.extra_u64;

  proto::BitOrder bit_order = backend.bit_order();
  out.k0.pred_meta.bit_order = bit_order;
  out.k1.pred_meta.bit_order = bit_order;
  out.k0.pred_meta.sem = proto::ShareSemantics::XorBytes;
  out.k1.pred_meta.sem = proto::ShareSemantics::XorBytes;
  out.k0.pred_meta.out_bytes = 1;
  out.k1.pred_meta.out_bytes = 1;
  out.k0.cut_pred_meta = out.k0.pred_meta;
  out.k1.cut_pred_meta = out.k1.pred_meta;
  out.k0.coeff_meta.bit_order = bit_order;
  out.k1.coeff_meta.bit_order = bit_order;
  out.k0.coeff_meta.sem = proto::ShareSemantics::AddU64;
  out.k1.coeff_meta.sem = proto::ShareSemantics::AddU64;

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

  // No cutpoints for truncation new (single piece).

  // Coeff step: party0 carries payload, party1 zeros (backend convention)
  std::vector<uint64_t> base = compiled.coeff.base_payload_words;
  if (base.empty()) base.resize(static_cast<size_t>(compiled.coeff.out_words), 0);
  out.k0.base_coeff_share = base;
  out.k1.base_coeff_share.assign(base.size(), 0ull);
  std::vector<uint64_t> total_delta(base.size(), 0ull);
  for (const auto& d : compiled.coeff.deltas_words) {
    for (size_t j = 0; j < base.size() && j < d.size(); j++) {
      total_delta[j] = proto::add_mod(total_delta[j], d[j]);
    }
  }
  out.k0.total_delta_share = total_delta;
  out.k1.total_delta_share.assign(total_delta.size(), 0ull);

  // Beaver triples for Bool DAG (selectors/b2a). Count multiplicative nodes.
  size_t bit_mul = 0;
  for (const auto& piece : compiled.bool_per_piece) {
    for (const auto& b : piece) bit_mul += count_bool_mul(b);
  }
  bit_mul = std::max(bit_mul, static_cast<size_t>(compiled.ell)); // at least one per bool output
  size_t triple_need = bit_mul * std::max<size_t>(1, batch_N);

  // selectors_from_cutbits uses (cuts + 1) multiplications when cuts>0; here cuts==0.
  out.k0.triples.resize(triple_need);
  out.k1.triples.resize(triple_need);
  for (size_t i = 0; i < triple_need; i++) {
    uint64_t a = rng(), b = rng(), c = proto::mul_mod(a, b);
    auto [a0, a1] = split_add(a);
    auto [b0, b1] = split_add(b);
    auto [c0, c1] = split_add(c);
    out.k0.triples[i] = proto::BeaverTriple64Share{a0, b0, c0};
    out.k1.triples[i] = proto::BeaverTriple64Share{a1, b1, c1};
  }

  size_t bit_and = triple_need;  // reuse count for bit triples
  out.k0.bit_triples.resize(bit_and);
  out.k1.bit_triples.resize(bit_and);
  for (size_t i = 0; i < bit_and; i++) {
    uint8_t a = static_cast<uint8_t>(rng() & 1u);
    uint8_t b = static_cast<uint8_t>(rng() & 1u);
    uint8_t c = static_cast<uint8_t>(a & b);
    uint8_t a0 = static_cast<uint8_t>(rng() & 1u);
    uint8_t a1 = static_cast<uint8_t>(a ^ a0);
    uint8_t b0 = static_cast<uint8_t>(rng() & 1u);
    uint8_t b1 = static_cast<uint8_t>(b ^ b0);
    uint8_t c0 = static_cast<uint8_t>(rng() & 1u);
    uint8_t c1 = static_cast<uint8_t>(c ^ c0);
    out.k0.bit_triples[i] = proto::BeaverTripleBitShare{a0, b0, c0};
    out.k1.bit_triples[i] = proto::BeaverTripleBitShare{a1, b1, c1};
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
  if (k.pred_meta.sem != proto::ShareSemantics::XorBytes) {
    throw std::runtime_error("composite_eval_share_backend: predicate semantics not XOR bytes");
  }
  if (k.cut_pred_meta.sem != proto::ShareSemantics::XorBytes) {
    throw std::runtime_error("composite_eval_share_backend: selector predicate semantics not XOR bytes");
  }
  if (k.coeff_meta.sem != proto::ShareSemantics::AddU64) {
    throw std::runtime_error("composite_eval_share_backend: coeff semantics must be additive");
  }

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
    auto get_pred = [&](size_t idx) { return pv.get(idx); };
    // Cut predicates remain scalar bits_xor below; we won't fill bits_xor vector here.
    // Wrap shares already additive.
    // Build selectors below uses cut_bits_xor.
    std::vector<uint64_t> cut_bits_xor(k.cut_pred_keys.size(), 0);
    if (k.use_packed_cut && k.packed_cut_words > 0) {
      size_t key_bytes_cut = k.packed_cut_key.bytes.size();
      std::vector<uint8_t> cut_flat(key_bytes_cut);
      std::memcpy(cut_flat.data(), k.packed_cut_key.bytes.data(), key_bytes_cut);
      std::vector<uint64_t> cut_masks(static_cast<size_t>(k.packed_cut_words), 0);
      sb->eval_packed_lt_many(key_bytes_cut, cut_flat.data(), std::vector<uint64_t>{hatx},
                              compiled.coeff.n, k.packed_cut_words, cut_masks.data());
      PredViewPacked pc{cut_masks.data(), static_cast<size_t>(k.packed_cut_words)};
      for (size_t ci = 0; ci < k.cut_pred_keys.size(); ci++) cut_bits_xor[ci] = pc.get(ci);
    } else {
      for (size_t ci = 0; ci < k.cut_pred_keys.size(); ci++) {
        cut_bits_xor[ci] = proto::eval_pred_bit_share(backend, compiled.coeff.n, k.cut_pred_meta, k.cut_pred_keys[ci], hatx) & 1ull;
      }
    }
    auto coeff_table = build_coeff_table(compiled, k, /*add_deltas=*/(party == 0));
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
        uint64_t bx = gates::eval_bool_xor_view(compiled.bool_per_piece[p][static_cast<size_t>(j)], get_pred, k.wrap_share, mul, k.bit_triples.data(), &bit_idx) & 1ull;
        uint64_t badd = gates::b2a_bit(bx, party, mul);
        uint64_t term = mul.mul(selectors[p], badd);
        bools[static_cast<size_t>(j)] = proto::add_mod(bools[static_cast<size_t>(j)], term);
      }
    }
    std::vector<uint64_t> ys(compiled.r, 0);
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
    std::vector<uint64_t> out;
    out.reserve(static_cast<size_t>(compiled.r + compiled.ell));
    for (auto v : ys) out.push_back(v);
    for (auto v : bools) out.push_back(v);
    return out;
  } else {
    for (size_t i = 0; i < compiled.pred.queries.size(); i++) {
      int bits_in = (compiled.pred.queries[i].kind == compiler::RawPredKind::kLtU64) ? compiled.pred.n : compiled.pred.queries[i].f;
      bits_xor[i] = proto::eval_pred_bit_share(backend, bits_in, k.pred_meta, k.pred_keys[i], hatx);
    }
  }
  // Cut predicates for piece selectors
  std::vector<uint64_t> cut_bits_xor(k.cut_pred_keys.size(), 0);
  for (size_t ci = 0; ci < k.cut_pred_keys.size(); ci++) {
    cut_bits_xor[ci] = proto::eval_pred_bit_share(backend, compiled.coeff.n, k.cut_pred_meta, k.cut_pred_keys[ci], hatx) & 1ull;
  }
  auto coeff_table = build_coeff_table(compiled, k, /*add_deltas=*/(party == 0));
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
  bool is_trunc_gate = (compiled.gate_kind == compiler::GateKind::FaithfulTR ||
                        compiled.gate_kind == compiler::GateKind::FaithfulARS ||
                        compiled.gate_kind == compiler::GateKind::GapARS);
  if (!is_trunc_gate && dynamic_cast<proto::ClearBackend*>(&backend) != nullptr) {
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
  if (k.pred_meta.sem != proto::ShareSemantics::XorBytes) {
    throw std::runtime_error("composite_eval_batch_backend: predicate semantics not XOR bytes");
  }
  if (k.cut_pred_meta.sem != proto::ShareSemantics::XorBytes) {
    throw std::runtime_error("composite_eval_batch_backend: selector predicate semantics not XOR bytes");
  }
  if (k.coeff_meta.sem != proto::ShareSemantics::AddU64) {
    throw std::runtime_error("composite_eval_batch_backend: coeff semantics must be additive");
  }

  // Evaluate predicate bits in batch (packed mask if available)
  std::vector<uint64_t> pred_bits_xor;
  std::vector<uint64_t> pred_masks;
  std::vector<uint64_t> cut_masks;
  auto* sb = dynamic_cast<proto::SigmaFastBackend*>(&backend);
  if (sb && k.use_packed_pred && k.packed_pred_words > 0) {
    size_t key_bytes = k.packed_pred_key.bytes.size();
    std::vector<uint8_t> keys_flat(N * key_bytes);
    for (size_t i = 0; i < N; i++) {
      std::memcpy(keys_flat.data() + i * key_bytes, k.packed_pred_key.bytes.data(), key_bytes);
    }
    pred_masks.resize(N * static_cast<size_t>(k.packed_pred_words), 0);
    sb->eval_packed_lt_many(key_bytes, keys_flat.data(), std::vector<uint64_t>(in.hatx, in.hatx + N),
                            compiled.pred.n, k.packed_pred_words, pred_masks.data());
  } else {
    pred_bits_xor.resize(compiled.pred.queries.size() * N, 0);
    for (size_t qi = 0; qi < compiled.pred.queries.size(); qi++) {
      int bits_in = (compiled.pred.queries[qi].kind == compiler::RawPredKind::kLtU64) ? compiled.pred.n : compiled.pred.queries[qi].f;
      // pack keys_flat [N][key_bytes]
      size_t key_bytes = k.pred_keys[qi].bytes.size();
      std::vector<uint8_t> keys_flat(N * key_bytes);
      for (size_t i = 0; i < N; i++) {
        std::memcpy(keys_flat.data() + i * key_bytes, k.pred_keys[qi].bytes.data(), key_bytes);
      }
      size_t out_bytes = static_cast<size_t>(k.pred_meta.out_bytes);
      if (out_bytes == 0) throw std::runtime_error("pred_meta.out_bytes must be >0");
      std::vector<uint8_t> outs_flat(N * out_bytes);
      backend.eval_dcf_many_u64(bits_in, key_bytes, keys_flat.data(), std::vector<uint64_t>(in.hatx, in.hatx + N), k.pred_meta.out_bytes, outs_flat.data());
      for (size_t i = 0; i < N; i++) {
        if (k.pred_meta.sem == proto::ShareSemantics::XorBytes) {
          pred_bits_xor[qi * N + i] = static_cast<uint64_t>(outs_flat[i * out_bytes] & 1u);
        } else {
          uint64_t v = 0;
          std::memcpy(&v, outs_flat.data() + i * out_bytes, std::min<size_t>(8, out_bytes));
          pred_bits_xor[qi * N + i] = v & 1ull;
        }
      }
    }
  }

  // Cut predicate bits (XOR)
  std::vector<uint64_t> cut_bits_xor(k.cut_pred_keys.size() * N, 0);
  if (sb && k.use_packed_cut && k.packed_cut_words > 0) {
    size_t key_bytes = k.packed_cut_key.bytes.size();
    std::vector<uint8_t> keys_flat(N * key_bytes);
    for (size_t i = 0; i < N; i++) {
      std::memcpy(keys_flat.data() + i * key_bytes, k.packed_cut_key.bytes.data(), key_bytes);
    }
    cut_masks.resize(N * static_cast<size_t>(k.packed_cut_words), 0);
    sb->eval_packed_lt_many(key_bytes, keys_flat.data(), std::vector<uint64_t>(in.hatx, in.hatx + N),
                            compiled.coeff.n, k.packed_cut_words, cut_masks.data());
    for (size_t i = 0; i < N; i++) {
      PredViewPacked pc{cut_masks.data() + i * static_cast<size_t>(k.packed_cut_words),
                        static_cast<size_t>(k.packed_cut_words)};
      for (size_t ci = 0; ci < k.cut_pred_keys.size(); ci++) {
        cut_bits_xor[ci * N + i] = pc.get(ci);
      }
    }
  } else {
    for (size_t ci = 0; ci < k.cut_pred_keys.size(); ci++) {
      size_t key_bytes = k.cut_pred_keys[ci].bytes.size();
      std::vector<uint8_t> keys_flat(N * key_bytes);
      for (size_t i = 0; i < N; i++) {
        std::memcpy(keys_flat.data() + i * key_bytes, k.cut_pred_keys[ci].bytes.data(), key_bytes);
      }
      size_t out_bytes = static_cast<size_t>(k.cut_pred_meta.out_bytes);
      if (out_bytes == 0) throw std::runtime_error("cut_pred_meta.out_bytes must be >0");
      std::vector<uint8_t> outs_flat(N * out_bytes);
      backend.eval_dcf_many_u64(compiled.coeff.n, key_bytes, keys_flat.data(), std::vector<uint64_t>(in.hatx, in.hatx + N),
                                k.cut_pred_meta.out_bytes, outs_flat.data());
      for (size_t i = 0; i < N; i++) {
        if (k.cut_pred_meta.sem == proto::ShareSemantics::XorBytes) {
          cut_bits_xor[ci * N + i] = static_cast<uint64_t>(outs_flat[i * out_bytes] & 1u);
        } else {
          uint64_t v = 0;
          std::memcpy(&v, outs_flat.data() + i * out_bytes, std::min<size_t>(8, out_bytes));
          cut_bits_xor[ci * N + i] = v & 1ull;
        }
      }
    }
  }

  auto coeff_table = build_coeff_table(compiled, k, /*add_deltas=*/(party == 0));

  proto::BeaverMul64 mul_single{party, ch, k.triples, 0};
  if (sb && k.use_packed_pred && k.packed_pred_words > 0) {
    const size_t block_sz = 64;
    size_t pieces = coeff_table.size();
    size_t stride = compiled.degree + 1;
    std::vector<uint64_t> cut_i(k.cut_pred_keys.size(), 0);
    std::vector<uint64_t> selectors_block(pieces * block_sz, 0);
    std::vector<uint64_t> bool_block;
    std::vector<uint64_t> coeff_sel(compiled.coeff.out_words, 0);
    for (size_t blk = 0; blk < N; blk += block_sz) {
      size_t bsize = std::min(block_sz, N - blk);
      std::fill(selectors_block.begin(), selectors_block.end(), 0ull);
      for (size_t off = 0; off < bsize; off++) {
        size_t idx = blk + off;
        for (size_t ci = 0; ci < k.cut_pred_keys.size(); ci++) {
          cut_i[ci] = cut_bits_xor[ci * N + idx] & 1ull;
        }
        auto sels = selectors_from_cutbits(cut_i, party, mul_single);
        for (size_t p = 0; p < sels.size(); p++) {
          selectors_block[p * block_sz + off] = sels[p];
        }
      }
      for (size_t p = 0; p < pieces && p < compiled.bool_per_piece.size(); p++) {
        const auto& exprs = compiled.bool_per_piece[p];
        if (exprs.empty()) continue;
        gates::eval_bool_xor_packed_block_soa(exprs,
                                              pred_masks.data() + blk * static_cast<size_t>(k.packed_pred_words),
                                              static_cast<size_t>(k.packed_pred_words),
                                              bsize, k.wrap_share, party, mul_single,
                                              k.bit_triples.data(), 0, bool_block);
        for (int j = 0; j < compiled.ell; j++) {
          for (size_t off = 0; off < bsize; off++) {
            uint64_t term = mul_single.mul(selectors_block[p * block_sz + off],
                                           bool_block[static_cast<size_t>(j) * bsize + off]);
            size_t out_idx = (blk + off) * compiled.ell + static_cast<size_t>(j);
            out.bool_share[out_idx] = proto::add_mod(out.bool_share[out_idx], term);
          }
        }
      }
      for (size_t off = 0; off < bsize; off++) {
        std::fill(coeff_sel.begin(), coeff_sel.end(), 0ull);
        for (size_t p = 0; p < pieces; p++) {
          uint64_t sel = selectors_block[p * block_sz + off];
          if (sel == 0) continue;
          for (int j = 0; j < compiled.coeff.out_words; j++) {
            uint64_t term = mul_single.mul(sel, coeff_table[p][static_cast<size_t>(j)]);
            coeff_sel[static_cast<size_t>(j)] = proto::add_mod(coeff_sel[static_cast<size_t>(j)], term);
          }
        }
        uint64_t x_share = (party == 0) ? proto::sub_mod(in.hatx[blk + off], k.r_in_share)
                                        : proto::sub_mod(0ull, k.r_in_share);
        for (int j = 0; j < compiled.r; j++) {
          uint64_t acc = coeff_sel[static_cast<size_t>(j * stride + compiled.degree)];
          for (int d = compiled.degree - 1; d >= 0; d--) {
            acc = mul_single.mul(acc, x_share);
            acc = proto::add_mod(acc, coeff_sel[static_cast<size_t>(j * stride + d)]);
          }
          out.haty_share[(blk + off) * compiled.r + static_cast<size_t>(j)] =
              proto::add_mod(acc, k.r_out_share[static_cast<size_t>(j)]);
        }
      }
    }
    return out;
  }

  // Fallback scalar path (no packed predicates)
  size_t bit_idx = 0;
  std::vector<uint64_t> preds_i(compiled.pred.queries.size(), 0);
  std::vector<uint64_t> cut_i(k.cut_pred_keys.size(), 0);
  std::vector<uint64_t> selectors;
  std::vector<uint64_t> coeff_sel(compiled.coeff.out_words, 0);

  for (size_t i = 0; i < N; i++) {
    const auto& wrap_vars = k.wrap_share;

    auto get_pred = [&](size_t idx) -> uint64_t {
      return (idx < preds_i.size()) ? preds_i[idx] : 0ull;
    };

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
        uint64_t bx = gates::eval_bool_xor_view(compiled.bool_per_piece[p][static_cast<size_t>(j)], get_pred, wrap_vars, mul_single, k.bit_triples.data(), &bit_idx) & 1ull;
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
  tape_append_u64(v, k.r_hi_share);
  tape_append_u64(v, k.wrap_sign_share);
  tape_append_u32(v, static_cast<uint32_t>(k.extra_params.size()));
  for (auto x : k.extra_params) tape_append_u64(v, x);

  v.push_back(static_cast<uint8_t>(k.pred_meta.bit_order));
  v.push_back(static_cast<uint8_t>(k.pred_meta.sem));
  tape_append_u32(v, static_cast<uint32_t>(k.pred_meta.out_bytes));
  v.push_back(static_cast<uint8_t>(k.cut_pred_meta.bit_order));
  v.push_back(static_cast<uint8_t>(k.cut_pred_meta.sem));
  tape_append_u32(v, static_cast<uint32_t>(k.cut_pred_meta.out_bytes));
  v.push_back(static_cast<uint8_t>(k.coeff_meta.bit_order));
  v.push_back(static_cast<uint8_t>(k.coeff_meta.sem));

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
  need(8);
  k.r_hi_share = tape_read_u64(p);
  need(8);
  k.wrap_sign_share = tape_read_u64(p);
  need(4);
  uint32_t extra_len = tape_read_u32(p);
  need(static_cast<size_t>(extra_len) * 8);
  k.extra_params.resize(extra_len);
  for (uint32_t i = 0; i < extra_len; i++) k.extra_params[i] = tape_read_u64(p);
  need(2 + 2 + 4 + 4 + 2);
  k.pred_meta.bit_order = static_cast<proto::BitOrder>(*p++);
  k.pred_meta.sem = static_cast<proto::ShareSemantics>(*p++);
  k.pred_meta.out_bytes = static_cast<int>(tape_read_u32(p));
  k.cut_pred_meta.bit_order = static_cast<proto::BitOrder>(*p++);
  k.cut_pred_meta.sem = static_cast<proto::ShareSemantics>(*p++);
  k.cut_pred_meta.out_bytes = static_cast<int>(tape_read_u32(p));
  k.coeff_meta.bit_order = static_cast<proto::BitOrder>(*p++);
  k.coeff_meta.sem = static_cast<proto::ShareSemantics>(*p++);

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

// Read tape and evaluate (single element) using an existing compiled new description.
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
                                                               gates::PostProcHook& hook) {
  auto out = composite_eval_batch_backend(party, backend, ch, k, F, in);
  proto::BeaverMul64 mul{party, ch, k.triples, 0};
  hook.configure(k.compiled.layout);
  if (auto* relu = dynamic_cast<gates::ReluARSPostProc*>(&hook)) {
    relu->r_hi_share = k.r_hi_share;
    relu->wrap_sign_share = k.wrap_sign_share;
    if (relu->delta.empty() && !k.extra_params.empty()) {
      relu->delta = k.extra_params;
    }
  } else if (auto* tr = dynamic_cast<gates::FaithfulTruncPostProc*>(&hook)) {
    tr->r_hi_share = k.r_hi_share;
    tr->r_in = k.compiled.r_in;
  } else if (auto* ars = dynamic_cast<gates::FaithfulArsPostProc*>(&hook)) {
    ars->r_hi_share = k.r_hi_share;
    ars->r_in = k.compiled.r_in;
  } else if (auto* gap = dynamic_cast<gates::GapArsPostProc*>(&hook)) {
    gap->r_hi_share = k.r_hi_share;
    gap->r_in = k.compiled.r_in;
  }
  hook.run_batch(party, ch, mul, in.hatx, out.haty_share.data(),
                 static_cast<size_t>(k.compiled.r),
                 out.bool_share.data(), static_cast<size_t>(k.compiled.ell),
                 in.N, out.haty_share.data());
  return out;
}

}  // namespace gates
