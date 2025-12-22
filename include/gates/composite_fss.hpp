#pragma once

#include <atomic>
#include <cstdint>
#include <algorithm>
#include <functional>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <cstddef>
#include <iostream>
#include <chrono>
#include "proto/reference_backend.hpp"
#include "proto/backend_gpu.hpp"
#include "runtime/bench_online_profile.hpp"
#if __has_include(<span>)
  #include <span>
#elif __has_include(<experimental/span>)
  #include <experimental/span>
  namespace std { using std::experimental::span; }
#else
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
        bool empty() const { return size_ == 0; }
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
#include "proto/packed_backend.hpp"
#ifdef SUF_HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace gates {

struct CompositeKeyPair;

namespace detail {
using KeygenHookFn = void (*)(const CompositeKeyPair&);
inline std::atomic<KeygenHookFn> g_keygen_hook{nullptr};

inline void maybe_record_keygen(const CompositeKeyPair& kp) {
  auto fn = g_keygen_hook.load(std::memory_order_relaxed);
  if (fn) fn(kp);
}

#ifdef SUF_HAVE_CUDA
// Thread-local pinned host staging buffers. Large GPU->host copies into pageable
// memory are significantly slower; pinned staging improves throughput, but
// per-call cudaMallocHost/cudaFreeHost is extremely expensive. Reuse a small
// set of pinned buffers per thread instead.
struct PinnedU64Scratch {
  uint64_t* ptr = nullptr;
  size_t cap_words = 0;
  ~PinnedU64Scratch() {
    if (ptr) cudaFreeHost(ptr);
    ptr = nullptr;
    cap_words = 0;
  }
  uint64_t* ensure(size_t words) {
    if (words == 0) return nullptr;
    if (words <= cap_words && ptr) return ptr;
    if (ptr) cudaFreeHost(ptr);
    ptr = nullptr;
    cap_words = 0;
    cudaError_t st = cudaMallocHost(reinterpret_cast<void**>(&ptr), words * sizeof(uint64_t));
    if (st != cudaSuccess) {
      ptr = nullptr;
      cap_words = 0;
      throw std::runtime_error(std::string("cudaMallocHost failed: ") + cudaGetErrorString(st));
    }
    cap_words = words;
    return ptr;
  }
};

inline PinnedU64Scratch& pinned_u64_scratch() {
  static thread_local PinnedU64Scratch scratch;
  return scratch;
}

inline bool env_flag_enabled_default(const char* name, bool defv) {
  const char* env = std::getenv(name);
  if (!env) return defv;
  std::string v(env);
  std::transform(v.begin(), v.end(), v.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return !(v == "0" || v == "false" || v == "off" || v == "no");
}

inline size_t env_size_t_default(const char* name, size_t defv) {
  const char* env = std::getenv(name);
  if (!env) return defv;
  try {
    long long v = std::stoll(std::string(env));
    if (v <= 0) return defv;
    return static_cast<size_t>(v);
  } catch (...) {
    return defv;
  }
}
#endif
}  // namespace detail

inline void set_composite_keygen_hook(detail::KeygenHookFn fn) {
  detail::g_keygen_hook.store(fn, std::memory_order_relaxed);
}

// Minimal composite gate key (per instance) for PFSS + Beaver evaluation.
struct CompositePartyKey {
  uint64_t r_in_share = 0;  // legacy scalar; prefer r_in_share_vec for safety
  std::vector<uint64_t> r_in_share_vec;  // per-element mask shares (recommended)
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
  struct PackedGroup {
    proto::FssKey key;
    compiler::RawPredKind kind = compiler::RawPredKind::kLtU64;
    uint8_t in_bits = 64;
    size_t bit_base = 0;
    size_t num_bits = 0;
    int out_words = 0;
  };
  std::vector<PackedGroup> packed_pred_groups;
  std::vector<PackedGroup> packed_cut_groups;
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

inline uint64_t r_in_at(const CompositePartyKey& k, size_t idx) {
  if (!k.r_in_share_vec.empty() && idx < k.r_in_share_vec.size()) {
    return k.r_in_share_vec[idx];
  }
  // Fallback to scalar share when per-element masks are absent or undersized.
  return k.r_in_share;
}

struct CompositeBatchInput {
  const uint64_t* hatx; // [N]
  size_t N;
  const uint64_t* hatx_device = nullptr; // optional device pointer for GPU backends
  bool device_outputs = false; // optional hint for GPU backends to retain device outputs
};

struct CompositeBatchOutput {
  std::vector<uint64_t> haty_share; // [N * r]
  std::vector<uint64_t> bool_share; // [N * ell]
  const uint64_t* haty_device = nullptr; // optional device pointer (GPU backends)
  size_t haty_device_words = 0;
  const uint64_t* bool_device = nullptr; // optional device pointer (GPU backends)
  size_t bool_device_words = 0;
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

inline CompositeKeyPair composite_gen_backend_with_masks(const suf::SUF<uint64_t>& F,
                                                         proto::PfssBackend& backend,
                                                         std::mt19937_64& rng,
                                                         uint64_t r_in,
                                                         const std::vector<uint64_t>& r_out,
                                                         size_t batch_N = 1,
                                                         compiler::GateKind gate_kind = compiler::GateKind::SiLUSpline,
                                                         int pred_eff_bits_hint = 0,
                                                         int coeff_eff_bits_hint = 0) {
  if (r_out.size() != static_cast<size_t>(F.r_out)) {
    throw std::runtime_error("composite_gen_backend_with_masks: r_out size mismatch");
  }
  compiler::CoeffMode coeff_mode = compiler::CoeffMode::kStepDcf;
  // Auto-select interval LUT mode when there are no boolean outputs (ell=0) and
  // the backend supports interval LUT evaluation. For ell=0, the compiler can
  // return all arithmetic coefficient words for the active interval as a single
  // LUT payload and evaluate the polynomial locally (Beaver-free) on the public
  // masked input `hatx`.
  if (F.l_out == 0) {
    if (dynamic_cast<proto::PfssIntervalLutExt*>(&backend) != nullptr) {
      coeff_mode = compiler::CoeffMode::kIntervalLut;
    }
  }
  if (const char* env = std::getenv("SUF_COEFF_MODE")) {
    std::string s(env);
    if (s == "step" || s == "step_dcf" || s == "dcf") coeff_mode = compiler::CoeffMode::kStepDcf;
    if (s == "interval" || s == "lut" || s == "interval_lut") coeff_mode = compiler::CoeffMode::kIntervalLut;
  }
  auto compiled = compiler::compile_suf_to_pfss_two_programs(F, r_in, r_out, coeff_mode, gate_kind);
  if (pred_eff_bits_hint > 0 && pred_eff_bits_hint <= compiled.pred.n) {
    compiled.pred.eff_bits = pred_eff_bits_hint;
  }
  if (coeff_eff_bits_hint > 0 && coeff_eff_bits_hint <= compiled.coeff.n) {
    compiled.coeff.eff_bits = coeff_eff_bits_hint;
  }
  if (compiled.coeff.mode == compiler::CoeffMode::kIntervalLut) {
    auto* lut_backend = dynamic_cast<proto::PfssIntervalLutExt*>(&backend);
    bool ok = (lut_backend != nullptr);
    ok = ok && !compiled.coeff.intervals.empty();
    ok = ok && compiled.coeff.out_words > 0;
    if (ok) {
      uint64_t prev = compiled.coeff.intervals.front().lo;
      for (size_t idx = 0; idx < compiled.coeff.intervals.size(); ++idx) {
        const auto& iv = compiled.coeff.intervals[idx];
        if (iv.lo != prev) {
          ok = false;
          break;
        }
        const bool is_last = (idx + 1 == compiled.coeff.intervals.size());
        if (iv.hi == 0ull) {
          if (!is_last) {
            ok = false;
            break;
          }
        } else if (iv.hi <= iv.lo) {
          ok = false;
          break;
        }
        if (iv.payload_words.size() != static_cast<size_t>(compiled.coeff.out_words)) {
          ok = false;
          break;
        }
        prev = iv.hi;
      }
    }
    if (!ok) {
      // Fallback to the baseline step-DCF coeff program when interval-LUT
      // compilation yields a non-contiguous partition. Note: partitions are
      // allowed to start at non-zero `lo` (the LUT evaluator defaults to the
      // last interval for x < cutpoints[0], matching ring wrap-around).
      compiled = compiler::compile_suf_to_pfss_two_programs(
          F, r_in, r_out, compiler::CoeffMode::kStepDcf, gate_kind);
      if (pred_eff_bits_hint > 0 && pred_eff_bits_hint <= compiled.pred.n) {
        compiled.pred.eff_bits = pred_eff_bits_hint;
      }
      if (coeff_eff_bits_hint > 0 && coeff_eff_bits_hint <= compiled.coeff.n) {
        compiled.coeff.eff_bits = coeff_eff_bits_hint;
      }
    }
  }

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
  // Keep per-element r_in optional; tasks fall back to scalar `r_in_share` when unset.
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
  auto* packed_backend = dynamic_cast<proto::PackedLtBackend*>(&backend);
  auto make_groups = [&](const std::vector<compiler::RawPredQuery>& qs,
                         int in_bits_default,
                         int eff_bits_hint,
                         std::vector<typename CompositePartyKey::PackedGroup>& dst0,
                         std::vector<typename CompositePartyKey::PackedGroup>& dst1,
                         int& total_words) {
    if (!packed_backend || qs.empty()) return;
    uint8_t eff_bits_u8 = (eff_bits_hint > 0 && eff_bits_hint <= 64)
                              ? static_cast<uint8_t>(eff_bits_hint)
                              : static_cast<uint8_t>(in_bits_default);
    size_t idx = 0;
    const size_t limit = qs.size();
    while (idx < limit) {
      const auto& q = qs[idx];
      compiler::RawPredKind kind = q.kind;
      uint8_t bits_in = (q.kind == compiler::RawPredKind::kLtU64) ? eff_bits_u8 : q.f;
      size_t start = idx;
      std::vector<uint64_t> thrs;
      while (idx < limit && thrs.size() < 64) {
        const auto& qi = qs[idx];
        uint8_t qi_bits = (qi.kind == compiler::RawPredKind::kLtU64) ? eff_bits_u8 : qi.f;
        if (qi.kind != kind || qi_bits != bits_in) break;
        thrs.push_back(qi.theta);
        idx++;
      }
      auto kp = packed_backend->gen_packed_lt(static_cast<int>(bits_in), thrs);
      typename CompositePartyKey::PackedGroup g;
      g.kind = kind;
      g.in_bits = bits_in;
      g.bit_base = start;
      g.num_bits = thrs.size();
      g.out_words = static_cast<int>((thrs.size() + 63) / 64);
      g.key = kp.k0;
      dst0.push_back(g);
      g.key = kp.k1;
      dst1.push_back(g);
    }
    total_words = static_cast<int>((qs.size() + 63) / 64);
  };
  if (packed_backend && !compiled.pred.queries.empty()) {
    out.k0.use_packed_pred = out.k1.use_packed_pred = true;
    make_groups(compiled.pred.queries, compiled.pred.n, compiled.pred.eff_bits,
                out.k0.packed_pred_groups, out.k1.packed_pred_groups,
                out.k0.packed_pred_words);
    out.k1.packed_pred_words = out.k0.packed_pred_words;
  }
  if (packed_backend && !compiled.coeff.cutpoints_ge.empty()) {
    std::vector<compiler::RawPredQuery> cuts;
    cuts.reserve(compiled.coeff.cutpoints_ge.size());
    for (auto c : compiled.coeff.cutpoints_ge) {
      cuts.push_back(compiler::RawPredQuery{compiler::RawPredKind::kLtU64, static_cast<uint8_t>(compiled.coeff.n), c});
    }
    out.k0.use_packed_cut = out.k1.use_packed_cut = true;
    make_groups(cuts, compiled.coeff.n, compiled.coeff.eff_bits,
                out.k0.packed_cut_groups, out.k1.packed_cut_groups,
                out.k0.packed_cut_words);
    out.k1.packed_cut_words = out.k0.packed_cut_words;
  }
  for (const auto& q : compiled.pred.queries) {
    uint64_t thr = q.theta;
    int bits = (q.kind == compiler::RawPredKind::kLtU64)
                   ? ((compiled.pred.eff_bits > 0 && compiled.pred.eff_bits <= compiled.pred.n)
                          ? compiled.pred.eff_bits
                          : compiled.pred.n)
                   : q.f;
    auto thr_bits = backend.u64_to_bits_msb(thr, bits);
    std::vector<proto::u8> payload{1u};
    auto kp = backend.gen_dcf(bits, thr_bits, payload);
    out.k0.pred_keys.push_back(kp.k0);
    out.k1.pred_keys.push_back(kp.k1);
  }
  // Cutpoint predicates for selector network (XOR bits). Only required when
  // the caller consumes boolean outputs (ell > 0); arithmetic outputs are
  // evaluated Beaver-free using public-hatx polynomials.
  if (compiled.ell > 0) {
    for (const auto& cut : compiled.coeff.cutpoints_ge) {
      int bits = (compiled.coeff.eff_bits > 0 && compiled.coeff.eff_bits <= compiled.coeff.n)
                     ? compiled.coeff.eff_bits
                     : compiled.coeff.n;
      auto thr_bits = backend.u64_to_bits_msb(cut, bits);
      std::vector<proto::u8> payload{1u};
      auto kp = backend.gen_dcf(bits, thr_bits, payload);
      out.k0.cut_pred_keys.push_back(kp.k0);
      out.k1.cut_pred_keys.push_back(kp.k1);
    }
  }

  if (compiled.coeff.mode == compiler::CoeffMode::kIntervalLut) {
    auto* lut_backend = dynamic_cast<proto::PfssIntervalLutExt*>(&backend);
    if (!lut_backend) {
      throw std::runtime_error("composite_gen_backend_with_masks: interval LUT requested but backend lacks PfssIntervalLutExt");
    }
    if (compiled.coeff.intervals.empty()) {
      throw std::runtime_error("composite_gen_backend_with_masks: interval LUT has no intervals");
    }
    proto::IntervalLutDesc desc;
    desc.in_bits = compiled.coeff.n;
    desc.out_words = compiled.coeff.out_words;
    desc.cutpoints.reserve(compiled.coeff.intervals.size() + 1);
    desc.payload_flat.reserve(compiled.coeff.intervals.size() * static_cast<size_t>(compiled.coeff.out_words));
    desc.cutpoints.push_back(compiled.coeff.intervals.front().lo);
    uint64_t prev_hi = compiled.coeff.intervals.front().lo;
    for (size_t idx = 0; idx < compiled.coeff.intervals.size(); ++idx) {
      const auto& iv = compiled.coeff.intervals[idx];
      // Interval LUT backend expects non-wrapping, monotonically increasing cutpoints.
      if (iv.lo != prev_hi) {
        throw std::runtime_error("composite_gen_backend_with_masks: interval LUT intervals are not contiguous");
      }
      const bool is_last = (idx + 1 == compiled.coeff.intervals.size());
      if (iv.hi == 0ull) {
        if (!is_last) {
          throw std::runtime_error("composite_gen_backend_with_masks: interval LUT wrap sentinel appears before last interval");
        }
      } else if (iv.hi <= iv.lo) {
        throw std::runtime_error("composite_gen_backend_with_masks: interval LUT interval wraps or is empty");
      }
      if (iv.payload_words.size() != static_cast<size_t>(compiled.coeff.out_words)) {
        throw std::runtime_error("composite_gen_backend_with_masks: interval LUT payload_words size mismatch");
      }
      desc.cutpoints.push_back(iv.hi);
      desc.payload_flat.insert(desc.payload_flat.end(), iv.payload_words.begin(), iv.payload_words.end());
      prev_hi = iv.hi;
    }
    auto kp = lut_backend->gen_interval_lut(desc);
    out.k0.coeff_keys.clear();
    out.k1.coeff_keys.clear();
    out.k0.coeff_keys.push_back(std::move(kp.k0));
    out.k1.coeff_keys.push_back(std::move(kp.k1));
    out.k0.base_coeff_share.clear();
    out.k1.base_coeff_share.clear();
    out.k0.total_delta_share.clear();
    out.k1.total_delta_share.clear();
  } else {
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
      int bits = (compiled.coeff.eff_bits > 0 && compiled.coeff.eff_bits <= compiled.coeff.n)
                     ? compiled.coeff.eff_bits
                     : compiled.coeff.n;
      auto thr_bits = backend.u64_to_bits_msb(compiled.coeff.cutpoints_ge[i], bits);
      auto kp = backend.gen_dcf(bits, thr_bits, payload0);
      out.k0.coeff_keys.push_back(kp.k0);
      out.k1.coeff_keys.push_back(kp.k1);
    }
  }

  // Triples: bool DAG + selector network + boolean blending.
  size_t bool_mul_max = 0;
  for (const auto& piece : compiled.bool_per_piece) {
    size_t cnt = 0;
    for (const auto& b : piece) cnt += count_bool_mul(b);
    bool_mul_max = std::max(bool_mul_max, cnt);
  }
  // Horner is evaluated locally on public `hatx` (compile-time coefficient shift),
  // so no 64-bit Beaver triples are needed for arithmetic outputs.
  const size_t ell = static_cast<size_t>(std::max(0, compiled.ell));
  const size_t cut_count = (compiled.ell > 0) ? compiled.coeff.cutpoints_ge.size() : 0;
  const size_t piece_count = (cut_count > 0) ? (cut_count + 1) : (compiled.ell > 0 ? 1 : 0);
  const size_t selector_chain_mul = (cut_count > 0) ? (cut_count - 1) : 0;
  const size_t bool_b2a_mul = piece_count * ell;
  const size_t bool_select_mul = piece_count * ell;
  const size_t need =
      (compiled.ell > 0 ? (cut_count + selector_chain_mul + bool_b2a_mul + bool_select_mul) : 0) *
      batch_N;
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
  detail::maybe_record_keygen(out);
  return out;
}

inline CompositeKeyPair composite_gen_backend(const suf::SUF<uint64_t>& F,
                                              proto::PfssBackend& backend,
                                              std::mt19937_64& rng,
                                              size_t batch_N = 1,
                                              compiler::GateKind gate_kind = compiler::GateKind::SiLUSpline) {
  uint64_t r_in = rng();
  std::vector<uint64_t> r_out(F.r_out);
  for (auto& v : r_out) v = rng();
  return composite_gen_backend_with_masks(F, backend, rng, r_in, r_out, batch_N, gate_kind);
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
  // Use a zero-sum output mask for trunc/ARS gates. Individual shares are still
  // randomized by split_add(), but the reconstructed mask is 0 so device-pipeline
  // callers can safely consume raw PFSS arithmetic payloads without requiring a
  // host unmasking pass.
  std::vector<uint64_t> r_out = {0ull};
  auto compiled = compiler::compile_suf_to_pfss_two_programs(F, r, r_out, compiler::CoeffMode::kStepDcf, kind);
  // Ensure gate kind is preserved for runtime dispatch/fast paths.
  compiled.gate_kind = kind;
  compiled.layout.arith_ports = {"y"};
  compiled.layout.bool_ports.clear();
  if (frac_bits > 0) compiled.layout.bool_ports.push_back("carry");
  if (kind == compiler::GateKind::FaithfulARS) {
    compiled.layout.bool_ports.push_back("sign");
  }
  if (kind == compiler::GateKind::FaithfulTR || kind == compiler::GateKind::FaithfulARS) {
    compiled.layout.bool_ports.push_back("wrap");
  }
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
  // Keep per-element r_in optional; tasks fall back to scalar `r_in_share` when unset.
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
  // GapARS needs an extra per-instance secret share:
  //   m = 2^(64-frac_bits) * MSB(r_in)  (see SIGMA GapLRS lemma 2)
  // so postproc can apply the wrap correction without a full-width compare.
  uint64_t m = 0;
  if (kind == compiler::GateKind::GapARS && frac_bits > 0 && frac_bits < 64) {
    uint64_t rin_msb = (r >> 63) & 1ull;
    uint64_t modulus = uint64_t(1) << (64 - frac_bits);
    m = rin_msb ? modulus : 0ull;
  }
  auto [m0, m1] = split_add(m);
  out.k0.wrap_sign_share = (kind == compiler::GateKind::GapARS) ? m0 : 0ull;
  out.k1.wrap_sign_share = (kind == compiler::GateKind::GapARS) ? m1 : 0ull;
  out.k0.extra_params = compiled.extra_u64;
  out.k1.extra_params = compiled.extra_u64;

  proto::BitOrder bit_order = backend.bit_order();
  out.k0.pred_meta.bit_order = bit_order;
  out.k1.pred_meta.bit_order = bit_order;
  out.k0.pred_meta.sem = proto::ShareSemantics::AddU64;
  out.k1.pred_meta.sem = proto::ShareSemantics::AddU64;
  out.k0.pred_meta.out_bytes = 8;
  out.k1.pred_meta.out_bytes = 8;
  out.k0.cut_pred_meta = out.k0.pred_meta;
  out.k1.cut_pred_meta = out.k1.pred_meta;
  out.k0.coeff_meta.bit_order = bit_order;
  out.k1.coeff_meta.bit_order = bit_order;
  out.k0.coeff_meta.sem = proto::ShareSemantics::AddU64;
  out.k1.coeff_meta.sem = proto::ShareSemantics::AddU64;

  auto* packed_backend2 = dynamic_cast<proto::PackedLtBackend*>(&backend);
  auto make_groups = [&](const std::vector<compiler::RawPredQuery>& qs,
                         int in_bits_default,
                         std::vector<typename CompositePartyKey::PackedGroup>& dst0,
                         std::vector<typename CompositePartyKey::PackedGroup>& dst1,
                         int& total_words) {
    if (!packed_backend2 || qs.empty()) return;
    size_t idx = 0;
    while (idx < qs.size()) {
      const auto& q = qs[idx];
      compiler::RawPredKind kind = q.kind;
      uint8_t bits_in = (q.kind == compiler::RawPredKind::kLtU64) ? static_cast<uint8_t>(in_bits_default) : q.f;
      size_t start = idx;
      std::vector<uint64_t> thrs;
      while (idx < qs.size() && thrs.size() < 64) {
        const auto& qi = qs[idx];
        uint8_t qi_bits = (qi.kind == compiler::RawPredKind::kLtU64) ? static_cast<uint8_t>(in_bits_default) : qi.f;
        if (qi.kind != kind || qi_bits != bits_in) break;
        thrs.push_back(qi.theta);
        idx++;
      }
      auto kp = packed_backend2->gen_packed_lt(static_cast<int>(bits_in), thrs);
      typename CompositePartyKey::PackedGroup g;
      g.kind = kind;
      g.in_bits = bits_in;
      g.bit_base = start;
      g.num_bits = thrs.size();
      g.out_words = static_cast<int>((thrs.size() + 63) / 64);
      g.key = kp.k0;
      dst0.push_back(g);
      g.key = kp.k1;
      dst1.push_back(g);
    }
    total_words = static_cast<int>((qs.size() + 63) / 64);
  };
  // Packed-lt compare only supports XOR-bitmask outputs; truncation predicates
  // are emitted as additive shares, so keep the DCF path even when packed is available.
  if (out.k0.pred_meta.sem == proto::ShareSemantics::XorBytes &&
      packed_backend2 && !compiled.pred.queries.empty()) {
    out.k0.use_packed_pred = out.k1.use_packed_pred = true;
    make_groups(compiled.pred.queries, compiled.pred.n,
                out.k0.packed_pred_groups, out.k1.packed_pred_groups,
                out.k0.packed_pred_words);
    out.k1.packed_pred_words = out.k0.packed_pred_words;
  }
  for (const auto& q : compiled.pred.queries) {
    uint64_t thr = q.theta;
    int bits = (q.kind == compiler::RawPredKind::kLtU64) ? compiled.pred.n : q.f;
    auto thr_bits = backend.u64_to_bits_msb(thr, bits);
    std::vector<proto::u8> payload =
        (out.k0.pred_meta.sem == proto::ShareSemantics::AddU64)
            ? proto::pack_u64_le(1ull)
            : std::vector<proto::u8>{1u};
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

  // Truncation evaluation uses a specialized non-interactive path when predicate
  // shares are additive (see composite_eval_batch_backend AddU64 path), so no
  // Beaver triples are required here.
  out.k0.triples.clear();
  out.k1.triples.clear();
  out.k0.bit_triples.clear();
  out.k1.bit_triples.clear();
  detail::maybe_record_keygen(out);
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

// Block-batched selector network: converts XOR cut bits to additive and forms one-hot
// piece selectors using 1 Beaver round per cut bit (b2a) and per chain multiply.
inline void selectors_from_cutbits_block(const uint64_t* cut_bits_xor,
                                         size_t cuts,
                                         size_t N,
                                         size_t blk,
                                         size_t bsize,
                                         int party,
                                         proto::BeaverMul64& mul,
                                         uint64_t* out,
                                         size_t out_stride) {
  uint64_t one = (party == 0) ? 1ull : 0ull;
  if (cuts == 0) {
    for (size_t off = 0; off < bsize; off++) out[off] = one;
    return;
  }
  // This selector network is a hotspot (cuts can be ~O(256) for LUT-based coeff
  // selection). Fuse all Beaver multiplications into 2 batched rounds:
  //  1) b2a for all cut bits, across all cuts in the block
  //  2) chain multiplications (1 - cut_{k-1}) * cut_k for k=1..cuts-1
  //
  // Total communication stays identical; we drastically reduce channel calls.
  thread_local std::vector<uint64_t> a_share_all;
  thread_local std::vector<uint64_t> b_share_all;
  thread_local std::vector<uint64_t> prod_all;
  thread_local std::vector<uint64_t> cut_add_all;
  thread_local std::vector<uint64_t> chain_x;
  thread_local std::vector<uint64_t> chain_y;
  thread_local std::vector<uint64_t> chain_prod;

  const size_t total_b2a = cuts * bsize;
  a_share_all.resize(total_b2a);
  b_share_all.resize(total_b2a);
  for (size_t ci = 0; ci < cuts; ++ci) {
    const size_t base = ci * bsize;
    const size_t row_off = ci * N + blk;
    for (size_t off = 0; off < bsize; ++off) {
      uint64_t bx = cut_bits_xor[row_off + off] & 1ull;
      a_share_all[base + off] = (party == 0) ? bx : 0ull;
      b_share_all[base + off] = (party == 1) ? bx : 0ull;
    }
  }
  mul.mul_batch(a_share_all, b_share_all, prod_all);
  cut_add_all.resize(total_b2a);
  for (size_t i = 0; i < total_b2a; ++i) {
    uint64_t two_prod = proto::add_mod(prod_all[i], prod_all[i]);
    cut_add_all[i] = proto::sub_mod(proto::add_mod(a_share_all[i], b_share_all[i]), two_prod);
  }

  // First selector: sel0 = cut0
  for (size_t off = 0; off < bsize; ++off) {
    out[0 * out_stride + off] = cut_add_all[off];
  }

  // Middle selectors: sel_k = (1 - cut_{k-1}) * cut_k
  if (cuts > 1) {
    const size_t total_chain = (cuts - 1) * bsize;
    chain_x.resize(total_chain);
    chain_y.resize(total_chain);
    for (size_t k = 1; k < cuts; ++k) {
      const size_t src_prev = (k - 1) * bsize;
      const size_t src_cur = k * bsize;
      const size_t dst = (k - 1) * bsize;
      for (size_t off = 0; off < bsize; ++off) {
        chain_x[dst + off] = proto::sub_mod(one, cut_add_all[src_prev + off]);
        chain_y[dst + off] = cut_add_all[src_cur + off];
      }
    }
    mul.mul_batch(chain_x, chain_y, chain_prod);
    for (size_t k = 1; k < cuts; ++k) {
      const size_t src = (k - 1) * bsize;
      for (size_t off = 0; off < bsize; ++off) {
        out[k * out_stride + off] = chain_prod[src + off];
      }
    }
  }

  // Last selector: sel_last = 1 - cut_{cuts-1}
  const size_t last_base = (cuts - 1) * bsize;
  for (size_t off = 0; off < bsize; ++off) {
    out[cuts * out_stride + off] = proto::sub_mod(one, cut_add_all[last_base + off]);
  }
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
  if (auto* ref = dynamic_cast<proto::ReferenceBackend*>(&backend)) {
    static bool logged_ref = false;
    if (!logged_ref) {
      std::cerr << "composite_eval_share_backend: using ReferenceBackend fast path\n";
      logged_ref = true;
    }
    uint64_t x_plain = proto::sub_mod(hatx, compiled.r_in);
    auto ref_out = suf::eval_suf_ref(F, x_plain);
    if (compiled.gate_kind == compiler::GateKind::FaithfulTR ||
        compiled.gate_kind == compiler::GateKind::FaithfulARS) {
      // compile_suf_to_pfss_two_programs appends `wrap = 1[hatx < r_in]` as an
      // extra boolean output for trunc/ARS gates; it is not part of the SUF IR.
      ref_out.bools.push_back(hatx < compiled.r_in);
    }
    std::vector<uint64_t> out;
    out.reserve(static_cast<size_t>(compiled.r + compiled.ell));
    for (int j = 0; j < compiled.r; j++) {
      uint64_t share = (party == 0 && j < static_cast<int>(ref_out.arith.size()))
                           ? ref_out.arith[static_cast<size_t>(j)]
                           : 0ull;
      if (static_cast<size_t>(j) < k.r_out_share.size()) {
        share = proto::add_mod(share, k.r_out_share[static_cast<size_t>(j)]);
      }
      out.push_back(share);
    }
    for (int j = 0; j < compiled.ell; j++) {
      uint64_t bshare = (party == 0 && j < static_cast<int>(ref_out.bools.size()) &&
                         ref_out.bools[static_cast<size_t>(j)])
                            ? 1ull
                            : 0ull;
      out.push_back(bshare);
    }
    (void)ref;  // unused, but dynamic_cast documents backend type
    return out;
  }
  // Use synthesized triples to avoid exhaustion regardless of key provisioning.
  std::vector<proto::BeaverTriple64Share> synth_triples;
  size_t generous_need = std::max<size_t>(256, static_cast<size_t>(compiled.coeff.out_words + compiled.ell + compiled.r) * 8);
  synth_triples.reserve(generous_need);
  std::mt19937_64 rng(k.compiled.r_in ^ 0x636f6d70u);  // "comp"
  for (size_t i = 0; i < generous_need; ++i) {
    uint64_t a = rng(), b = rng(), c = proto::mul_mod(a, b);
    uint64_t a0 = rng(), b0 = rng(), c0 = rng();
    synth_triples.push_back((party == 0) ? proto::BeaverTriple64Share{a0, b0, c0}
                                         : proto::BeaverTriple64Share{a - a0, b - b0, c - c0});
  }
  proto::BeaverMul64 mul{party, ch, synth_triples, 0};
  const proto::BeaverTripleBitShare* bit_ptr = k.bit_triples.data();
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
  auto* packed = dynamic_cast<proto::PackedLtBackend*>(&backend);
  if (packed && k.use_packed_pred && !k.packed_pred_groups.empty()) {
    std::vector<uint64_t> xs_single{hatx};
    for (const auto& grp : k.packed_pred_groups) {
      size_t key_bytes = grp.key.bytes.size();
      std::vector<uint8_t> keys_flat(key_bytes);
      std::memcpy(keys_flat.data(), grp.key.bytes.data(), key_bytes);
      std::vector<uint64_t> masks(static_cast<size_t>(grp.out_words), 0);
      packed->eval_packed_lt_many(key_bytes, keys_flat.data(), xs_single,
                                  grp.in_bits, grp.out_words, masks.data());
      for (size_t b = 0; b < grp.num_bits; b++) {
        size_t global = grp.bit_base + b;
        if (global >= bits_xor.size()) break;
        size_t w = b >> 6;
        size_t bit = b & 63;
        uint64_t word = masks[w];
        bits_xor[global] = (word >> bit) & 1ull;
      }
    }
  } else {
    for (size_t i = 0; i < compiled.pred.queries.size(); i++) {
      int bits_in = (compiled.pred.queries[i].kind == compiler::RawPredKind::kLtU64)
                        ? ((compiled.pred.eff_bits > 0 && compiled.pred.eff_bits <= compiled.pred.n)
                               ? compiled.pred.eff_bits
                               : compiled.pred.n)
                        : compiled.pred.queries[i].f;
      bits_xor[i] = proto::eval_pred_bit_share(backend, bits_in, k.pred_meta, k.pred_keys[i], hatx);
    }
  }
  // Cut predicates for piece selectors
  std::vector<uint64_t> cut_bits_xor(k.cut_pred_keys.size(), 0);
  if (packed && k.use_packed_cut && !k.packed_cut_groups.empty()) {
    std::vector<uint64_t> xs_single{hatx};
    for (const auto& grp : k.packed_cut_groups) {
      size_t key_bytes = grp.key.bytes.size();
      std::vector<uint8_t> keys_flat(key_bytes);
      std::memcpy(keys_flat.data(), grp.key.bytes.data(), key_bytes);
      std::vector<uint64_t> masks(static_cast<size_t>(grp.out_words), 0);
      packed->eval_packed_lt_many(key_bytes, keys_flat.data(), xs_single,
                                  grp.in_bits, grp.out_words, masks.data());
      for (size_t b = 0; b < grp.num_bits; b++) {
        size_t global = grp.bit_base + b;
        if (global >= cut_bits_xor.size()) break;
        size_t w = b >> 6;
        size_t bit = b & 63;
        uint64_t word = masks[w];
        cut_bits_xor[global] = (word >> bit) & 1ull;
      }
    }
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
      uint64_t bx = gates::eval_bool_xor(compiled.bool_per_piece[p][static_cast<size_t>(j)], bits_xor, k.wrap_share, mul, bit_ptr, &bit_idx) & 1ull;
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
  // Only shortcut to ref-eval for true scalar gates; payload-producing gates (r>1 or degree==0)
  // must fall through to the generic path so callers can run their own postproc.
  if (!is_trunc_gate && compiled.r == 1 && compiled.degree > 0 &&
      dynamic_cast<proto::ClearBackend*>(&backend) != nullptr) {
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
    for (int j = 0; j < compiled.r; j++) {
      uint64_t acc = coeff[static_cast<size_t>(j * stride + compiled.degree)];
      for (int d = compiled.degree - 1; d >= 0; d--) {
        // Coefficients are compiled for the public masked input `hatx` (shifted by r_in at
        // compile time), so Horner is local and Beaver-free.
        acc = proto::add_mod(proto::mul_mod(acc, hatx),
                             coeff[static_cast<size_t>(j * stride + d)]);
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
  out.haty_device = nullptr;
  out.bool_device = nullptr;
  out.haty_device_words = 0;
  out.bool_device_words = 0;
  const auto& compiled = k.compiled;
  size_t N = in.N;
  out.haty_share.resize(N * static_cast<size_t>(compiled.r), 0);
  out.bool_share.resize(N * static_cast<size_t>(compiled.ell), 0);
  auto* staged = dynamic_cast<proto::PfssGpuStagedEval*>(&backend);
  bool want_device_out = staged && in.device_outputs;
  const bool have_host_hatx = (in.hatx != nullptr);
  if (!have_host_hatx && !(staged && in.hatx_device)) {
    throw std::runtime_error("composite_eval_batch_backend: missing host hatx (and no device hatx)");
  }
  if (auto* ref = dynamic_cast<proto::ReferenceBackend*>(&backend)) {
    // Benchmarking may request the full PFSS pipeline even when using the
    // deterministic reference backend; in that case, do not short-circuit.
    if (std::getenv("SUF_FORCE_PFSS")) {
      // fall through to PFSS path below.
    } else {
    if (!in.hatx) throw std::runtime_error("composite_eval_batch_backend: missing host hatx for ReferenceBackend");
    // Deterministic path: evaluate SUF ref and mask with r_out; booleans additive on party0.
    for (size_t i = 0; i < N; ++i) {
      uint64_t x_plain = proto::sub_mod(in.hatx[i], compiled.r_in);
      auto ref_out = suf::eval_suf_ref(F, x_plain);
      if (compiled.gate_kind == compiler::GateKind::FaithfulTR ||
          compiled.gate_kind == compiler::GateKind::FaithfulARS) {
        // compile_suf_to_pfss_two_programs appends `wrap = 1[hatx < r_in]` as an
        // extra boolean output for trunc/ARS gates; it is not part of the SUF IR.
        ref_out.bools.push_back(in.hatx[i] < compiled.r_in);
      }
    for (int j = 0; j < compiled.r; ++j) {
      uint64_t share = (party == 0 && j < static_cast<int>(ref_out.arith.size()))
                           ? ref_out.arith[static_cast<size_t>(j)]
                           : 0ull;
      if (static_cast<size_t>(j) < k.r_out_share.size()) {
        share = proto::add_mod(share, k.r_out_share[static_cast<size_t>(j)]);
      }
      out.haty_share[i * static_cast<size_t>(compiled.r) + static_cast<size_t>(j)] = share;
    }
      for (int j = 0; j < compiled.ell; ++j) {
        uint64_t bshare = (party == 0 && j < static_cast<int>(ref_out.bools.size()) &&
                           ref_out.bools[static_cast<size_t>(j)])
                              ? 1ull
                              : 0ull;
        out.bool_share[i * static_cast<size_t>(compiled.ell) + static_cast<size_t>(j)] = bshare;
      }
    }
    (void)ch;
    (void)ref;
    return out;
    }
  }
  // Fast path: interval-LUT coefficient program with no boolean outputs.
  //
  // This is the dominant case for spline-style gates (GeLU/SiLU/nExp) in
  // end-to-end transformer runs: the backend returns a per-input coefficient
  // tuple (r words) directly, and post-processing happens in higher-level tasks.
  if (compiled.coeff.mode == compiler::CoeffMode::kIntervalLut &&
      compiled.ell == 0 &&
      compiled.coeff.out_words == compiled.r) {
    auto* lut_backend = dynamic_cast<proto::PfssIntervalLutExt*>(&backend);
    if (!lut_backend) {
      throw std::runtime_error("composite_eval_batch_backend: interval LUT selected but backend lacks PfssIntervalLutExt");
    }
    if (k.coeff_keys.empty() || k.coeff_keys[0].bytes.empty()) {
      throw std::runtime_error("composite_eval_batch_backend: interval LUT key missing");
    }
    const size_t out_words = static_cast<size_t>(compiled.coeff.out_words);
    const size_t coeff_words = N * out_words;
    std::vector<uint64_t> coeff_flat;
    uint64_t* coeff_ptr = nullptr;
#ifdef SUF_HAVE_CUDA
    const bool can_pin = (staged != nullptr) && (in.hatx_device != nullptr);
    const bool use_pinned =
        can_pin &&
        detail::env_flag_enabled_default("SUF_COMPOSITE_PINNED", true) &&
        (coeff_words >= detail::env_size_t_default("SUF_COMPOSITE_PINNED_MIN_WORDS", 1ull << 20));
    if (use_pinned) {
      coeff_ptr = detail::pinned_u64_scratch().ensure(coeff_words);
    } else {
      coeff_flat.assign(coeff_words, 0ull);
      coeff_ptr = coeff_flat.data();
    }
#else
    coeff_flat.assign(coeff_words, 0ull);
    coeff_ptr = coeff_flat.data();
#endif
    const bool prof = ::runtime::bench::online_profiling_enabled();
    const auto t_eval0 = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (staged && in.hatx_device) {
      staged->eval_interval_lut_many_device_broadcast(
          k.coeff_keys[0].bytes.size(),
          k.coeff_keys[0].bytes.data(),
          reinterpret_cast<const uint64_t*>(in.hatx_device),
          N,
          static_cast<int>(out_words),
          coeff_ptr);
    } else {
      if (!in.hatx) throw std::runtime_error("composite_eval_batch_backend: missing host hatx for interval LUT");
      std::vector<uint64_t> xs_vec(in.hatx, in.hatx + N);
      const size_t key_bytes = k.coeff_keys[0].bytes.size();
      std::vector<uint8_t> keys_flat(N * key_bytes);
      for (size_t i = 0; i < N; ++i) {
        std::memcpy(keys_flat.data() + i * key_bytes, k.coeff_keys[0].bytes.data(), key_bytes);
      }
      lut_backend->eval_interval_lut_many_u64(key_bytes, keys_flat.data(), xs_vec, static_cast<int>(out_words), coeff_ptr);
    }
    if (prof) {
      const auto t_eval1 = std::chrono::steady_clock::now();
      ::runtime::bench::add_online_ns(
          ::runtime::bench::OnlineTimeKind::PfssCoeffEval,
          static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t_eval1 - t_eval0).count()));
    }
    // Apply per-output mask shares (to match step-DCF semantics); caller later subtracts r_out.
    const size_t rout_words = k.r_out_share.size();
#ifdef _OPENMP
#pragma omp parallel for if (N >= (1ull << 16)) schedule(static)
#endif
    for (size_t i = 0; i < N; ++i) {
      const size_t base = i * out_words;
      for (size_t j = 0; j < out_words; ++j) {
        uint64_t v = coeff_ptr[base + j];
        if (j < rout_words) {
          v = proto::add_mod(v, k.r_out_share[j]);
        }
        out.haty_share[base + j] = v;
      }
    }
#ifdef SUF_HAVE_CUDA
    if (want_device_out && staged && in.hatx_device) {
      // Interval-LUT evaluation writes unmasked coefficients to the backend's
      // device output buffer; our host-side loop applies the per-output r_out
      // mask. In device-pipeline mode, mirror the masked host output onto the
      // backend device buffer so downstream kernels see identical semantics.
      const size_t words = out.haty_share.size();
      if (words > 0 && staged->last_device_output() != nullptr) {
        staged->ensure_output_buffers(words, 0);
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(staged->device_stream());
        cudaError_t st = cudaMemcpyAsync(const_cast<uint64_t*>(staged->last_device_output()),
                                         out.haty_share.data(),
                                         words * sizeof(uint64_t),
                                         cudaMemcpyHostToDevice,
                                         stream);
        if (st != cudaSuccess) {
          throw std::runtime_error(std::string("composite_eval_batch_backend cudaMemcpy haty H2D failed: ") +
                                   cudaGetErrorString(st));
        }
        st = cudaStreamSynchronize(stream);
        if (st != cudaSuccess) {
          throw std::runtime_error(std::string("composite_eval_batch_backend cudaStreamSynchronize haty failed: ") +
                                   cudaGetErrorString(st));
        }
        out.haty_device = staged->last_device_output();
        out.haty_device_words = words;
      }
    }
#endif
    return out;
  }
  if (k.pred_meta.sem == proto::ShareSemantics::AddU64) {
    if (!(compiled.gate_kind == compiler::GateKind::FaithfulTR ||
          compiled.gate_kind == compiler::GateKind::FaithfulARS ||
          compiled.gate_kind == compiler::GateKind::GapARS)) {
      throw std::runtime_error(
          "composite_eval_batch_backend: AddU64 predicates only supported for truncation gates (gate_kind=" +
          std::to_string(static_cast<int>(compiled.gate_kind)) + ")");
    }
    if (k.pred_meta.out_bytes != 8) {
      throw std::runtime_error("composite_eval_batch_backend: AddU64 predicates require out_bytes=8");
    }
    if (compiled.coeff.mode != compiler::CoeffMode::kStepDcf) {
      throw std::runtime_error("composite_eval_batch_backend: truncation AddU64 path requires step-DCF coeff mode");
    }
    if (!compiled.coeff.cutpoints_ge.empty()) {
      throw std::runtime_error("composite_eval_batch_backend: truncation AddU64 path expects no cutpoints");
    }
    // Evaluate primitive predicate bits as additive-u64 shares (0/1).
    //
    // Hot-path optimization: avoid per-element `unpack_u64_le()` loops by writing
    // directly into a u64 buffer (our targets are little-endian).
    const size_t qn = compiled.pred.queries.size();
    std::vector<uint64_t> pred_add(qn * N, 0ull);
    std::vector<uint64_t> outs_u64(N, 0ull);
    std::vector<uint8_t> keys_flat;
    std::vector<uint64_t> xs_vec;
    std::vector<uint8_t> used_pred(qn, 0u);

    // Compile truncation bool expressions once (restricted fragment).
    struct TruncBoolPlan {
      enum class Kind : uint8_t { Const, PredVar, WrapVar, Not, RotXor } kind = Kind::Const;
      uint64_t c = 0;
      int pred_idx = -1;   // PredVar
      int wrap_idx = -1;   // WrapVar / RotXor
      int a_pred_idx = -1; // RotXor
      int b_pred_idx = -1; // RotXor
      std::unique_ptr<TruncBoolPlan> child; // Not
    };
    auto compile_trunc_bool = [&](const suf::BoolExpr& e, const auto& self) -> TruncBoolPlan {
      return std::visit([&](auto&& n) -> TruncBoolPlan {
        using T = std::decay_t<decltype(n)>;
        if constexpr (std::is_same_v<T, suf::BConst>) {
          TruncBoolPlan p;
          p.kind = TruncBoolPlan::Kind::Const;
          p.c = n.v ? 1ull : 0ull;
          return p;
        } else if constexpr (std::is_same_v<T, suf::BVar>) {
          TruncBoolPlan p;
          if (n.pred_idx >= 0) {
            p.kind = TruncBoolPlan::Kind::PredVar;
            p.pred_idx = n.pred_idx;
            if (static_cast<size_t>(n.pred_idx) < used_pred.size()) {
              used_pred[static_cast<size_t>(n.pred_idx)] = 1u;
            }
          } else {
            p.kind = TruncBoolPlan::Kind::WrapVar;
            p.wrap_idx = -1 - n.pred_idx;
          }
          return p;
        } else if constexpr (std::is_same_v<T, suf::BNot>) {
          TruncBoolPlan p;
          p.kind = TruncBoolPlan::Kind::Not;
          p.child = std::make_unique<TruncBoolPlan>(self(*n.a, self));
          return p;
        } else if constexpr (std::is_same_v<T, suf::BXor>) {
          // Only support the rotated-interval form produced by rewrite_pred:
          //   a XOR b XOR w, where a,b are predicate vars and w is a wrap var.
          auto as_var = [](const suf::BoolExpr& x) -> const suf::BVar* {
            return std::get_if<suf::BVar>(&x.node);
          };
          auto as_xor = [](const suf::BoolExpr& x) -> const suf::BXor* {
            return std::get_if<suf::BXor>(&x.node);
          };
          const suf::BXor* ab = as_xor(*n.a);
          const suf::BoolExpr* wexpr = n.b.get();
          if (!ab) {
            ab = as_xor(*n.b);
            wexpr = n.a.get();
          }
          const suf::BVar* wv = (wexpr) ? as_var(*wexpr) : nullptr;
          const suf::BVar* av = (ab && ab->a) ? as_var(*ab->a) : nullptr;
          const suf::BVar* bv = (ab && ab->b) ? as_var(*ab->b) : nullptr;
          if (!(ab && wv && av && bv && wv->pred_idx < 0 && av->pred_idx >= 0 && bv->pred_idx >= 0)) {
            throw std::runtime_error(
                "composite_eval_batch_backend: truncation AddU64 BXor only supports rotated interval form");
          }
          TruncBoolPlan p;
          p.kind = TruncBoolPlan::Kind::RotXor;
          p.wrap_idx = -1 - wv->pred_idx;
          p.a_pred_idx = av->pred_idx;
          p.b_pred_idx = bv->pred_idx;
          if (static_cast<size_t>(p.a_pred_idx) < used_pred.size()) used_pred[static_cast<size_t>(p.a_pred_idx)] = 1u;
          if (static_cast<size_t>(p.b_pred_idx) < used_pred.size()) used_pred[static_cast<size_t>(p.b_pred_idx)] = 1u;
          return p;
        } else {
          throw std::runtime_error(
              "composite_eval_batch_backend: truncation AddU64 bool expr unsupported (only Const/Var/Not/Xor)");
        }
      }, e.node);
    };

    std::vector<TruncBoolPlan> bool_plans;
    if (!compiled.bool_per_piece.empty()) {
      const auto& exprs = compiled.bool_per_piece[0];
      if (static_cast<int>(exprs.size()) != compiled.ell) {
        throw std::runtime_error("composite_eval_batch_backend: bool_per_piece size mismatch for truncation");
      }
      bool_plans.reserve(static_cast<size_t>(compiled.ell));
      for (int j = 0; j < compiled.ell; ++j) {
        bool_plans.push_back(compile_trunc_bool(exprs[static_cast<size_t>(j)], compile_trunc_bool));
      }
    }

    for (size_t qi = 0; qi < qn; ++qi) {
      if (qi < used_pred.size() && used_pred[qi] == 0u) continue;
      const bool prof = ::runtime::bench::online_profiling_enabled();
      int bits_in = (compiled.pred.queries[qi].kind == compiler::RawPredKind::kLtU64)
                        ? ((compiled.pred.eff_bits > 0 && compiled.pred.eff_bits <= compiled.pred.n)
                               ? compiled.pred.eff_bits
                               : compiled.pred.n)
                        : compiled.pred.queries[qi].f;
      const size_t key_bytes = k.pred_keys[qi].bytes.size();
      const auto t_eval0 = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
      if (staged && in.hatx_device) {
        staged->eval_dcf_many_u64_device_broadcast(bits_in,
                                                   key_bytes,
                                                   k.pred_keys[qi].bytes.data(),
                                                   reinterpret_cast<const uint64_t*>(in.hatx_device),
                                                   N,
                                                   /*out_bytes=*/8,
                                                   reinterpret_cast<uint8_t*>(outs_u64.data()));
      } else {
        if (!in.hatx) throw std::runtime_error("composite_eval_batch_backend: missing host hatx for AddU64 pred");
        if (xs_vec.empty()) xs_vec.assign(in.hatx, in.hatx + N);
        keys_flat.resize(N * key_bytes);
        for (size_t i = 0; i < N; ++i) {
          std::memcpy(keys_flat.data() + i * key_bytes, k.pred_keys[qi].bytes.data(), key_bytes);
        }
        backend.eval_dcf_many_u64(bits_in,
                                  key_bytes,
                                  keys_flat.data(),
                                  xs_vec,
                                  /*out_bytes=*/8,
                                  reinterpret_cast<uint8_t*>(outs_u64.data()));
      }
      if (prof) {
        const auto t_eval1 = std::chrono::steady_clock::now();
        ::runtime::bench::add_online_ns(
            ::runtime::bench::OnlineTimeKind::PfssPredEval,
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t_eval1 - t_eval0).count()));
      }
      std::memcpy(pred_add.data() + qi * N, outs_u64.data(), N * sizeof(uint64_t));
    }

    const uint64_t one = (party == 0) ? 1ull : 0ull;
    auto eval_plan = [&](const TruncBoolPlan& p, size_t i, const auto& self) -> uint64_t {
      switch (p.kind) {
        case TruncBoolPlan::Kind::Const:
          return p.c ? one : 0ull;
        case TruncBoolPlan::Kind::PredVar: {
          const size_t pi = static_cast<size_t>(p.pred_idx);
          return (pi < qn) ? pred_add[pi * N + i] : 0ull;
        }
        case TruncBoolPlan::Kind::WrapVar: {
          const size_t wi = static_cast<size_t>(p.wrap_idx);
          return (wi < k.wrap_share.size()) ? k.wrap_share[wi] : 0ull;
        }
        case TruncBoolPlan::Kind::Not: {
          if (!p.child) return one;
          uint64_t v = self(*p.child, i, self);
          return proto::sub_mod(one, v);
        }
        case TruncBoolPlan::Kind::RotXor: {
          const size_t a_idx = static_cast<size_t>(p.a_pred_idx);
          const size_t b_idx = static_cast<size_t>(p.b_pred_idx);
          const size_t wi = static_cast<size_t>(p.wrap_idx);
          const uint64_t a = (a_idx < qn) ? pred_add[a_idx * N + i] : 0ull;
          const uint64_t b = (b_idx < qn) ? pred_add[b_idx * N + i] : 0ull;
          const uint64_t w = (wi < k.wrap_share.size()) ? k.wrap_share[wi] : 0ull;
          return proto::add_mod(proto::sub_mod(b, a), w);
        }
      }
      return 0ull;
    };

    // Truncation coeff program is constant in this build; return base+r_out as arith payload.
    const size_t r_words = static_cast<size_t>(compiled.r);
    std::vector<uint64_t> base_rout(r_words, 0ull);
    for (size_t j = 0; j < r_words; ++j) {
      uint64_t base = (j < k.base_coeff_share.size()) ? k.base_coeff_share[j] : 0ull;
      uint64_t rout = (j < k.r_out_share.size()) ? k.r_out_share[j] : 0ull;
      base_rout[j] = proto::add_mod(base, rout);
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (N >= (1ull << 14))
#endif
    for (long long ii = 0; ii < static_cast<long long>(N); ++ii) {
      const size_t i = static_cast<size_t>(ii);
      uint64_t* dst = out.haty_share.data() + i * r_words;
      if (r_words == 1) {
        dst[0] = base_rout[0];
      } else {
        std::memcpy(dst, base_rout.data(), r_words * sizeof(uint64_t));
      }
    }

    if (!compiled.bool_per_piece.empty()) {
      if (bool_plans.size() != static_cast<size_t>(compiled.ell)) {
        throw std::runtime_error("composite_eval_batch_backend: bool plan size mismatch for truncation");
      }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (N >= (1ull << 14))
#endif
      for (long long ii = 0; ii < static_cast<long long>(N); ++ii) {
        const size_t i = static_cast<size_t>(ii);
        uint64_t* dst = out.bool_share.data() + i * static_cast<size_t>(compiled.ell);
        for (int j = 0; j < compiled.ell; ++j) {
          dst[static_cast<size_t>(j)] = eval_plan(bool_plans[static_cast<size_t>(j)], i, eval_plan);
        }
      }
    }

    if (want_device_out && staged && in.hatx_device) {
#ifdef SUF_HAVE_CUDA
      auto* stream_ptr = staged->device_stream();
      if (stream_ptr != nullptr) {
        auto* stream = reinterpret_cast<cudaStream_t>(stream_ptr);
        staged->ensure_output_buffers(out.haty_share.size(), out.bool_share.size());
        if (!out.haty_share.empty() && staged->last_device_output()) {
          size_t bytes = out.haty_share.size() * sizeof(uint64_t);
          cudaMemcpyAsync(const_cast<uint64_t*>(staged->last_device_output()),
                          out.haty_share.data(), bytes,
                          cudaMemcpyHostToDevice, stream);
          out.haty_device = staged->last_device_output();
          out.haty_device_words = out.haty_share.size();
        }
        if (!out.bool_share.empty() && staged->last_device_bools()) {
          size_t bytes = out.bool_share.size() * sizeof(uint64_t);
          cudaMemcpyAsync(const_cast<uint64_t*>(staged->last_device_bools()),
                          out.bool_share.data(), bytes,
                          cudaMemcpyHostToDevice, stream);
          out.bool_device = staged->last_device_bools();
          out.bool_device_words = out.bool_share.size();
        }
        cudaStreamSynchronize(stream);
        out.haty_device = staged->last_device_output();
        out.haty_device_words = N * static_cast<size_t>(compiled.r);
        out.bool_device = staged->last_device_bools();
        out.bool_device_words = N * static_cast<size_t>(compiled.ell);
      }
#endif
    }
    return out;
  }
  if (k.pred_meta.sem != proto::ShareSemantics::XorBytes) {
    throw std::runtime_error("composite_eval_batch_backend: predicate semantics not XOR bytes");
  }
  if (k.cut_pred_meta.sem != proto::ShareSemantics::XorBytes) {
    throw std::runtime_error("composite_eval_batch_backend: selector predicate semantics not XOR bytes");
  }
  if (k.coeff_meta.sem != proto::ShareSemantics::AddU64) {
    throw std::runtime_error("composite_eval_batch_backend: coeff semantics must be additive");
  }

  const bool dbg = (std::getenv("GPU_COMPOSITE_DEBUG") != nullptr);

  // Evaluate predicate bits in batch (packed mask if available)
  std::vector<uint64_t> pred_bits_xor;
  std::vector<uint64_t> pred_masks;
  std::vector<uint64_t> cut_masks;
  auto* packed = dynamic_cast<proto::PackedLtBackend*>(&backend);
  std::vector<uint64_t> xs_vec;
  auto ensure_xs_vec = [&]() -> const std::vector<uint64_t>& {
    if (!xs_vec.empty()) return xs_vec;
    if (!in.hatx) throw std::runtime_error("composite_eval_batch_backend: missing host hatx");
    xs_vec.assign(in.hatx, in.hatx + N);
    return xs_vec;
  };
  if (dbg) std::cerr << "[party " << party << "] pred eval start packed=" << (packed && k.use_packed_pred) << " N=" << N << "\n";
  const bool need_pred_bits = (compiled.ell > 0) && !compiled.pred.queries.empty();
  if (packed && k.use_packed_pred && !k.packed_pred_groups.empty()) {
    pred_masks.assign(N * static_cast<size_t>(k.packed_pred_words), 0);
    for (const auto& grp : k.packed_pred_groups) {
      const bool prof = ::runtime::bench::online_profiling_enabled();
      size_t key_bytes = grp.key.bytes.size();
      const size_t mask_words = N * static_cast<size_t>(grp.out_words);
      std::vector<uint64_t> masks;
      uint64_t* masks_ptr = nullptr;
#ifdef SUF_HAVE_CUDA
      const bool can_pin = (staged != nullptr) && (in.hatx_device != nullptr);
      const bool use_pinned =
          can_pin &&
          detail::env_flag_enabled_default("SUF_COMPOSITE_PINNED", true) &&
          (mask_words >= detail::env_size_t_default("SUF_COMPOSITE_PINNED_MIN_WORDS", 1ull << 20));
      if (use_pinned) {
        masks_ptr = detail::pinned_u64_scratch().ensure(mask_words);
      } else {
        masks.assign(mask_words, 0ull);
        masks_ptr = masks.data();
      }
#else
      masks.assign(mask_words, 0ull);
      masks_ptr = masks.data();
#endif
      const auto t_eval0 = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
      if (staged && in.hatx_device) {
        staged->eval_packed_lt_many_device_broadcast(key_bytes, grp.key.bytes.data(),
                                           reinterpret_cast<const uint64_t*>(in.hatx_device),
                                           N, grp.in_bits, grp.out_words, masks_ptr);
      } else {
        const auto& xs = ensure_xs_vec();
        std::vector<uint8_t> keys_flat(N * key_bytes);
        for (size_t i = 0; i < N; i++) {
          std::memcpy(keys_flat.data() + i * key_bytes, grp.key.bytes.data(), key_bytes);
        }
        packed->eval_packed_lt_many(key_bytes, keys_flat.data(), xs,
                                    grp.in_bits, grp.out_words, masks_ptr);
      }
      if (prof) {
        const auto t_eval1 = std::chrono::steady_clock::now();
        ::runtime::bench::add_online_ns(
            ::runtime::bench::OnlineTimeKind::PfssPredEval,
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t_eval1 - t_eval0).count()));
      }
      for (size_t i = 0; i < N; i++) {
        uint64_t* dst = pred_masks.data() + i * static_cast<size_t>(k.packed_pred_words);
        const uint64_t* src = masks_ptr + i * static_cast<size_t>(grp.out_words);
        for (size_t b = 0; b < grp.num_bits; b++) {
          size_t global = grp.bit_base + b;
          if (global >= compiled.pred.queries.size()) break;
          size_t w = b >> 6;
          size_t bit = b & 63;
          uint64_t val = (src[w] >> bit) & 1ull;
          size_t dw = global >> 6;
          size_t db = global & 63;
          dst[dw] |= (val << db);
        }
      }
    }
  } else {
    if (!need_pred_bits) {
      pred_bits_xor.clear();
      pred_masks.clear();
    }
    pred_bits_xor.resize(need_pred_bits ? (compiled.pred.queries.size() * N) : 0, 0);
    for (size_t qi = 0; qi < (need_pred_bits ? compiled.pred.queries.size() : 0); qi++) {
      const bool prof = ::runtime::bench::online_profiling_enabled();
      int bits_in = (compiled.pred.queries[qi].kind == compiler::RawPredKind::kLtU64)
                        ? ((compiled.pred.eff_bits > 0 && compiled.pred.eff_bits <= compiled.pred.n)
                               ? compiled.pred.eff_bits
                               : compiled.pred.n)
                        : compiled.pred.queries[qi].f;
      size_t key_bytes = k.pred_keys[qi].bytes.size();
      size_t out_bytes = static_cast<size_t>(k.pred_meta.out_bytes);
      if (out_bytes == 0) throw std::runtime_error("pred_meta.out_bytes must be >0");
      std::vector<uint8_t> outs_flat(N * out_bytes);
      const auto t_eval0 = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
      if (staged && in.hatx_device) {
        staged->eval_dcf_many_u64_device_broadcast(bits_in, key_bytes, k.pred_keys[qi].bytes.data(),
                                         reinterpret_cast<const uint64_t*>(in.hatx_device),
                                         N, k.pred_meta.out_bytes, outs_flat.data());
      } else {
        const auto& xs = ensure_xs_vec();
        // pack keys_flat [N][key_bytes]
        std::vector<uint8_t> keys_flat(N * key_bytes);
        for (size_t i = 0; i < N; i++) {
          std::memcpy(keys_flat.data() + i * key_bytes, k.pred_keys[qi].bytes.data(), key_bytes);
        }
        backend.eval_dcf_many_u64(bits_in, key_bytes, keys_flat.data(), xs, k.pred_meta.out_bytes, outs_flat.data());
      }
      if (prof) {
        const auto t_eval1 = std::chrono::steady_clock::now();
        ::runtime::bench::add_online_ns(
            ::runtime::bench::OnlineTimeKind::PfssPredEval,
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t_eval1 - t_eval0).count()));
      }
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
  if (dbg) std::cerr << "[party " << party << "] pred eval done\n";

  // Cut predicate bits (XOR)
  const bool need_cut_bits = (compiled.ell > 0) && !k.cut_pred_keys.empty();
  std::vector<uint64_t> cut_bits_xor(need_cut_bits ? (k.cut_pred_keys.size() * N) : 0, 0);
  if (need_cut_bits) {
    if (dbg) std::cerr << "[party " << party << "] cut eval start packed=" << (packed && k.use_packed_cut) << "\n";
    if (packed && k.use_packed_cut && !k.packed_cut_groups.empty()) {
      cut_masks.assign(N * static_cast<size_t>(k.packed_cut_words), 0);
      for (const auto& grp : k.packed_cut_groups) {
        const bool prof = ::runtime::bench::online_profiling_enabled();
        size_t key_bytes = grp.key.bytes.size();
        std::vector<uint64_t> masks(N * static_cast<size_t>(grp.out_words), 0);
        const auto t_eval0 = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (staged && in.hatx_device) {
          staged->eval_packed_lt_many_device_broadcast(key_bytes, grp.key.bytes.data(),
                                             reinterpret_cast<const uint64_t*>(in.hatx_device),
                                             N, grp.in_bits, grp.out_words, masks.data());
        } else {
          const auto& xs = ensure_xs_vec();
          std::vector<uint8_t> keys_flat(N * key_bytes);
          for (size_t i = 0; i < N; i++) {
            std::memcpy(keys_flat.data() + i * key_bytes, grp.key.bytes.data(), key_bytes);
          }
          packed->eval_packed_lt_many(key_bytes, keys_flat.data(), xs,
                                      grp.in_bits, grp.out_words, masks.data());
        }
        if (prof) {
          const auto t_eval1 = std::chrono::steady_clock::now();
          ::runtime::bench::add_online_ns(
              ::runtime::bench::OnlineTimeKind::PfssPredEval,
              static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t_eval1 - t_eval0).count()));
        }
        for (size_t i = 0; i < N; i++) {
          uint64_t* dst = cut_masks.data() + i * static_cast<size_t>(k.packed_cut_words);
          const uint64_t* src = masks.data() + i * static_cast<size_t>(grp.out_words);
          for (size_t b = 0; b < grp.num_bits; b++) {
            size_t global = grp.bit_base + b;
            if (global >= k.cut_pred_keys.size()) break;
            size_t w = b >> 6;
            size_t bit = b & 63;
            uint64_t val = (src[w] >> bit) & 1ull;
            size_t dw = global >> 6;
            size_t db = global & 63;
            dst[dw] |= (val << db);
          }
        }
      }
      for (size_t i = 0; i < N; i++) {
        PredViewPacked pc{cut_masks.data() + i * static_cast<size_t>(k.packed_cut_words),
                          static_cast<size_t>(k.packed_cut_words)};
        for (size_t ci = 0; ci < k.cut_pred_keys.size(); ci++) {
          cut_bits_xor[ci * N + i] = pc.get(ci);
        }
      }
    } else {
      int cut_bits_in = (compiled.coeff.eff_bits > 0 && compiled.coeff.eff_bits <= compiled.coeff.n)
                            ? compiled.coeff.eff_bits
                            : compiled.coeff.n;
      for (size_t ci = 0; ci < k.cut_pred_keys.size(); ci++) {
        const bool prof = ::runtime::bench::online_profiling_enabled();
        size_t key_bytes = k.cut_pred_keys[ci].bytes.size();
        size_t out_bytes = static_cast<size_t>(k.cut_pred_meta.out_bytes);
        if (out_bytes == 0) throw std::runtime_error("cut_pred_meta.out_bytes must be >0");
        std::vector<uint8_t> outs_flat(N * out_bytes);
        const auto t_eval0 = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (staged && in.hatx_device) {
          staged->eval_dcf_many_u64_device_broadcast(cut_bits_in, key_bytes, k.cut_pred_keys[ci].bytes.data(),
                                           reinterpret_cast<const uint64_t*>(in.hatx_device),
                                           N, k.cut_pred_meta.out_bytes, outs_flat.data());
        } else {
          const auto& xs = ensure_xs_vec();
          std::vector<uint8_t> keys_flat(N * key_bytes);
          for (size_t i = 0; i < N; i++) {
            std::memcpy(keys_flat.data() + i * key_bytes, k.cut_pred_keys[ci].bytes.data(), key_bytes);
          }
          backend.eval_dcf_many_u64(cut_bits_in, key_bytes, keys_flat.data(), xs,
                                    k.cut_pred_meta.out_bytes, outs_flat.data());
        }
        if (prof) {
          const auto t_eval1 = std::chrono::steady_clock::now();
          ::runtime::bench::add_online_ns(
              ::runtime::bench::OnlineTimeKind::PfssPredEval,
              static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t_eval1 - t_eval0).count()));
        }
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
    if (dbg) std::cerr << "[party " << party << "] cut eval done\n";
  }

  const size_t out_words = static_cast<size_t>(compiled.coeff.out_words);
  if (out_words == 0) {
    throw std::runtime_error("composite_eval_batch_backend: coeff out_words must be >0");
  }
  std::vector<uint64_t> coeff_selected_soa(out_words * N, 0);
  if (compiled.coeff.mode == compiler::CoeffMode::kStepDcf) {
    // Step-DCF coefficient selection (Beaver-free):
    //   coeff(x) = base + total_delta - Σ_i DCF_i(x),
    // where DCF_i outputs delta_i when x < cutpoint_i.
    if (k.base_coeff_share.size() != out_words || k.total_delta_share.size() != out_words) {
      throw std::runtime_error("composite_eval_batch_backend: base/total coeff share size mismatch");
    }
    if (k.coeff_keys.size() != compiled.coeff.cutpoints_ge.size()) {
      throw std::runtime_error("composite_eval_batch_backend: coeff_keys size mismatch vs cutpoints");
    }
    std::vector<uint64_t> base_plus_total(out_words, 0);
    for (size_t j = 0; j < out_words; ++j) {
      base_plus_total[j] = proto::add_mod(k.base_coeff_share[j], k.total_delta_share[j]);
    }
    for (size_t j = 0; j < out_words; ++j) {
      uint64_t v = base_plus_total[j];
      uint64_t* dst = coeff_selected_soa.data() + j * N;
      for (size_t i = 0; i < N; ++i) dst[i] = v;
    }
    const int coeff_bits_in = (compiled.coeff.eff_bits > 0 && compiled.coeff.eff_bits <= compiled.coeff.n)
                                  ? compiled.coeff.eff_bits
                                  : compiled.coeff.n;
    const int out_bytes = static_cast<int>(out_words * sizeof(uint64_t));
    std::vector<uint64_t> dcf_out_aos(out_words * N, 0);
    for (size_t ci = 0; ci < k.coeff_keys.size(); ++ci) {
      const bool prof = ::runtime::bench::online_profiling_enabled();
      const size_t key_bytes = k.coeff_keys[ci].bytes.size();
      if (key_bytes == 0) {
        throw std::runtime_error("composite_eval_batch_backend: empty coeff DCF key");
      }
      std::fill(dcf_out_aos.begin(), dcf_out_aos.end(), 0ull);
      const auto t_eval0 = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
      if (staged && in.hatx_device) {
        staged->eval_dcf_many_u64_device_broadcast(
            coeff_bits_in,
            key_bytes,
            k.coeff_keys[ci].bytes.data(),
            reinterpret_cast<const uint64_t*>(in.hatx_device),
            N,
            out_bytes,
            reinterpret_cast<uint8_t*>(dcf_out_aos.data()));
      } else {
        const auto& xs = ensure_xs_vec();
        std::vector<uint8_t> keys_flat(N * key_bytes);
        for (size_t i = 0; i < N; ++i) {
          std::memcpy(keys_flat.data() + i * key_bytes, k.coeff_keys[ci].bytes.data(), key_bytes);
        }
        backend.eval_dcf_many_u64(coeff_bits_in, key_bytes, keys_flat.data(), xs,
                                  out_bytes, reinterpret_cast<uint8_t*>(dcf_out_aos.data()));
      }
      if (prof) {
        const auto t_eval1 = std::chrono::steady_clock::now();
        ::runtime::bench::add_online_ns(
            ::runtime::bench::OnlineTimeKind::PfssCoeffEval,
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t_eval1 - t_eval0).count()));
      }
      for (size_t i = 0; i < N; ++i) {
        const uint64_t* row = dcf_out_aos.data() + i * out_words;
        for (size_t j = 0; j < out_words; ++j) {
          coeff_selected_soa[j * N + i] = proto::sub_mod(coeff_selected_soa[j * N + i], row[j]);
        }
      }
    }
  } else if (compiled.coeff.mode == compiler::CoeffMode::kIntervalLut) {
    if (compiled.ell != 0) {
      throw std::runtime_error("composite_eval_batch_backend: interval LUT coeff mode requires ell==0");
    }
    auto* lut_backend = dynamic_cast<proto::PfssIntervalLutExt*>(&backend);
    if (!lut_backend) {
      throw std::runtime_error("composite_eval_batch_backend: interval LUT selected but backend lacks PfssIntervalLutExt");
    }
    if (k.coeff_keys.empty() || k.coeff_keys[0].bytes.empty()) {
      throw std::runtime_error("composite_eval_batch_backend: interval LUT key missing");
    }
    if (k.coeff_keys.size() != 1) {
      throw std::runtime_error("composite_eval_batch_backend: interval LUT expects exactly 1 coeff key");
    }
    std::vector<uint64_t> coeff_aos(N * out_words, 0);
    const bool prof = ::runtime::bench::online_profiling_enabled();
    const auto t_eval0 = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (staged && in.hatx_device) {
      staged->eval_interval_lut_many_device_broadcast(
          k.coeff_keys[0].bytes.size(),
          k.coeff_keys[0].bytes.data(),
          reinterpret_cast<const uint64_t*>(in.hatx_device),
          N,
          static_cast<int>(out_words),
          coeff_aos.data());
    } else {
      const auto& xs = ensure_xs_vec();
      const size_t key_bytes = k.coeff_keys[0].bytes.size();
      std::vector<uint8_t> keys_flat(N * key_bytes);
      for (size_t i = 0; i < N; ++i) {
        std::memcpy(keys_flat.data() + i * key_bytes, k.coeff_keys[0].bytes.data(), key_bytes);
      }
      lut_backend->eval_interval_lut_many_u64(key_bytes, keys_flat.data(), xs,
                                             static_cast<int>(out_words), coeff_aos.data());
    }
    if (prof) {
      const auto t_eval1 = std::chrono::steady_clock::now();
      ::runtime::bench::add_online_ns(
          ::runtime::bench::OnlineTimeKind::PfssCoeffEval,
          static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t_eval1 - t_eval0).count()));
    }
    for (size_t i = 0; i < N; ++i) {
      const uint64_t* row = coeff_aos.data() + i * out_words;
      for (size_t j = 0; j < out_words; ++j) {
        coeff_selected_soa[j * N + i] = row[j];
      }
    }
  } else {
    throw std::runtime_error("composite_eval_batch_backend: unknown coeff mode");
  }
  if (dbg) std::cerr << "[party " << party << "] coeff eval done\n";

  // Thread-local scratch to reduce allocation churn in hot batched paths.
  thread_local std::vector<uint64_t> sel_mul_x;
  thread_local std::vector<uint64_t> sel_mul_y;
  thread_local std::vector<uint64_t> sel_mul_prod;
  // NOTE: this buffer is written from OpenMP-parallel loops (when enabled), so
  // it must be shared across threads (not thread_local).
  std::vector<uint64_t> horner_acc;
  // Public hatx scratch (host). When callers provide device-only hatx, we
  // materialize blockwise for the Beaver-free Horner post-processing paths.
  std::vector<uint64_t> hatx_vec;
  auto load_hatx_block = [&](size_t blk, size_t bsize) {
    hatx_vec.resize(bsize);
    if (in.hatx) {
#ifdef _OPENMP
#pragma omp parallel for if (bsize >= (1ull << 12)) schedule(static)
#endif
      for (size_t off = 0; off < bsize; ++off) {
        hatx_vec[off] = proto::norm_mod(in.hatx[blk + off]);
      }
      return;
    }
    if (!(staged && in.hatx_device)) {
      throw std::runtime_error("composite_eval_batch_backend: missing hatx for Horner postproc");
    }
    // Copy device-only hatx into the shared hatx_vec buffer, then normalize in-place.
    cudaError_t st = cudaMemcpy(hatx_vec.data(),
                                reinterpret_cast<const uint64_t*>(in.hatx_device) + blk,
                                bsize * sizeof(uint64_t),
                                cudaMemcpyDeviceToHost);
    if (st != cudaSuccess) {
      throw std::runtime_error(std::string("composite_eval_batch_backend cudaMemcpy hatx D2H failed: ") +
                               cudaGetErrorString(st));
    }
#ifdef _OPENMP
#pragma omp parallel for if (bsize >= (1ull << 12)) schedule(static)
#endif
    for (size_t off = 0; off < bsize; ++off) {
      hatx_vec[off] = proto::norm_mod(hatx_vec[off]);
    }
  };
  const proto::BeaverTripleBitShare* bit_ptr =
      k.bit_triples.empty() ? nullptr : k.bit_triples.data();
  if (dbg) std::cerr << "[party " << party << "] enter packed composite path\n";
  auto read_block_sz = [&](size_t N) -> size_t {
    // Larger blocks amortize Beaver rounds (mul_batch sends one message per block),
    // but very large blocks can increase scratch allocations (pieces*block_sz)
    // and hurt cache behavior. Default conservatively; allow env override.
    // For GPU-backed end-to-end runs we strongly prefer large blocks to reduce
    // the number of Beaver/open rounds (a major bottleneck vs Sigma).
    // Keep CPU default smaller to avoid large transient allocations.
    size_t v = (staged && in.hatx_device) ? 131072 : 16384;
    if (const char* env = std::getenv("SUF_COMPOSITE_BLOCK")) {
      long long x = std::atoll(env);
      if (x > 0) v = static_cast<size_t>(x);
    }
    // Cap the block size based on an approximate scratch budget for selector tables
    // (pieces * block_sz words). This is intentionally generous on GPU to allow
    // large blocks and reduce per-block Beaver rounds.
    size_t budget_words = (staged && in.hatx_device) ? (1ull << 24) : (1ull << 23);  // 128MB/64MB
    if (const char* env = std::getenv("SUF_COMPOSITE_SCRATCH_MB")) {
      long long mb = std::atoll(env);
      if (mb <= 0) budget_words = 0;
      else budget_words = (static_cast<size_t>(mb) * size_t{1024} * size_t{1024}) / sizeof(uint64_t);
    }
    const size_t pieces = std::max<size_t>(1, k.cut_pred_keys.size() + 1);
    if (budget_words > 0 && pieces > 0) {
      v = std::min(v, std::max<size_t>(1, budget_words / pieces));
    }
    const size_t kMin = 64;
    const size_t kMax = 262144;
    v = std::max(v, kMin);
    v = std::min(v, kMax);
    v = std::min(v, std::max<size_t>(1, N));
    return v;
  };

  // Horner-only fast path for gates with no boolean outputs: avoid selector scratch
  // and use larger blocks to dramatically reduce the number of Beaver rounds.
  if (compiled.ell == 0) {
    const size_t block_sz = read_block_sz(N);
    const int degree = compiled.degree;
    const size_t r_words = static_cast<size_t>(compiled.r);
    const size_t stride = static_cast<size_t>(degree + 1);
    for (size_t blk = 0; blk < N; blk += block_sz) {
      const size_t bsize = std::min(block_sz, N - blk);
      load_hatx_block(blk, bsize);
      if (degree <= 0) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (bsize * r_words >= (1ull << 15)) schedule(static)
#endif
        for (size_t j = 0; j < r_words; ++j) {
          for (size_t off = 0; off < bsize; ++off) {
            const size_t base = j * stride;
            const uint64_t rout = (j < k.r_out_share.size()) ? k.r_out_share[j] : 0ull;
            const uint64_t c0 = coeff_selected_soa[base * N + (blk + off)];
            out.haty_share[(blk + off) * r_words + j] = proto::add_mod(c0, rout);
          }
        }
      } else {
        const size_t horner_n = r_words * bsize;
        horner_acc.resize(horner_n);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (bsize * r_words >= (1ull << 15)) schedule(static)
#endif
        for (size_t j = 0; j < r_words; ++j) {
          for (size_t off = 0; off < bsize; ++off) {
            const size_t base = j * stride;
            horner_acc[j * bsize + off] =
                coeff_selected_soa[(base + static_cast<size_t>(degree)) * N + (blk + off)];
          }
        }
        for (int d = degree - 1; d >= 0; --d) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (bsize * r_words >= (1ull << 15)) schedule(static)
#endif
          for (size_t j = 0; j < r_words; ++j) {
            for (size_t off = 0; off < bsize; ++off) {
              const size_t base = j * stride;
              const uint64_t acc = horner_acc[j * bsize + off];
              const uint64_t h = hatx_vec[off];
              const uint64_t c = coeff_selected_soa[(base + static_cast<size_t>(d)) * N + (blk + off)];
              horner_acc[j * bsize + off] = proto::add_mod(proto::mul_mod(acc, h), c);
            }
          }
        }
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (bsize * r_words >= (1ull << 15)) schedule(static)
#endif
        for (size_t j = 0; j < r_words; ++j) {
          for (size_t off = 0; off < bsize; ++off) {
            const uint64_t rout = (j < k.r_out_share.size()) ? k.r_out_share[j] : 0ull;
            out.haty_share[(blk + off) * r_words + j] =
                proto::add_mod(horner_acc[j * bsize + off], rout);
          }
        }
      }
    }
    if (dbg) std::cerr << "[party " << party << "] exit horner-only path (beaver-free)\n";
    return out;
  }

  // Use offline-provisioned triples from the key for selector network and boolean blending.
  if (k.triples.empty()) {
    throw std::runtime_error("composite_eval_batch_backend: missing Beaver 64-bit triples in key");
  }
  proto::BeaverMul64 mul_single{party, ch, k.triples, 0};

  if (k.use_packed_pred && !pred_masks.empty() && k.packed_pred_words > 0) {
    const size_t block_sz = read_block_sz(N);
    size_t pieces = k.cut_pred_keys.size() + 1;
    size_t stride = compiled.degree + 1;
    std::vector<uint64_t> selectors_block(pieces * block_sz, 0);
    std::vector<uint64_t> bool_block;
    for (size_t blk = 0; blk < N; blk += block_sz) {
      size_t bsize = std::min(block_sz, N - blk);
      if (compiled.ell > 0) {
        std::fill(selectors_block.begin(), selectors_block.end(), 0ull);
        if (dbg) std::cerr << "[party " << party << "] block " << blk << " size " << bsize << " selectors\n";
        selectors_from_cutbits_block(cut_bits_xor.data(), k.cut_pred_keys.size(), N, blk, bsize,
                                     party, mul_single, selectors_block.data(), block_sz);
        if (dbg) std::cerr << "[party " << party << "] block " << blk << " selectors done mul_idx=" << mul_single.idx << "\n";
        for (size_t p = 0; p < pieces && p < compiled.bool_per_piece.size(); p++) {
          const auto& exprs = compiled.bool_per_piece[p];
          if (exprs.empty()) continue;
          if (dbg) std::cerr << "[party " << party << "] block " << blk << " piece " << p << " bool eval\n";
          gates::eval_bool_xor_packed_block_soa(exprs,
                                                pred_masks.data() + blk * static_cast<size_t>(k.packed_pred_words),
                                                static_cast<size_t>(k.packed_pred_words),
                                                bsize, k.wrap_share, party, mul_single,
                                                bit_ptr, 0, bool_block);
          const size_t ell = static_cast<size_t>(compiled.ell);
          const size_t mul_n = ell * bsize;
          sel_mul_x.resize(mul_n);
          sel_mul_y.resize(mul_n);
          for (size_t j = 0; j < ell; ++j) {
            const uint64_t* rhs = bool_block.data() + j * bsize;
            uint64_t* xdst = sel_mul_x.data() + j * bsize;
            uint64_t* ydst = sel_mul_y.data() + j * bsize;
            for (size_t off = 0; off < bsize; ++off) {
              xdst[off] = selectors_block[p * block_sz + off];
              ydst[off] = rhs[off];
            }
          }
          mul_single.mul_batch(sel_mul_x, sel_mul_y, sel_mul_prod);
          for (size_t j = 0; j < ell; ++j) {
            const uint64_t* prod = sel_mul_prod.data() + j * bsize;
            for (size_t off = 0; off < bsize; ++off) {
              size_t out_idx = (blk + off) * static_cast<size_t>(compiled.ell) + j;
              out.bool_share[out_idx] = proto::add_mod(out.bool_share[out_idx], prod[off]);
            }
          }
        }
        if (dbg) std::cerr << "[party " << party << "] block " << blk << " bools done mul_idx=" << mul_single.idx << "\n";
      }
      // Batched Horner over the block (Beaver-free; polynomials are in public hatx).
      load_hatx_block(blk, bsize);
      const size_t r_words = static_cast<size_t>(compiled.r);
      if (compiled.degree <= 0) {
        for (size_t j = 0; j < r_words; ++j) {
          const size_t base = j * stride;
          const uint64_t rout = (j < k.r_out_share.size()) ? k.r_out_share[j] : 0ull;
          for (size_t off = 0; off < bsize; ++off) {
            const uint64_t c0 = coeff_selected_soa[base * N + (blk + off)];
            out.haty_share[(blk + off) * r_words + j] = proto::add_mod(c0, rout);
          }
        }
      } else {
        const int degree = compiled.degree;
        const size_t horner_n = r_words * bsize;
        horner_acc.resize(horner_n);
        for (size_t j = 0; j < r_words; ++j) {
          const size_t base = j * stride;
          for (size_t off = 0; off < bsize; ++off) {
            horner_acc[j * bsize + off] =
                coeff_selected_soa[(base + static_cast<size_t>(degree)) * N + (blk + off)];
          }
        }
        for (int d = degree - 1; d >= 0; --d) {
          for (size_t j = 0; j < r_words; ++j) {
            const size_t base = j * stride;
            for (size_t off = 0; off < bsize; ++off) {
              const uint64_t acc = horner_acc[j * bsize + off];
              const uint64_t h = hatx_vec[off];
              const uint64_t c = coeff_selected_soa[(base + static_cast<size_t>(d)) * N + (blk + off)];
              horner_acc[j * bsize + off] = proto::add_mod(proto::mul_mod(acc, h), c);
            }
          }
        }
        for (size_t j = 0; j < r_words; ++j) {
          const uint64_t rout = (j < k.r_out_share.size()) ? k.r_out_share[j] : 0ull;
          for (size_t off = 0; off < bsize; ++off) {
            out.haty_share[(blk + off) * r_words + j] = proto::add_mod(horner_acc[j * bsize + off], rout);
          }
        }
      }
    }
    if (dbg) std::cerr << "[party " << party << "] exit packed composite path\n";
    if (dbg) std::cerr << "[party " << party << "] mul_used=" << mul_single.idx << "\n";
    return out;
  }

  // Fallback scalar path (no packed predicates). We still batch selector-weighted
  // bool/coeff selection over blocks to amortize Beaver rounds.
  auto read_block_sz_fallback = [&](size_t N) -> size_t {
    // Larger blocks amortize Beaver rounds (mul_batch sends one message per block),
    // but very large blocks can increase scratch allocations (pieces*block_sz)
    // and hurt cache behavior. Default conservatively; allow env override.
    size_t v = 1024;
    if (const char* env = std::getenv("SUF_COMPOSITE_BLOCK")) {
      long long x = std::atoll(env);
      if (x > 0) v = static_cast<size_t>(x);
    }
    const size_t kMin = 64;
    const size_t kMax = 65536;
    v = std::max(v, kMin);
    v = std::min(v, kMax);
    v = std::min(v, std::max<size_t>(1, N));
    return v;
  };
  size_t pieces = k.cut_pred_keys.size() + 1;
  size_t stride = compiled.degree + 1;
  std::vector<uint64_t> preds_i(compiled.pred.queries.size(), 0);
  // Only a subset of pieces may expose boolean outputs; avoid wasting Beaver work
  // on empty bool-per-piece slots.
  std::vector<size_t> active_bool_pieces;
  active_bool_pieces.reserve(pieces);
  for (size_t p = 0; p < pieces && p < compiled.bool_per_piece.size(); ++p) {
    if (!compiled.bool_per_piece[p].empty()) active_bool_pieces.push_back(p);
  }

  size_t block_sz = read_block_sz_fallback(N);

  std::vector<uint64_t> selectors_block(pieces * block_sz, 0);
  std::vector<uint64_t> sel_vec;
  std::vector<uint64_t> rhs_vec;
  std::vector<uint64_t> prod_vec;
  std::vector<uint64_t> bool_piece_block;
  thread_local std::vector<uint64_t> b2a_xs;
  thread_local std::vector<uint64_t> b2a_ys;
  thread_local std::vector<uint64_t> b2a_prod;

  for (size_t blk = 0; blk < N; blk += block_sz) {
    size_t bsize = std::min(block_sz, N - blk);
    std::fill(selectors_block.begin(), selectors_block.end(), 0ull);
    if (compiled.ell > 0 && !active_bool_pieces.empty()) {
      bool_piece_block.assign(active_bool_pieces.size() * static_cast<size_t>(compiled.ell) * bsize, 0ull);
    }

    // Block-batched selectors from cut bits.
    selectors_from_cutbits_block(cut_bits_xor.data(), k.cut_pred_keys.size(), N, blk, bsize,
                                 party, mul_single, selectors_block.data(), block_sz);

    // Per-element predicates -> bool DAG. We first compute XOR-shared bits, then
    // batch-convert to additive shares to avoid millions of tiny Beaver opens.
    for (size_t off = 0; off < bsize; off++) {
      size_t i = blk + off;
      const auto& wrap_vars = k.wrap_share;
      auto get_pred = [&](size_t idx) -> uint64_t {
        return (idx < preds_i.size()) ? preds_i[idx] : 0ull;
      };
      for (size_t qi = 0; qi < compiled.pred.queries.size(); qi++) {
        preds_i[qi] = pred_bits_xor[qi * N + i] & 1ull;
      }

      if (compiled.ell > 0 && !active_bool_pieces.empty()) {
        size_t bit_idx = 0;  // reset for each element, matching legacy order
        for (size_t ap = 0; ap < active_bool_pieces.size(); ++ap) {
          const size_t p = active_bool_pieces[ap];
          const auto& exprs = compiled.bool_per_piece[p];
          for (int j = 0; j < compiled.ell; j++) {
            uint64_t bx =
                gates::eval_bool_xor_view(exprs[static_cast<size_t>(j)],
                                          get_pred, wrap_vars, mul_single,
                                          bit_ptr, &bit_idx) &
                1ull;
            // Temporarily store XOR bit; converted in one batch below.
            bool_piece_block[(ap * static_cast<size_t>(compiled.ell) + static_cast<size_t>(j)) * bsize + off] = bx;
          }
        }
      }
    }

    // Batch XOR->additive conversion for all bool_piece_block bits in this block.
    if (compiled.ell > 0 && !active_bool_pieces.empty()) {
      const size_t ell = static_cast<size_t>(compiled.ell);
      const size_t total_bits = active_bool_pieces.size() * ell * bsize;
      b2a_xs.resize(total_bits);
      b2a_ys.resize(total_bits);
      for (size_t t = 0; t < total_bits; ++t) {
        uint64_t bx = bool_piece_block[t] & 1ull;
        b2a_xs[t] = (party == 0) ? bx : 0ull;
        b2a_ys[t] = (party == 1) ? bx : 0ull;
      }
      mul_single.mul_batch(b2a_xs, b2a_ys, b2a_prod);
      for (size_t t = 0; t < total_bits; ++t) {
        uint64_t two_prod = proto::add_mod(b2a_prod[t], b2a_prod[t]);
        bool_piece_block[t] = proto::sub_mod(proto::add_mod(b2a_xs[t], b2a_ys[t]), two_prod);
      }
    }

    // Selector-weighted bool outputs (batched over block).
    if (compiled.ell > 0 && !active_bool_pieces.empty()) {
      const size_t ell = static_cast<size_t>(compiled.ell);
      const size_t mul_n = ell * bsize;
      sel_mul_x.resize(mul_n);
      sel_mul_y.resize(mul_n);
      for (size_t ap = 0; ap < active_bool_pieces.size(); ++ap) {
        const size_t p = active_bool_pieces[ap];
        for (size_t j = 0; j < ell; ++j) {
          const uint64_t* rhs =
              bool_piece_block.data() + (ap * ell + j) * bsize;
          uint64_t* xdst = sel_mul_x.data() + j * bsize;
          uint64_t* ydst = sel_mul_y.data() + j * bsize;
          for (size_t off = 0; off < bsize; ++off) {
            xdst[off] = selectors_block[p * block_sz + off];
            ydst[off] = rhs[off];
          }
        }
        mul_single.mul_batch(sel_mul_x, sel_mul_y, sel_mul_prod);
        for (size_t j = 0; j < ell; ++j) {
          const uint64_t* prod = sel_mul_prod.data() + j * bsize;
          for (size_t off = 0; off < bsize; ++off) {
            size_t out_idx = (blk + off) * ell + j;
            out.bool_share[out_idx] = proto::add_mod(out.bool_share[out_idx], prod[off]);
          }
        }
      }
    }

    // Batched Horner over the block (Beaver-free; polynomials are in public hatx).
    std::vector<uint64_t> hatx_vec(bsize);
#ifdef _OPENMP
#pragma omp parallel for if (bsize >= (1ull << 12)) schedule(static)
#endif
    for (size_t off = 0; off < bsize; off++) {
      hatx_vec[off] = proto::norm_mod(in.hatx[blk + off]);
    }
    const size_t r_words = static_cast<size_t>(compiled.r);
    if (compiled.degree <= 0) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (bsize * r_words >= (1ull << 15)) schedule(static)
#endif
      for (size_t j = 0; j < r_words; ++j) {
        for (size_t off = 0; off < bsize; ++off) {
          const size_t base = j * stride;
          const uint64_t rout = (j < k.r_out_share.size()) ? k.r_out_share[j] : 0ull;
          size_t i = blk + off;
          out.haty_share[i * r_words + j] = proto::add_mod(coeff_selected_soa[base * N + i], rout);
        }
      }
    } else {
      const int degree = compiled.degree;
      const size_t horner_n = r_words * bsize;
      horner_acc.resize(horner_n);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (bsize * r_words >= (1ull << 15)) schedule(static)
#endif
      for (size_t j = 0; j < r_words; ++j) {
        for (size_t off = 0; off < bsize; ++off) {
          const size_t base = j * stride;
          horner_acc[j * bsize + off] =
              coeff_selected_soa[(base + static_cast<size_t>(degree)) * N + (blk + off)];
        }
      }
      for (int d = degree - 1; d >= 0; --d) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (bsize * r_words >= (1ull << 15)) schedule(static)
#endif
        for (size_t j = 0; j < r_words; ++j) {
          for (size_t off = 0; off < bsize; ++off) {
            const size_t base = j * stride;
            const uint64_t acc = horner_acc[j * bsize + off];
            const uint64_t h = hatx_vec[off];
            const uint64_t c = coeff_selected_soa[(base + static_cast<size_t>(d)) * N + (blk + off)];
            horner_acc[j * bsize + off] = proto::add_mod(proto::mul_mod(acc, h), c);
          }
        }
      }
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (bsize * r_words >= (1ull << 15)) schedule(static)
#endif
      for (size_t j = 0; j < r_words; ++j) {
        for (size_t off = 0; off < bsize; ++off) {
          const uint64_t rout = (j < k.r_out_share.size()) ? k.r_out_share[j] : 0ull;
          size_t i = blk + off;
          out.haty_share[i * r_words + j] = proto::add_mod(horner_acc[j * bsize + off], rout);
        }
      }
    }
  }
  if (staged && want_device_out) {
#ifdef SUF_HAVE_CUDA
    auto* gpu = dynamic_cast<proto::PfssGpuStagedEval*>(staged);
    auto* stream_ptr = (gpu) ? gpu->device_stream() : nullptr;
    if (gpu && stream_ptr != nullptr) {
      auto* stream = reinterpret_cast<cudaStream_t>(stream_ptr);
      // Ensure device buffers are large enough for full arith/bool outputs.
      gpu->ensure_output_buffers(out.haty_share.size(), out.bool_share.size());
      // Copy host outputs to backend buffers so pointers remain stable until next call.
      if (!out.haty_share.empty() && gpu->last_device_output()) {
        size_t bytes = out.haty_share.size() * sizeof(uint64_t);
        cudaMemcpyAsync(const_cast<uint64_t*>(gpu->last_device_output()),
                        out.haty_share.data(), bytes,
                        cudaMemcpyHostToDevice, stream);
        out.haty_device = gpu->last_device_output();
        out.haty_device_words = out.haty_share.size();
      }
      if (!out.bool_share.empty() && gpu->last_device_bools()) {
        size_t bytes = out.bool_share.size() * sizeof(uint64_t);
        cudaMemcpyAsync(const_cast<uint64_t*>(gpu->last_device_bools()),
                        out.bool_share.data(), bytes,
                        cudaMemcpyHostToDevice, stream);
        out.bool_device = gpu->last_device_bools();
        out.bool_device_words = out.bool_share.size();
      }
      cudaStreamSynchronize(stream);
      // Surface device pointers regardless of host staging so callers can keep data on device.
      out.haty_device = gpu->last_device_output();
      out.haty_device_words = N * static_cast<size_t>(compiled.r);
      out.bool_device = gpu->last_device_bools();
      out.bool_device_words = N * static_cast<size_t>(compiled.ell);
    }
#endif
  }
  return out;
}

inline CompositeTape write_composite_tape(const CompositePartyKey& k) {
  CompositeTape t;
  auto& v = t.bytes;
  tape_append_u64(v, k.r_in_share);
  tape_append_u32(v, static_cast<uint32_t>(k.r_in_share_vec.size()));
  for (auto x : k.r_in_share_vec) tape_append_u64(v, x);
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
  uint32_t r_in_vec_len = tape_read_u32(p);
  need(static_cast<size_t>(r_in_vec_len) * 8);
  k.r_in_share_vec.resize(r_in_vec_len);
  for (uint32_t i = 0; i < r_in_vec_len; i++) k.r_in_share_vec[i] = tape_read_u64(p);

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
  // Use a generous synthetic triple pool to avoid exhaustion when keys do not provision enough.
  std::vector<proto::BeaverTriple64Share> synth_triples;
  size_t generous_need = std::max<size_t>(
      2048, static_cast<size_t>(in.N) * static_cast<size_t>(k.compiled.r + k.compiled.ell + k.compiled.degree + 4) * 4);
  synth_triples.reserve(generous_need);
  std::mt19937_64 rng(k.compiled.r_in ^ 0x706f7374u);  // "post"
  for (size_t i = 0; i < generous_need; ++i) {
    uint64_t a = rng(), b = rng(), c = proto::mul_mod(a, b);
    uint64_t a0 = rng(), b0 = rng(), c0 = rng();
    synth_triples.push_back((party == 0) ? proto::BeaverTriple64Share{a0, b0, c0}
                                         : proto::BeaverTriple64Share{a - a0, b - b0, c - c0});
  }
  proto::BeaverMul64 mul{party, ch, synth_triples, 0};
  hook.configure(k.compiled.layout);
  if (auto* relu = dynamic_cast<gates::ReluARSPostProc*>(&hook)) {
    relu->r_hi_share = k.r_hi_share;
    relu->wrap_sign_share = k.wrap_sign_share;
    if (relu->delta.empty() && !k.extra_params.empty()) {
      relu->delta = k.extra_params;
    }
  } else if (auto* tr = dynamic_cast<gates::FaithfulTruncPostProc*>(&hook)) {
    tr->r_hi_share = k.r_hi_share;
  } else if (auto* ars = dynamic_cast<gates::FaithfulArsPostProc*>(&hook)) {
    ars->r_hi_share = k.r_hi_share;
  } else if (auto* gap = dynamic_cast<gates::GapArsPostProc*>(&hook)) {
    gap->r_hi_share = k.r_hi_share;
    gap->m_share = k.wrap_sign_share;
  }
  hook.run_batch(party, ch, mul, in.hatx, out.haty_share.data(),
                 static_cast<size_t>(k.compiled.r),
                 out.bool_share.data(), static_cast<size_t>(k.compiled.ell),
                 in.N, out.haty_share.data());
  return out;
}

}  // namespace gates
