#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "compiler/range_analysis.hpp"
#include "compiler/layer_graph.hpp"
#include "compiler/truncation_pass_runner.hpp"
#include "nn/tensor_view.hpp"
#include "runtime/pfss_superbatch.hpp"
#include "runtime/open_collector.hpp"
#include "runtime/pfss_phase_planner.hpp"
#include "runtime/pfss_async_runner.hpp"
#include "proto/backend_factory.hpp"
#include "mpc/net.hpp"

namespace proto {
struct IChannel;
}

namespace nn {

// Simple per-layer builder context that records ops/ranges/scales and allows
// callers to attach truncation plans.
struct LayerContext {
  compiler::LayerGraph graph;
  compiler::TruncationPassContext* trunc_ctx = nullptr;  // owned externally
  runtime::PfssSuperBatch* pfss_batch = nullptr;         // optional runtime batching surface
  runtime::OpenCollector* open_collector = nullptr;      // optional batched opens surface
  runtime::PfssLayerPlanner* pfss_layer_planner = nullptr;  // optional cross-phase planner
  runtime::PfssGpuStager* pfss_gpu_stager = nullptr;         // optional device staging surface
  proto::PfssBackendBatch* pfss_backend_override = nullptr;  // optional backend (e.g., GPU) override
  std::unique_ptr<proto::PfssBackendBatch> owned_pfss_backend;  // managed override if set
  proto::IChannel* pfss_chan = nullptr;  // optional dedicated PFSS byte channel
  net::Chan* pfss_net_chan = nullptr;  // optional dedicated PFSS channel; defaults to main chan
#ifdef SUF_HAVE_CUDA
  // Optional CUDA stream override used for OpenCollector device packing/unpacking
  // and other CUDA helpers even when the PFSS backend is CPU-based.
  void* cuda_stream_override = nullptr;  // cudaStream_t
#endif
  int frac_bits = 16;
  // Enable a layer-wide super-plan: suppress inner PFSS/Open drains inside attention/MLP
  // and only drain at explicit layer barriers chosen by the caller.
  bool disable_inner_barriers = false;
  bool enable_hoist = false;  // enable conservative rescale hoisting when finalizing
  bool allow_async_pfss = false;  // if true, caller may detach PFSS flushes (requires safe channel)
  std::unique_ptr<runtime::PfssAsyncRunner> pfss_async_runner;  // retained across layers when async
  bool force_eager_pfss = false;  // optional: disable lazy PFSS scheduling for tests/regressions
  std::optional<compiler::TruncationPassResult> last_trunc;
  std::unordered_map<int, TensorView<uint64_t>> bindings;

  void bind(int tid, const TensorView<uint64_t>& buf) { bindings[tid] = buf; }
  std::optional<TensorView<uint64_t>> view(int tid) const {
    auto it = bindings.find(tid);
    if (it == bindings.end()) return std::nullopt;
    return it->second;
  }

  proto::PfssBackendBatch& trunc_backend() {
    if (pfss_backend_override) return *pfss_backend_override;
    if (owned_pfss_backend) {
      pfss_backend_override = owned_pfss_backend.get();
      return *owned_pfss_backend;
    }
    if (!trunc_ctx) throw std::runtime_error("LayerContext: trunc_ctx is required");
    return trunc_ctx->backend();
  }
  const proto::PfssBackendBatch& trunc_backend() const {
    if (pfss_backend_override) return *pfss_backend_override;
    if (owned_pfss_backend) return *owned_pfss_backend;
    if (!trunc_ctx) throw std::runtime_error("LayerContext: trunc_ctx is required");
    return trunc_ctx->backend();
  }

  void set_backend_override(std::unique_ptr<proto::PfssBackendBatch> b) {
    owned_pfss_backend = std::move(b);
    pfss_backend_override = owned_pfss_backend.get();
  }

  // Optional helper: select backend from env (SUF_PFSS_BACKEND) if no override present.
  void select_backend_from_env() {
    if (pfss_backend_override || owned_pfss_backend) return;
    const char* env = std::getenv("SUF_PFSS_BACKEND");
    if (!env) return;  // leave trunc_ctx backend intact when no env override is set
    owned_pfss_backend = proto::make_pfss_backend(proto::PfssBackendOptions{
        .kind = proto::parse_backend_kind(env),
        .allow_gpu_stub = true});
    pfss_backend_override = owned_pfss_backend.get();
  }

  // Optional helper: return GPU compute stream if backend supports staged eval.
  void* pfss_compute_stream() const {
    const proto::PfssBackendBatch* b = pfss_backend_override ? pfss_backend_override : owned_pfss_backend.get();
#ifdef SUF_HAVE_CUDA
    if (auto* gpu = dynamic_cast<const proto::PfssGpuStagedEval*>(b)) {
      return gpu->device_stream();
    }
#endif
    (void)b;
    return nullptr;
  }

  // Unified CUDA stream accessor used by runtime scheduling. Prefers explicit override.
  void* cuda_stream() const {
#ifdef SUF_HAVE_CUDA
    if (cuda_stream_override) return cuda_stream_override;
#endif
    return pfss_compute_stream();
  }

  bool uses_gpu_backend() const {
#ifdef SUF_HAVE_CUDA
    const proto::PfssBackendBatch* b = pfss_backend_override ? pfss_backend_override : owned_pfss_backend.get();
    return (b && dynamic_cast<const proto::PfssGpuStagedEval*>(b) != nullptr);
#else
    return false;
#endif
  }
};

inline compiler::Scale make_scale(int frac_bits, bool is_signed = true) {
  compiler::Scale s;
  s.frac_bits = frac_bits;
  s.is_signed = is_signed;
  return s;
}

struct SecretTensor {
  TensorView<uint64_t> share;
  compiler::Scale scale;
  compiler::RangeInterval range = compiler::RangeInterval::whole(true);
  int tid = -1;
  size_t producer_op = static_cast<size_t>(-1);
  LayerContext* ctx = nullptr;

  bool valid() const { return tid >= 0; }
};

inline compiler::RangeInterval clamp_silu_range(int frac_bits) {
  // SiLU with typical LUT input clamp [-6, 6]; output is non-expansive on the
  // negative side (min near -0.3), so tighten the negative bound to ~-1.
  int64_t pos_bound = static_cast<int64_t>(6ll << frac_bits);
  int64_t neg_bound = static_cast<int64_t>(1ll << frac_bits);
  compiler::RangeInterval r;
  r.is_signed = true;
  r.lo = -neg_bound;
  r.hi = pos_bound;
  return r;
}

inline compiler::RangeInterval clamp_nexp_range(int frac_bits) {
  // nExp output in (0, 1]; represent as [0, 1<<frac_bits].
  compiler::RangeInterval r;
  r.is_signed = true;
  r.lo = 0;
  r.hi = static_cast<int64_t>(1ll << frac_bits);
  return r;
}

inline compiler::RangeInterval clamp_recip_range(int frac_bits, double max_in) {
  // Reciprocal of [1, max_in] lies in [1/max_in, 1].
  compiler::RangeInterval r;
  r.is_signed = true;
  int64_t hi = static_cast<int64_t>(1ll << frac_bits);
  int64_t lo = static_cast<int64_t>(std::llround((1.0 / std::max(max_in, 1.0)) * std::ldexp(1.0, frac_bits)));
  r.lo = lo;
  r.hi = hi;
  return r;
}

inline compiler::RangeInterval clamp_softmax_range(int frac_bits) {
  // Softmax probabilities in [0,1].
  compiler::RangeInterval r;
  r.is_signed = true;
  r.lo = 0;
  r.hi = static_cast<int64_t>(1ll << frac_bits);
  return r;
}

inline compiler::RangeInterval clamp_gelu_range(int frac_bits) {
  // GeLU output roughly in [-4, 4] for typical inputs.
  compiler::RangeInterval r;
  r.is_signed = true;
  int64_t bound = static_cast<int64_t>(4ll << frac_bits);
  r.lo = -bound;
  r.hi = bound;
  return r;
}

inline compiler::RangeInterval clamp_layernorm_range(int frac_bits) {
  // LayerNorm affine output is typically bounded to a small multiple of unit variance.
  compiler::RangeInterval r;
  r.is_signed = true;
  int64_t bound = static_cast<int64_t>(8ll << frac_bits);
  r.lo = -bound;
  r.hi = bound;
  return r;
}

// Lightweight helpers to avoid boilerplate in layer builders.
inline SecretTensor make_secret_tensor(LayerContext* ctx,
                                       const TensorView<uint64_t>& share,
                                       const compiler::Scale& scale,
                                       const compiler::RangeInterval& range =
                                           compiler::RangeInterval::whole(true)) {
  SecretTensor t;
  t.share = share;
  t.scale = scale;
  t.range = range;
  t.ctx = ctx;
  if (!ctx) return t;
  t.tid = ctx->graph.add_tensor(scale, range);
  return t;
}

// Public (unmasked) tensor: used for public weights/constants in the compiler graph
// (e.g., LayerNorm gamma/beta) so mask bounds don't get inflated.
inline SecretTensor make_public_tensor(LayerContext* ctx,
                                       const compiler::Scale& scale,
                                       const compiler::RangeInterval& range) {
  SecretTensor t;
  t.share = TensorView<uint64_t>();  // no backing buffer
  t.scale = scale;
  t.range = range;
  t.ctx = ctx;
  if (!ctx) return t;
  t.tid = ctx->graph.add_public_tensor(scale, range);
  return t;
}

inline SecretTensor record_matmul(LayerContext* ctx,
                                  const SecretTensor& x,
                                  compiler::MatmulAttrs attrs,
                                  const compiler::Scale& out_scale,
                                  const compiler::RangeInterval& out_range =
                                      compiler::RangeInterval::whole(true),
                                  const TensorView<uint64_t>& out_share = {}) {
  SecretTensor t;
  t.share = out_share;
  t.scale = out_scale;
  t.range = out_range;
  t.ctx = ctx;
  if (!ctx || !x.valid()) return t;
  if (attrs.params && ctx->pfss_batch) {
    attrs.params->pfss_batch = ctx->pfss_batch;
    attrs.params->defer_trunc_finalize = true;  // allow batching; caller flushes
    attrs.params->require_truncation = true;
    attrs.params->open_collector = ctx->open_collector;
    attrs.params->defer_open_flush = true;
  }
  attrs.x_range = x.range;
  size_t op_idx = ctx->graph.current_op_index();
  t.tid = ctx->graph.add_matmul_beaver(x.tid, attrs, out_scale, out_range);
  t.range = compiler::matmul_accum_range(x.range, attrs.w_range, attrs.K);
  t.producer_op = op_idx;
  return t;
}

inline SecretTensor record_rescale(LayerContext* ctx,
                                   const SecretTensor& input,
                                   const compiler::RescaleAttrs& attrs,
                                   const compiler::Scale& out_scale,
                                   const compiler::RangeInterval& out_range =
                                       compiler::RangeInterval::whole(true),
                                   const TensorView<uint64_t>& out_share = {}) {
  SecretTensor t;
  t.share = out_share;
  t.scale = out_scale;
  t.range = out_range;
  t.ctx = ctx;
  if (!ctx || !input.valid()) return t;
  compiler::RescaleAttrs attrs_copy = attrs;
  // Populate missing frac hints from scales.
  if (attrs_copy.from_frac == 0) attrs_copy.from_frac = input.scale.frac_bits;
  if (attrs_copy.to_frac == 0) attrs_copy.to_frac = out_scale.frac_bits;
  compiler::GateKind kind = compiler::GateKind::FaithfulARS;
  if (static_cast<size_t>(input.tid) < ctx->graph.tensors().size()) {
    const auto& tf = ctx->graph.tensors()[static_cast<size_t>(input.tid)];
    kind = compiler::select_trunc_kind(tf.abs, attrs_copy.to_frac, tf.gap, tf.mask_abs);
  } else {
    kind = compiler::select_trunc_kind(input.range, attrs_copy.to_frac, compiler::RangeKind::Hint, std::nullopt, input.scale.is_signed ? compiler::default_mask_bound(attrs_copy.to_frac) : 0);
  }
  attrs_copy.prefer_gapars = attrs_copy.prefer_gapars || (kind == compiler::GateKind::GapARS);
  t.tid = ctx->graph.add_rescale(input.tid, attrs_copy, out_scale, out_range);
  compiler::RangeInterval inferred = compiler::shift_down(input.range, attrs_copy.from_frac - attrs_copy.to_frac);
  if (out_range.lo != std::numeric_limits<int64_t>::min() ||
      out_range.hi != std::numeric_limits<int64_t>::max()) {
    t.range = compiler::intersect(out_range, inferred);
  } else {
    t.range = inferred;
  }
  return t;
}

inline SecretTensor record_clamp(LayerContext* ctx,
                                 const SecretTensor& input,
                                 const compiler::RangeInterval& clamp_r,
                                 const compiler::Scale& out_scale,
                                 const TensorView<uint64_t>& out_share = {}) {
  SecretTensor t;
  t.share = out_share;
  t.scale = out_scale;
  t.range = clamp_r;
  t.ctx = ctx;
  if (!ctx || !input.valid()) return t;
  t.tid = ctx->graph.add_clamp(input.tid, clamp_r, out_scale);
  return t;
}

inline compiler::RangeInterval range_from_public_weights(
    const TensorView<int64_t>& W) {
  compiler::RangeInterval r = compiler::RangeInterval::whole(true);
  if (!W.data || W.numel() == 0) {
    r.lo = 0;
    r.hi = 0;
    return r;
  }
  const bool cache_enabled = []() {
    const char* env = std::getenv("SUF_CACHE_WEIGHT_BOUNDS");
    if (!env) return true;
    std::string v(env);
    std::transform(v.begin(), v.end(), v.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return !(v == "0" || v == "false" || v == "off" || v == "no");
  }();
  struct Key {
    const int64_t* ptr = nullptr;
    size_t n = 0;
    bool operator==(const Key& o) const { return ptr == o.ptr && n == o.n; }
  };
  struct KeyHash {
    size_t operator()(const Key& k) const noexcept {
      size_t h = 1469598103934665603ull;
      auto mix = [&](size_t v) {
        h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
      };
      mix(reinterpret_cast<size_t>(k.ptr));
      mix(static_cast<size_t>(k.n));
      return h;
    }
  };
  if (cache_enabled) {
    static std::mutex mu;
    static std::unordered_map<Key, int64_t, KeyHash> cache;
    Key key{W.data, W.numel()};
    {
      std::lock_guard<std::mutex> lk(mu);
      auto it = cache.find(key);
      if (it != cache.end()) {
        int64_t max_abs = it->second;
        r.lo = -max_abs;
        r.hi = max_abs;
        return r;
      }
    }
    int64_t max_abs = 0;
    for (size_t i = 0; i < W.numel(); ++i) {
      int64_t v = W.data[i];
      int64_t a = (v >= 0) ? v : -v;
      if (a > max_abs) max_abs = a;
    }
    {
      std::lock_guard<std::mutex> lk(mu);
      cache.emplace(key, max_abs);
    }
    r.lo = -max_abs;
    r.hi = max_abs;
    return r;
  }
  int64_t max_abs = 0;
  for (size_t i = 0; i < W.numel(); ++i) {
    int64_t v = W.data[i];
    int64_t a = (v >= 0) ? v : -v;
    if (a > max_abs) max_abs = a;
  }
  r.lo = -max_abs;
  r.hi = max_abs;
  return r;
}

inline int64_t row_l1_max(const TensorView<int64_t>& W, bool w_transposed = false) {
  // Column L1 norm bound (per output column); returns the maximum across outputs.
  if (!W.data || W.numel() == 0 || W.dims < 2) return 0;
  const bool cache_enabled = []() {
    const char* env = std::getenv("SUF_CACHE_WEIGHT_BOUNDS");
    if (!env) return true;
    std::string v(env);
    std::transform(v.begin(), v.end(), v.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return !(v == "0" || v == "false" || v == "off" || v == "no");
  }();
  struct Key {
    const int64_t* ptr = nullptr;
    size_t rows = 0;
    size_t cols = 0;
    bool transposed = false;
    bool operator==(const Key& o) const {
      return ptr == o.ptr && rows == o.rows && cols == o.cols && transposed == o.transposed;
    }
  };
  struct KeyHash {
    size_t operator()(const Key& k) const noexcept {
      size_t h = 1469598103934665603ull;
      auto mix = [&](size_t v) {
        h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
      };
      mix(reinterpret_cast<size_t>(k.ptr));
      mix(static_cast<size_t>(k.rows));
      mix(static_cast<size_t>(k.cols));
      mix(static_cast<size_t>(k.transposed ? 1 : 0));
      return h;
    }
  };
  size_t rows = W.shape[0];
  size_t cols = W.shape[1];
  if (w_transposed) std::swap(rows, cols);
  if (cache_enabled) {
    static std::mutex mu;
    static std::unordered_map<Key, int64_t, KeyHash> cache;
    Key key{W.data, W.shape[0], W.shape[1], w_transposed};
    {
      std::lock_guard<std::mutex> lk(mu);
      auto it = cache.find(key);
      if (it != cache.end()) return it->second;
    }
    int64_t best = 0;
    for (size_t c = 0; c < cols; ++c) {
      int64_t acc = 0;
      for (size_t r = 0; r < rows; ++r) {
        size_t idx = w_transposed ? (c * rows + r) : (r * cols + c);
        int64_t v = W.data[idx];
        acc += (v >= 0) ? v : -v;
      }
      if (acc > best) best = acc;
    }
    {
      std::lock_guard<std::mutex> lk(mu);
      cache.emplace(key, best);
    }
    return best;
  }
  int64_t best = 0;
  for (size_t c = 0; c < cols; ++c) {
    int64_t acc = 0;
    for (size_t r = 0; r < rows; ++r) {
      size_t idx = w_transposed ? (c * rows + r) : (r * cols + c);
      int64_t v = W.data[idx];
      acc += (v >= 0) ? v : -v;
    }
    if (acc > best) best = acc;
  }
  return best;
}

inline void finalize_layer(LayerContext& ctx,
                           int party,
                           net::Chan& ch,
                           proto::PfssBackendBatch& backend,
                           bool flush_pfss = true) {
  ctx.graph.propagate_ranges();
  if (ctx.enable_hoist) {
    ctx.graph.hoist_rescales();
    ctx.graph.propagate_ranges();
  }
  if (!ctx.trunc_ctx) return;
  ctx.last_trunc = ctx.graph.lower_truncations(*ctx.trunc_ctx, ctx.pfss_batch);
  if (flush_pfss && ctx.pfss_batch && !ctx.pfss_batch->empty()) {
    runtime::ProtoChanFromNet pch(ch);
    ctx.pfss_batch->flush_and_finalize(party, backend, pch);
  }
}

}  // namespace nn
