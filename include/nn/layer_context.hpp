#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <stdexcept>

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
  net::Chan* pfss_net_chan = nullptr;  // optional dedicated PFSS channel; defaults to main chan
  int frac_bits = 16;
  // Enable a layer-wide super-plan: suppress inner PFSS/Open drains inside attention/MLP
  // and only drain at explicit layer barriers chosen by the caller.
  bool disable_inner_barriers = false;
  bool enable_hoist = false;  // enable conservative rescale hoisting when finalizing
  bool allow_async_pfss = false;  // if true, caller may detach PFSS flushes (requires safe channel)
  std::unique_ptr<runtime::PfssAsyncRunner> pfss_async_runner;  // retained across layers when async
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
    owned_pfss_backend = proto::make_pfss_backend_from_env();
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
    kind = compiler::select_trunc_kind(tf.abs, attrs_copy.to_frac, tf.gap);
  } else {
    kind = compiler::select_trunc_kind(input.range, attrs_copy.to_frac);
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
  size_t rows = W.shape[0];
  size_t cols = W.shape[1];
  if (w_transposed) std::swap(rows, cols);
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
