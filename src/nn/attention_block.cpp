#include "nn/attention_block.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <optional>
#include <random>
#include <vector>
#include <unordered_map>
#include "compiler/truncation_lowering.hpp"
#include "compiler/range_analysis.hpp"
#include "gates/nexp_gate.hpp"
#include "gates/reciprocal_gate.hpp"
#include "gates/reciprocal_composite.hpp"
#include "gates/nexp_composite.hpp"
#include "gates/softmax_composite.hpp"
#include "nn/softmax_block_task.hpp"
#include "runtime/pfss_phase_planner.hpp"
#include "runtime/pfss_superbatch.hpp"
#include "runtime/phase_tasks.hpp"

namespace nn {

namespace {

inline int64_t to_signed(uint64_t v) { return static_cast<int64_t>(v); }
inline uint64_t to_ring(int64_t v) { return static_cast<uint64_t>(v); }

// Temporary helper to open shares to plaintext (will be removed once score/prob path is fully taskified).
void open_to_plain(int party,
                   net::Chan& ch,
                   const uint64_t* local,
                   size_t len,
                   std::vector<int64_t>& plain_out) {
  plain_out.resize(len);
  std::vector<uint64_t> other(len, 0);
  if (party == 0) {
    for (size_t i = 0; i < len; ++i) ch.send_u64(local[i]);
    for (size_t i = 0; i < len; ++i) other[i] = ch.recv_u64();
  } else {
    for (size_t i = 0; i < len; ++i) other[i] = ch.recv_u64();
    for (size_t i = 0; i < len; ++i) ch.send_u64(local[i]);
  }
  for (size_t i = 0; i < len; ++i) {
    plain_out[i] = to_signed(local[i]) + to_signed(other[i]);
  }
}

struct RowBroadcastTripleMaterial {
  int rows = 0;
  int cols = 0;
  std::vector<uint64_t> A0, A1;
  std::vector<uint64_t> B0, B1;
  std::vector<uint64_t> C0, C1;
};

RowBroadcastTripleMaterial make_row_broadcast_triples(int rows, int cols, std::mt19937_64& rng) {
  RowBroadcastTripleMaterial mat;
  mat.rows = rows;
  mat.cols = cols;
  size_t count = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  mat.A0.resize(count);
  mat.A1.resize(count);
  mat.B0.resize(static_cast<size_t>(rows));
  mat.B1.resize(static_cast<size_t>(rows));
  mat.C0.resize(count);
  mat.C1.resize(count);

  std::vector<uint64_t> B(rows);
  for (int r = 0; r < rows; ++r) {
    uint64_t b = rng();
    uint64_t b0 = rng();
    uint64_t b1 = b - b0;
    B[static_cast<size_t>(r)] = b;
    mat.B0[static_cast<size_t>(r)] = b0;
    mat.B1[static_cast<size_t>(r)] = b1;
  }
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      size_t idx = static_cast<size_t>(r * cols + c);
      uint64_t a = rng();
      uint64_t a0 = rng();
      uint64_t a1 = a - a0;
      uint64_t c_val = proto::mul_mod(a, B[static_cast<size_t>(r)]);
      uint64_t c0 = rng();
      uint64_t c1 = c_val - c0;
      mat.A0[idx] = a0;
      mat.A1[idx] = a1;
      mat.C0[idx] = c0;
      mat.C1[idx] = c1;
    }
  }
  return mat;
}

class RowBroadcastTripleProviderImpl : public runtime::RowBroadcastTripleProvider {
 public:
  RowBroadcastTripleProviderImpl(const RowBroadcastTripleMaterial& mat, int party)
      : mat_(mat), party_(party) {}

  runtime::RowBroadcastTriple reserve_mul(int rows, int cols) override {
    if (rows != mat_.rows || cols != mat_.cols) {
      throw std::runtime_error("RowBroadcastTripleProviderImpl: shape mismatch");
    }
    size_t count = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    const auto& A = (party_ == 0) ? mat_.A0 : mat_.A1;
    const auto& B = (party_ == 0) ? mat_.B0 : mat_.B1;
    const auto& C = (party_ == 0) ? mat_.C0 : mat_.C1;
    return {std::span<const uint64_t>(A.data(), count),
            std::span<const uint64_t>(B.data(), static_cast<size_t>(rows)),
            std::span<const uint64_t>(C.data(), count)};
  }

 private:
  const RowBroadcastTripleMaterial& mat_;
  int party_ = 0;
};

class CachedRowBroadcastTripleProvider : public runtime::RowBroadcastTripleProvider {
 public:
  explicit CachedRowBroadcastTripleProvider(int party) : party_(party) {}

  runtime::RowBroadcastTriple reserve_mul(int rows, int cols) override {
    const uint64_t key = (static_cast<uint64_t>(rows) << 32) | static_cast<uint32_t>(cols);
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      // Deterministic per-shape seed so both parties build identical materials and only keep per-shape cache.
      std::mt19937_64 rng(base_seed_ ^ key);
      auto mat = make_row_broadcast_triples(rows, cols, rng);
      it = cache_.emplace(key, std::move(mat)).first;
    }
    const auto& mat = it->second;
    const auto& A = (party_ == 0) ? mat.A0 : mat.A1;
    const auto& B = (party_ == 0) ? mat.B0 : mat.B1;
    const auto& C = (party_ == 0) ? mat.C0 : mat.C1;
    return {std::span<const uint64_t>(A.data(), A.size()),
            std::span<const uint64_t>(B.data(), B.size()),
            std::span<const uint64_t>(C.data(), C.size())};
  }

 private:
  static constexpr uint64_t base_seed_ = 0x9e3779b97f4a7c15ull;
  int party_ = 0;
  std::unordered_map<uint64_t, RowBroadcastTripleMaterial> cache_;
};

// Simple deterministic triple cache keyed by triple count (separate per party and seed domain).
static std::vector<proto::BeaverTriple64Share>& ensure_cached_triples(
    std::unordered_map<uint64_t, std::vector<proto::BeaverTriple64Share>>& cache,
    uint64_t seed_base,
    size_t count,
    int party) {
  auto it = cache.find(count);
  if (it != cache.end()) return it->second;
  std::mt19937_64 rng(seed_base ^ static_cast<uint64_t>(count));
  std::vector<proto::BeaverTriple64Share> triples(count);
  for (auto& tri : triples) {
    uint64_t a = rng();
    uint64_t b = rng();
    uint64_t c = proto::mul_mod(a, b);
    uint64_t a0 = rng();
    uint64_t b0 = rng();
    uint64_t c0 = rng();
    tri = (party == 0) ? proto::BeaverTriple64Share{a0, b0, c0}
                       : proto::BeaverTriple64Share{a - a0, b - b0, c - c0};
  }
  auto [ins_it, _] = cache.emplace(count, std::move(triples));
  return ins_it->second;
}

}  // namespace

void attention_forward(const AttentionConfig& cfg,
                       int party,
                       net::Chan& ch,
                       const TensorView<uint64_t>& X_share,
                       const TensorView<int64_t>& Wqkv_public,
                       const TensorView<int64_t>& Wout_public,
                       KVCache& cache,
                       TensorView<uint64_t> Y_share,
                       LayerContext* ctx,
                       runtime::PhaseExecutor* pe) {
  size_t B = X_share.shape[0];
  size_t T = X_share.shape[1];
  size_t D = cfg.D;
  size_t H = cfg.H;
  size_t Dh = cfg.Dh;
  int fb = cfg.frac_bits;
  if (!ctx || !ctx->trunc_ctx) {
    throw std::runtime_error("attention_forward: LayerContext with truncation context required (no legacy rescale)");
  }
  if (cfg.legacy_softmax) {
    throw std::runtime_error("attention_forward: legacy softmax path disabled (inline shifts not allowed)");
  }
  bool use_phase_softmax = !cfg.legacy_softmax && pe && ctx && ctx->trunc_ctx;

  std::vector<uint64_t> qkv(B * T * 3 * D, 0);

  compiler::RangeInterval q_range = compiler::RangeInterval::whole(true);
  compiler::AbsBound x_abs_hint{};
  std::optional<compiler::GapCert> x_gap_hint = std::nullopt;
  bool have_x_abs = false;
  compiler::AbsBound qkv_abs_hint{};
  std::optional<compiler::GapCert> qkv_gap_hint = std::nullopt;
  bool have_qkv_abs = false;
  if (ctx) {
    compiler::Scale q_scale = make_scale(fb, true);
    compiler::RangeInterval x_range = compiler::RangeInterval::whole(true);
    if (!ctx->graph.tensors().empty()) {
      x_range = ctx->graph.tensors().back().range;
    }
    SecretTensor x_t = make_secret_tensor(ctx, X_share, q_scale, x_range);
    q_range = x_range;
    if (x_t.valid() && static_cast<size_t>(x_t.tid) < ctx->graph.tensors().size()) {
      const auto& tf = ctx->graph.tensors()[static_cast<size_t>(x_t.tid)];
      x_abs_hint = tf.abs;
      x_gap_hint = tf.gap;
      have_x_abs = true;
    }

    compiler::MatmulAttrs qkv_attrs;
    qkv_attrs.M = B * T;
    qkv_attrs.K = D;
    qkv_attrs.N = 3 * D;
    qkv_attrs.w_transposed = false;
    qkv_attrs.params = nullptr;
    qkv_attrs.frac_bits = fb;
    qkv_attrs.x_range = x_range;
    qkv_attrs.row_l1_max = row_l1_max(Wqkv_public, qkv_attrs.w_transposed);
    qkv_attrs.w_range = range_from_public_weights(Wqkv_public);
    compiler::RangeInterval qkv_accum_range =
        qkv_attrs.row_l1_max > 0
            ? compiler::propagate_matmul_accum_rowl1(x_range, qkv_attrs.row_l1_max)
            : compiler::propagate_matmul_accum(x_range, qkv_attrs.w_range, qkv_attrs.K);
    auto qkv_acc = record_matmul(
        ctx, x_t, qkv_attrs, make_scale(2 * fb, true),
        qkv_accum_range,
        view2(qkv.data(), B * T, 3 * D));

    compiler::RescaleAttrs qkv_rescale_attrs;
    qkv_rescale_attrs.matmul_op = qkv_acc.producer_op;
    qkv_rescale_attrs.from_frac = 2 * fb;
    qkv_rescale_attrs.to_frac = fb;
    compiler::RangeInterval qkv_range =
        compiler::propagate_matmul_out(x_range, qkv_attrs.w_range, qkv_attrs.K, fb);
    SecretTensor qkv_t =
        record_rescale(ctx, qkv_acc, qkv_rescale_attrs, q_scale, qkv_range,
                       view2(qkv.data(), B * T, 3 * D));
    q_range = qkv_t.range;
    if (qkv_t.valid() && static_cast<size_t>(qkv_t.tid) < ctx->graph.tensors().size()) {
      const auto& tf = ctx->graph.tensors()[static_cast<size_t>(qkv_t.tid)];
      qkv_abs_hint = tf.abs;
      qkv_gap_hint = tf.gap;
      have_qkv_abs = true;
    }

    compiler::MatmulAttrs out_attrs;
    out_attrs.M = B * T;
    out_attrs.K = D;
    out_attrs.N = D;
    out_attrs.w_transposed = false;
    out_attrs.params = nullptr;
    out_attrs.frac_bits = fb;
    compiler::RangeInterval prob_range = clamp_softmax_range(fb);
    int max_len = std::max(cache.S_max, T);
    compiler::RangeInterval ctx_range =
        compiler::propagate_matmul_out(prob_range, qkv_t.range, max_len, fb);
    out_attrs.x_range = ctx_range;
    out_attrs.row_l1_max = row_l1_max(Wout_public, out_attrs.w_transposed);
    out_attrs.w_range = range_from_public_weights(Wout_public);
    auto out_acc = record_matmul(
        ctx, qkv_t, out_attrs, make_scale(2 * fb, true),
        out_attrs.row_l1_max > 0
            ? compiler::propagate_matmul_accum_rowl1(qkv_t.range, out_attrs.row_l1_max)
            : compiler::propagate_matmul_accum(qkv_t.range, out_attrs.w_range, out_attrs.K),
        Y_share);

    compiler::RescaleAttrs out_rescale_attrs;
    out_rescale_attrs.matmul_op = out_acc.producer_op;
    out_rescale_attrs.from_frac = 2 * fb;
    out_rescale_attrs.to_frac = fb;
    compiler::RangeInterval out_range =
        compiler::propagate_matmul_out(qkv_t.range, out_attrs.w_range, out_attrs.K, fb);
    (void)record_rescale(ctx, out_acc, out_rescale_attrs, q_scale, out_range, Y_share);
    (void)out_range;
    // q_range persists for downstream range use.
    (void)q_range;
  }

  assert(D == H * Dh);
  assert(Wqkv_public.shape[0] == D && Wqkv_public.shape[1] == 3 * D);
  assert(cache.B == B && cache.H == H && cache.Dh == Dh);

  MatmulParams mp;
  mp.frac_bits = fb;
  mp.w_transposed = false;
  mp.local_rescale = false;
  mp.allow_legacy_shift = false;
  mp.overlap_stream = ctx ? ctx->pfss_compute_stream() : nullptr;
  net::Chan* pfss_nc = (ctx && ctx->pfss_net_chan) ? ctx->pfss_net_chan : &ch;

  if (!ctx) {
    throw std::runtime_error("attention_forward: LayerContext required (no local rescale fallback)");
  }
  // Preserve PFSS/Open batches across phases so a layer-level planner can control flushing.
  // Make stall-driven behavior explicit for attention/softmax/out regions.
  pe->set_lazy_mode(true);
  pe->set_keep_batches(true);
  auto record_phase_plan = [&](runtime::PfssPhasePlanner& planner) {
    if (!ctx || !ctx->pfss_layer_planner) return;
    const auto& st = planner.stats();
    if (st.coeff_jobs == 0 && st.trunc_jobs == 0 && st.coeff_flushes == 0 && st.trunc_flushes == 0) return;
    ctx->pfss_layer_planner->record_phase(planner, pe->pfss_coeff_batch(), pe->pfss_trunc_batch());
    if (party == 0) {
      std::cerr << "[pfss-phase][attn] coeff_jobs=" << st.coeff_jobs
                << " trunc_jobs=" << st.trunc_jobs
                << " coeff_flushes=" << st.coeff_flushes
                << " trunc_flushes=" << st.trunc_flushes
                << " coeff_hatx=" << pe->pfss_coeff_batch().stats().max_bucket_hatx
                << " trunc_hatx=" << pe->pfss_trunc_batch().stats().max_bucket_hatx << "\n";
    }
    pe->pfss_coeff_batch().reset_stats();
    pe->pfss_trunc_batch().reset_stats();
  };
  auto barrier = [&](const runtime::PfssLayerPlanner::BarrierPolicy& pol) {
    if (ctx && ctx->disable_inner_barriers) return;
    if (ctx && ctx->pfss_layer_planner) {
      runtime::ProtoChanFromNet pch_bar(*pfss_nc);
      ctx->pfss_layer_planner->barrier(
          party,
          ctx->trunc_backend(),
          pe->pfss_coeff_batch(),
          pe->pfss_trunc_batch(),
          pch_bar,
          &pe->open_collector(),
          &ch,
          pol);
    }
  };
  auto enter_phase = [&]() {
    if (ctx && ctx->pfss_layer_planner) ctx->pfss_layer_planner->enter_phase();
  };

  matmul_publicW(view2(const_cast<uint64_t*>(X_share.data), B * T, D),
                 Wqkv_public,
                 view2(qkv.data(), B * T, 3 * D),
                 mp);
  compiler::RangeInterval wqkv_range = range_from_public_weights(Wqkv_public);
  compiler::AbsBound wqkv_abs = compiler::abs_from_range(wqkv_range, true);
  wqkv_abs.kind = compiler::RangeKind::Proof;
  runtime::PfssPhasePlanner pfss_phase_planner;
  pfss_phase_planner.bind(&pe->pfss_coeff_batch(), &pe->pfss_trunc_batch());
  // Truncate qkv accum (Q2f -> Qf) via composite truncation (no local shift).
  compiler::GateParams trunc_params_qkv;
  trunc_params_qkv.kind = compiler::GateKind::AutoTrunc;
  trunc_params_qkv.frac_bits = fb;
  trunc_params_qkv.range_hint = compiler::matmul_accum_range(
      q_range, wqkv_range, D);
  trunc_params_qkv.per_element_masks = true;
  compiler::AbsBound qkv_acc_abs =
      compiler::matmul_accum_abs(have_x_abs ? x_abs_hint : compiler::abs_from_range(q_range, true),
                                 wqkv_abs,
                                 D);
  trunc_params_qkv.abs_hint = qkv_acc_abs;
  if (qkv_acc_abs.kind == compiler::RangeKind::Proof) {
    trunc_params_qkv.gap_hint =
        compiler::gap_from_abs(qkv_acc_abs, fb, compiler::default_mask_bound(fb));
  }
  std::mt19937_64 rng_qkv(123);
  auto trunc_qkv_bundle =
      compiler::lower_truncation_gate(ctx->trunc_backend(), rng_qkv, trunc_params_qkv, qkv.size());
  {
    runtime::PhaseResources truncR;
    runtime::ProtoChanFromNet pch(*pfss_nc);
    truncR.party = party;
    truncR.net_chan = &ch;
    truncR.pfss_backend = &ctx->trunc_backend();
    truncR.pfss_chan = &pch;
    truncR.pfss_trunc = &pe->pfss_coeff_batch();  // share batch for trunc + coeff
    truncR.opens = &pe->open_collector();
    truncR.pfss_planner = &pfss_phase_planner;
    runtime::PfssSuperBatch::Limits pfss_lim;
    pfss_lim.max_pending_jobs = 1ull << 12;
    pfss_lim.max_pending_hatx_words = 1ull << 21;
    pfss_lim.max_flushes = 1ull << 9;
    pe->pfss_trunc_batch().set_limits(pfss_lim);
    runtime::OpenCollector::Limits open_lim;
    open_lim.max_pending_words = 1ull << 22;
    pe->open_collector().set_limits(open_lim);
    pe->set_max_flushes(1ull << 11);
    pe->begin_phase(runtime::PhaseExecutor::Phase::kQKV_Score);
    enter_phase();
    auto trunc_task = std::make_unique<runtime::TruncTask>(
        &trunc_qkv_bundle,
        std::span<const uint64_t>(qkv.data(), qkv.size()),
        std::span<uint64_t>(qkv.data(), qkv.size()));
    pe->add_task(std::move(trunc_task));
    pe->run(truncR);
    barrier(runtime::PfssLayerPlanner::BarrierPolicy{.drain_open = true,
                                                     .drain_pfss_coeff = true,
                                                     .drain_pfss_trunc = true});
    record_phase_plan(pfss_phase_planner);
  }

  std::vector<uint64_t> ctx_shares(B * T * H * Dh, 0);
  // Triple cache for secret matmul paths.
  auto ensure_triples = [&](size_t need) {
    if (ctx && ctx->trunc_ctx) {
      auto& kp = *ctx->trunc_ctx;  // reuse trunc_ctx backend for triple generation if needed
      (void)kp;  // placeholder; in real impl we would pull from a triple provider
    }
  };
  size_t init_len = cache.cur_len;

  gates::NExpGateParams nexp_params;
  nexp_params.frac_bits = fb;
  nexp_params.segments = 16;
  auto nexp_spec = gates::make_nexp_spec(nexp_params);
  compiler::RangeInterval nexp_range = clamp_nexp_range(fb);
  auto recip_spec =
      gates::make_recip_affine_init_spec(fb, static_cast<double>(std::max(cache.S_max, T + init_len)));
  compiler::RangeInterval recip_range = clamp_recip_range(
      fb, static_cast<double>(std::max(cache.S_max, T + init_len)));
  std::optional<gates::NexpTaskMaterial> nexp_mat;
  runtime::CubicPolyBundle nexp_bundle{};
  std::optional<gates::RecipTaskMaterial> recip_mat;
  runtime::RecipTaskBundle recip_bundle{};
  std::optional<compiler::TruncationLoweringResult> prob_gapars;
  std::optional<compiler::TruncationLoweringResult> prob_faithful;
  runtime::TruncChoice prob_choice{};
  runtime::PhaseResources phase_R;
  phase_R.party = party;
  phase_R.net_chan = &ch;
  runtime::ProtoChanFromNet pch(*pfss_nc);
  runtime::PfssPhasePlanner phase_planner;
  std::mt19937_64 rng(123);
  if (use_phase_softmax) {
    runtime::PfssSuperBatch::Limits pfss_lim;
    pfss_lim.max_pending_jobs = 1ull << 12;
    pfss_lim.max_pending_hatx_words = 1ull << 21;
    pfss_lim.max_flushes = 1ull << 9;
    pe->pfss_coeff_batch().set_limits(pfss_lim);
    pe->pfss_trunc_batch().set_limits(pfss_lim);
    runtime::OpenCollector::Limits open_lim;
    open_lim.max_pending_words = 1ull << 22;
    pe->open_collector().set_limits(open_lim);
    pe->set_max_flushes(1ull << 11);
    size_t triple_need = 3 * cache.S_max * B * H;
    nexp_mat = gates::dealer_make_nexp_task_material(ctx->trunc_backend(),
                                                     nexp_params,
                                                     rng,
                                                     triple_need,
                                                     static_cast<size_t>(B * H * cache.S_max));
    nexp_bundle = gates::make_nexp_cubic_bundle(*nexp_mat, fb);
    recip_mat = gates::dealer_make_recip_task_material(ctx->trunc_backend(),
                                                       fb,
                                                       /*nr_iters=*/1,
                                                       rng,
                                                       static_cast<size_t>(B * H));
    recip_bundle = gates::make_recip_bundle(*recip_mat);
      compiler::GateParams gap_p;
      gap_p.kind = compiler::GateKind::AutoTrunc;
      gap_p.frac_bits = fb;
      gap_p.range_hint = clamp_softmax_range(fb);
      gap_p.per_element_masks = true;
      gap_p.abs_hint = compiler::AbsBound{
          /*is_signed=*/true,
          static_cast<uint64_t>(1ull << fb),
          compiler::RangeKind::Proof};
      gap_p.gap_hint =
          compiler::gap_from_abs(gap_p.abs_hint, fb, compiler::default_mask_bound(fb));
    prob_gapars = compiler::lower_truncation_gate(
        ctx->trunc_backend(), rng, gap_p, static_cast<size_t>(B * H * cache.S_max));
    compiler::GateParams faithful_p;
    faithful_p.kind = compiler::GateKind::FaithfulTR;
    faithful_p.frac_bits = fb;
    prob_faithful = compiler::lower_truncation_gate(
        ctx->trunc_backend(), rng, faithful_p, static_cast<size_t>(B * H * cache.S_max));
    prob_choice.gapars = prob_gapars ? &*prob_gapars : nullptr;
    prob_choice.faithful = prob_faithful ? &*prob_faithful : nullptr;
    prob_choice.shift_bits = fb;
    prob_choice.signed_value = false;  // probabilities are non-negative
    phase_R.pfss_backend = &ctx->trunc_backend();
    phase_R.pfss_chan = &pch;
    phase_R.pfss_coeff = &pe->pfss_coeff_batch();
    phase_R.pfss_trunc = phase_R.pfss_coeff;
    phase_R.opens = &pe->open_collector();
    phase_planner.bind(phase_R.pfss_coeff, phase_R.pfss_trunc);
    phase_R.pfss_planner = &phase_planner;
  }

  int64_t inv_sqrt = static_cast<int64_t>(
      std::llround((1.0 / std::sqrt(static_cast<double>(Dh))) * std::ldexp(1.0, fb)));
  if (inv_sqrt == 0) inv_sqrt = 1;

  std::vector<uint64_t> stepK(B * H * Dh, 0), stepV(B * H * Dh, 0);
  for (size_t t = 0; t < T; ++t) {
    // Slice K/V for this token.
    for (size_t b = 0; b < B; ++b) {
      size_t base = (b * T + t) * 3 * D;
      const uint64_t* k_src = qkv.data() + base + D;
      const uint64_t* v_src = qkv.data() + base + 2 * D;
      for (size_t h = 0; h < H; ++h) {
        for (size_t d = 0; d < Dh; ++d) {
          size_t idx = (b * H + h) * Dh + d;
          stepK[idx] = k_src[h * Dh + d];
          stepV[idx] = v_src[h * Dh + d];
        }
      }
    }

    kv_append_token(cache, view3(stepK.data(), B, H, Dh), view3(stepV.data(), B, H, Dh));
    size_t cur_len = cache.cur_len;

    size_t rows = B * H;
    size_t cols = cur_len;

    if (!use_phase_softmax) {
      // Legacy plaintext softmax path: open Q/K/V and compute context locally, keeping shares simple.
      for (size_t b = 0; b < B; ++b) {
        size_t q_base = (b * T + t) * 3 * D;
        const uint64_t* q_ptr = qkv.data() + q_base;
        for (size_t h = 0; h < H; ++h) {
          std::vector<int64_t> q_plain;
          open_to_plain(party, ch, q_ptr + h * Dh, Dh, q_plain);

          const uint64_t* k_head = kv_head_ptr(cache, b, h);
          const uint64_t* v_head = kv_head_ptr_v(cache, b, h);
          std::vector<int64_t> k_plain, v_plain;
          open_to_plain(party, ch, k_head, cur_len * Dh, k_plain);
          open_to_plain(party, ch, v_head, cur_len * Dh, v_plain);

          std::vector<int64_t> scores(cur_len, 0);
          for (size_t s = 0; s < cur_len; ++s) {
            __int128 acc = 0;
            for (size_t d = 0; d < Dh; ++d) {
              acc += static_cast<__int128>(q_plain[d]) *
                     static_cast<__int128>(k_plain[s * Dh + d]);
            }
            acc >>= fb;
            acc = (acc * static_cast<__int128>(inv_sqrt)) >> fb;
            scores[s] = static_cast<int64_t>(acc);
          }

          int64_t max_sc = scores.empty() ? 0 : *std::max_element(scores.begin(), scores.end());
          std::vector<int64_t> expv(cur_len, 0);
          int64_t sum = 0;
          for (size_t s = 0; s < cur_len; ++s) {
            int64_t diff = max_sc - scores[s];
            expv[s] = gates::ref_nexp_fixed(nexp_spec, diff);
            sum += expv[s];
          }
          if (sum == 0) sum = 1;
          int64_t inv = gates::ref_reciprocal_fixed(recip_spec, sum, fb, 1);

          std::vector<int64_t> prob(cur_len, 0);
          for (size_t s = 0; s < cur_len; ++s) {
            __int128 p = static_cast<__int128>(expv[s]) * static_cast<__int128>(inv);
            prob[s] = static_cast<int64_t>(p >> fb);
          }

          size_t ctx_base = ((b * T + t) * H + h) * Dh;
          for (size_t d = 0; d < Dh; ++d) {
            __int128 acc = 0;
            for (size_t s = 0; s < cur_len; ++s) {
              acc += static_cast<__int128>(prob[s]) *
                     static_cast<__int128>(v_plain[s * Dh + d]);
            }
            int64_t ctx_plain = static_cast<int64_t>(acc >> fb);
            ctx_shares[ctx_base + d] = (party == 0) ? to_ring(ctx_plain) : 0;
          }
        }
      }
      continue;
    }

    std::vector<uint64_t> t_share;
    std::vector<int> plan_valid_lens;
    if (use_phase_softmax && rows > 0) {
      t_share.resize(rows * cols, 0);
      // Track the valid prefix length for each row (causal masking).
      // Softmax task will skip work beyond valid_lens[r].
      plan_valid_lens.resize(rows, static_cast<int>(cur_len));
    }

    // Compute scores = (Q*K^T) / sqrt(Dh) on shares using MatmulTask.
    // Shapes: Q [rows x Dh], K [rows x cur_len x Dh] stored as contiguous per head.
    std::vector<uint64_t> q_mat(rows * Dh, 0);
    std::vector<uint64_t> k_mat(rows * cur_len * Dh, 0);
    std::vector<uint64_t> score_share(rows * cur_len, 0);
    // Fill Q/K share matrices.
    for (size_t b = 0; b < B; ++b) {
      size_t q_base = (b * T + t) * 3 * D;
      const uint64_t* q_ptr = qkv.data() + q_base;
      for (size_t h = 0; h < H; ++h) {
        size_t row = b * H + h;
        for (size_t d = 0; d < Dh; ++d) {
          q_mat[row * Dh + d] = q_ptr[h * Dh + d];
        }
        const uint64_t* k_head = kv_head_ptr(cache, b, h);
        for (size_t s = 0; s < cur_len; ++s) {
          for (size_t d = 0; d < Dh; ++d) {
            size_t idx = (row * cur_len + s) * Dh + d;
            k_mat[idx] = k_head[s * Dh + d];
          }
        }
      }
    }

    // Need Dh * (rows*cur_len) triples for the matmul products.
    static std::unordered_map<uint64_t, std::vector<proto::BeaverTriple64Share>> score_cache_p0;
    static std::unordered_map<uint64_t, std::vector<proto::BeaverTriple64Share>> score_cache_p1;
    auto& score_cache = (party == 0) ? score_cache_p0 : score_cache_p1;
    auto& score_triples = ensure_cached_triples(score_cache, 0x73636f72u /*"scor"*/, rows * cur_len * Dh, party);

    // Phase: score matmul (unscaled Q2f).
    pe->begin_phase(runtime::PhaseExecutor::Phase::kQKV_Score);
    enter_phase();
    for (size_t row = 0; row < rows; ++row) {
      auto q_row = std::span<const uint64_t>(q_mat.data() + row * Dh, Dh);
      auto k_row = std::span<const uint64_t>(k_mat.data() + row * cur_len * Dh, cur_len * Dh);
      auto out_row = std::span<uint64_t>(score_share.data() + row * cur_len, cur_len);
      auto tri_row = std::span<const proto::BeaverTriple64Share>(
          score_triples.data() + row * cur_len * Dh, cur_len * Dh);
      auto score_task = std::make_unique<runtime::MatmulTask>(
          /*M=*/1, static_cast<int>(Dh), static_cast<int>(cur_len), q_row, k_row, out_row, tri_row);
      pe->add_task(std::move(score_task));
    }
    pe->run(phase_R);
    barrier(runtime::PfssLayerPlanner::BarrierPolicy{.drain_open = true,
                                                     .drain_pfss_coeff = true,
                                                     .drain_pfss_trunc = true});
    record_phase_plan(phase_planner);

    // Rescale scores Q2f -> Qf via truncation (no inline shift).
    std::vector<uint64_t> score_qf(score_share.size(), 0);
    compiler::GateParams score_trunc_p;
    score_trunc_p.frac_bits = fb;
    compiler::RangeInterval score_acc_range =
        compiler::matmul_accum_range(q_range, q_range, Dh);
    score_trunc_p.kind = compiler::GateKind::AutoTrunc;
    score_trunc_p.range_hint = score_acc_range;
    score_trunc_p.per_element_masks = true;
    compiler::AbsBound q_abs = have_qkv_abs ? qkv_abs_hint : compiler::abs_from_range(q_range, true);
    compiler::AbsBound score_acc_abs = compiler::matmul_accum_abs(q_abs, q_abs, Dh);
    score_trunc_p.abs_hint = score_acc_abs;
    if (score_acc_abs.kind == compiler::RangeKind::Proof) {
      score_trunc_p.gap_hint =
          compiler::gap_from_abs(score_acc_abs, fb, compiler::default_mask_bound(fb));
    }
    std::mt19937_64 rng_score(0x73636f72u);
    auto score_trunc_bundle =
        compiler::lower_truncation_gate(ctx->trunc_backend(), rng_score, score_trunc_p, score_share.size());
    {
      pe->begin_phase(runtime::PhaseExecutor::Phase::kQKV_Score);
      enter_phase();
      auto trunc_task = std::make_unique<runtime::TruncTask>(
          &score_trunc_bundle,
          std::span<const uint64_t>(score_share.data(), score_share.size()),
          std::span<uint64_t>(score_qf.data(), score_qf.size()));
      pe->add_task(std::move(trunc_task));
      pe->run(phase_R);
      barrier(runtime::PfssLayerPlanner::BarrierPolicy{.drain_open = true,
                                                       .drain_pfss_coeff = true,
                                                       .drain_pfss_trunc = true});
      record_phase_plan(phase_planner);
    }

    // Scale by inv_sqrt (public Qf): score_qf (Qf) * inv_sqrt (Qf) -> Q2f, then trunc to Qf.
    std::vector<uint64_t> score_scaled_q2f(score_qf.size(), 0);
    for (size_t i = 0; i < score_qf.size(); ++i) {
      __int128 prod = static_cast<__int128>(to_signed(score_qf[i])) *
                      static_cast<__int128>(inv_sqrt);
      score_scaled_q2f[i] = to_ring(static_cast<int64_t>(prod));
    }
    std::vector<uint64_t> score_scaled_qf(score_qf.size(), 0);
    {
      pe->begin_phase(runtime::PhaseExecutor::Phase::kQKV_Score);
      enter_phase();
      auto trunc_task = std::make_unique<runtime::TruncTask>(
          &score_trunc_bundle,
          std::span<const uint64_t>(score_scaled_q2f.data(), score_scaled_q2f.size()),
          std::span<uint64_t>(score_scaled_qf.data(), score_scaled_qf.size()));
      pe->add_task(std::move(trunc_task));
      pe->run(phase_R);
      barrier(runtime::PfssLayerPlanner::BarrierPolicy{.drain_open = true,
                                                       .drain_pfss_coeff = true,
                                                       .drain_pfss_trunc = true});
      record_phase_plan(phase_planner);
    }

    // Scale scores by inv_sqrt (public scalar) and prepare softmax inputs.
    // Explicit Rescale is required; legacy inline shift is disallowed.
    for (size_t row = 0; row < rows; ++row) {
      uint64_t max_share = 0;
      for (size_t s = 0; s < cur_len; ++s) {
        size_t idx = row * cur_len + s;
        uint64_t v = score_scaled_qf[idx];
        score_share[idx] = v;
        if (v > max_share) max_share = v;
      }
      if (use_phase_softmax) {
        int64_t cap = static_cast<int64_t>(16ll << fb);
        size_t base = row * cols;
        for (size_t s = 0; s < cur_len; ++s) {
          int64_t diff = to_signed(max_share) - to_signed(score_share[base + s]);
          if (diff < 0) diff = 0;
          if (diff > cap) diff = cap;
          t_share[base + s] = (party == 0) ? to_ring(diff) : 0ull;
        }
      }
    }

    std::vector<uint64_t> prob_shares;
    if (use_phase_softmax && rows > 0) {
      static CachedRowBroadcastTripleProvider row_triples_p0(/*party=*/0);
      static CachedRowBroadcastTripleProvider row_triples_p1(/*party=*/1);
      auto& row_triples = (party == 0) ? row_triples_p0 : row_triples_p1;

      runtime::TruncChoice choice = prob_choice;
      if (!choice.faithful) choice.faithful = recip_bundle.trunc_fb;
      if (!choice.gapars) choice.gapars = choice.faithful;

      prob_shares.resize(rows * cols, 0);
      nn::SoftmaxPlan plan;
      plan.frac_bits = fb;
      plan.rows = static_cast<int>(rows);
      plan.cols = static_cast<int>(cols);
      plan.valid_lens = plan_valid_lens;
      plan.nexp = nexp_bundle;
      plan.recip = recip_bundle;
      plan.prob_trunc = choice;
      plan.row_triples = &row_triples;
      plan.prob_range = clamp_softmax_range(fb);
      compiler::RangeInterval prob_range = plan.prob_range.value();
      // Clamp reciprocal init/output range based on max sum bound if provided.
      compiler::RangeInterval recip_range = clamp_recip_range(fb, cfg.recip_max_sum);
      (void)recip_range;  // clamp available for future range-aware selection

      // Build faithful trunc bundle for prob*V (Q2f -> Qf).
    compiler::GateParams ctx_trunc_p;
    ctx_trunc_p.kind = compiler::GateKind::AutoTrunc;
    ctx_trunc_p.frac_bits = fb;
    ctx_trunc_p.range_hint = clamp_softmax_range(fb);
      ctx_trunc_p.abs_hint = compiler::AbsBound{
          /*is_signed=*/true,
          static_cast<uint64_t>(1ull << fb),
          compiler::RangeKind::Proof};
      ctx_trunc_p.gap_hint =
          compiler::gap_from_abs(ctx_trunc_p.abs_hint, fb, compiler::default_mask_bound(fb));
      std::mt19937_64 rng_ctx(77);
      auto ctx_trunc = compiler::lower_truncation_gate(
          ctx->trunc_backend(), rng_ctx, ctx_trunc_p, rows * Dh);

      // Prepare per-row V material ahead of time.
      std::vector<std::vector<uint64_t>> v_mats(rows);
      for (size_t row = 0; row < rows; ++row) {
        size_t b = row / H;
        size_t h = row % H;
        const uint64_t* v_head = kv_head_ptr_v(cache, b, h);
        v_mats[row].resize(cur_len * Dh);
        for (size_t s = 0; s < cur_len; ++s) {
          for (size_t d = 0; d < Dh; ++d) {
            v_mats[row][s * Dh + d] = v_head[s * Dh + d];
          }
        }
      }

      // Matmul+trunc rows: drive after softmax inside one executor run.
      class SoftmaxProbTask final : public runtime::detail::PhaseTask {
       public:
        SoftmaxProbTask(std::unique_ptr<nn::SoftmaxBlockTask> sm_task,
                        runtime::TruncChoice trunc_choice,
                        compiler::RangeInterval prob_range,
                        size_t rows,
                        size_t cols,
                        size_t Dh,
                        size_t cur_len,
                        std::vector<uint64_t>& prob_shares,
                        std::vector<std::vector<uint64_t>>& v_mats,
                        std::vector<uint64_t>& ctx_shares,
                        KVCache& cache,
                        int H,
                        size_t T,
                        size_t t,
                        int party,
                        int fb)
            : sm_task_(std::move(sm_task)),
              trunc_choice_(trunc_choice),
              prob_range_(prob_range),
              rows_(rows),
              cols_(cols),
              Dh_(Dh),
              cur_len_(cur_len),
              prob_shares_(prob_shares),
              v_mats_(v_mats),
              ctx_shares_(ctx_shares),
              cache_(cache),
              H_(H),
              T_(T),
              t_(t),
              party_(party),
              fb_(fb) {}

        bool done() const override { return st_ == St::Done; }

        runtime::detail::Need step(runtime::PhaseResources& R) override {
          switch (st_) {
            case St::Softmax: {
              auto need = sm_task_->step(R);
              if (!sm_task_->done()) return need;
              st_ = St::Rows;
              return runtime::detail::Need::None;
            }
            case St::Rows: {
              if (row_cursor_ >= rows_) {
                st_ = St::Done;
                return runtime::detail::Need::None;
              }
              if (!row_task_) {
                size_t row = row_cursor_;
                size_t b = row / static_cast<size_t>(H_);
                size_t h = row % static_cast<size_t>(H_);
                std::span<const uint64_t> prob_row(prob_shares_.data() + row * cols_, cols_);
                static std::unordered_map<uint64_t, std::vector<proto::BeaverTriple64Share>> probv_cache_p0;
                static std::unordered_map<uint64_t, std::vector<proto::BeaverTriple64Share>> probv_cache_p1;
                auto& probv_cache = (party_ == 0) ? probv_cache_p0 : probv_cache_p1;
                auto& triples = ensure_cached_triples(probv_cache, 0x70726f62u /*"prob"*/, cur_len_ * Dh_, party_);
                size_t ctx_off = ((b * T_ + t_) * static_cast<size_t>(H_) + h) * Dh_;
                std::span<uint64_t> ctx_row(ctx_shares_.data() + ctx_off, Dh_);
                auto mm = std::make_unique<runtime::MatmulTask>(
                    1, static_cast<int>(cur_len_), static_cast<int>(Dh_), prob_row,
                    std::span<const uint64_t>(v_mats_[row].data(), v_mats_[row].size()), ctx_row,
                    std::span<const proto::BeaverTriple64Share>(triples.data(), triples.size()));
                const auto* trunc_bundle =
                    select_trunc_bundle(trunc_choice_, prob_range_, static_cast<int>(fb_));
                auto trunc = std::make_unique<runtime::TruncTask>(
                    trunc_bundle ? trunc_bundle : trunc_choice_.faithful,
                    std::span<const uint64_t>(ctx_row.data(), ctx_row.size()), ctx_row);
                row_task_ = std::make_unique<SeqTask>(std::move(mm), std::move(trunc));
              }
              auto need = row_task_->step(R);
              if (!row_task_->done()) return need;
              row_task_.reset();
              ++row_cursor_;
              return runtime::detail::Need::None;
            }
            case St::Done:
              return runtime::detail::Need::None;
          }
          return runtime::detail::Need::None;
        }

       private:
        class SeqTask final : public runtime::detail::PhaseTask {
         public:
          SeqTask(std::unique_ptr<runtime::detail::PhaseTask> first,
                  std::unique_ptr<runtime::detail::PhaseTask> second)
              : first_(std::move(first)), second_(std::move(second)) {}
          bool done() const override { return state_ == St::Done; }
          runtime::detail::Need step(runtime::PhaseResources& R) override {
            switch (state_) {
              case St::First: {
                auto need = first_->step(R);
                if (!first_->done()) return need;
                state_ = St::Second;
                return runtime::detail::Need::None;
              }
              case St::Second: {
                auto need = second_->step(R);
                if (!second_->done()) return need;
                state_ = St::Done;
                return runtime::detail::Need::None;
              }
              case St::Done:
                return runtime::detail::Need::None;
            }
            return runtime::detail::Need::None;
          }

         private:
          enum class St { First, Second, Done } state_ = St::First;
          std::unique_ptr<runtime::detail::PhaseTask> first_;
          std::unique_ptr<runtime::detail::PhaseTask> second_;
        };

        enum class St { Softmax, Rows, Done } st_ = St::Softmax;
        std::unique_ptr<nn::SoftmaxBlockTask> sm_task_;
        runtime::TruncChoice trunc_choice_{};
        compiler::RangeInterval prob_range_{};
        size_t rows_ = 0;
        size_t cols_ = 0;
        size_t Dh_ = 0;
        size_t cur_len_ = 0;
        std::vector<uint64_t>& prob_shares_;
        std::vector<std::vector<uint64_t>>& v_mats_;
        std::vector<uint64_t>& ctx_shares_;
        KVCache& cache_;
        int H_ = 0;
        size_t T_ = 0;
        size_t t_ = 0;
        int party_ = 0;
        int fb_ = 0;
        size_t row_cursor_ = 0;
        std::unique_ptr<SeqTask> row_task_;
      };

      pe->begin_phase(runtime::PhaseExecutor::Phase::kSoftmax);
      enter_phase();
      auto sm_task = std::make_unique<nn::SoftmaxBlockTask>(
          plan,
          std::span<const uint64_t>(t_share.data(), t_share.size()),
          std::span<uint64_t>(prob_shares.data(), prob_shares.size()));
      pe->add_task(std::make_unique<SoftmaxProbTask>(std::move(sm_task),
                                                     prob_choice,
                                                     prob_range,
                                                     rows,
                                                     cols,
                                                     Dh,
                                                     cur_len,
                                                     prob_shares,
                                                     v_mats,
                                                     ctx_shares,
                                                     cache,
                                                     static_cast<int>(H),
                                                     T,
                                                     t,
                                                     party,
                                                     fb));
      pe->run(phase_R);
      barrier(runtime::PfssLayerPlanner::BarrierPolicy{.drain_open = true,
                                                       .drain_pfss_coeff = true,
                                                       .drain_pfss_trunc = true});
      record_phase_plan(phase_planner);
    }
  }

  std::vector<uint64_t> merged(B * T * D, 0);
  for (size_t b = 0; b < B; ++b) {
    for (size_t t = 0; t < T; ++t) {
      for (size_t h = 0; h < H; ++h) {
        for (size_t d = 0; d < Dh; ++d) {
          size_t dst = (b * T + t) * D + h * Dh + d;
          size_t src = ((b * T + t) * H + h) * Dh + d;
          merged[dst] = ctx_shares[src];
        }
      }
    }
  }

  matmul_publicW(view2(merged.data(), B * T, D),
                 Wout_public,
                 Y_share,
                 mp);
  // Truncate output accum (Q2f -> Qf) via composite truncation.
  compiler::RangeInterval prob_r = clamp_softmax_range(fb);
  compiler::RangeInterval ctx_r =
      compiler::propagate_matmul_out(prob_r, q_range, static_cast<size_t>(std::max(cache.S_max, T)), fb);
  compiler::RangeInterval wout_r = range_from_public_weights(Wout_public);
  int row_l1_out = row_l1_max(Wout_public, mp.w_transposed);
  compiler::RangeInterval out_acc =
      (row_l1_out > 0)
          ? compiler::propagate_matmul_accum_rowl1(ctx_r, row_l1_out)
          : compiler::propagate_matmul_accum(ctx_r, wout_r, D);
  compiler::GateParams trunc_params_out;
  trunc_params_out.kind = compiler::GateKind::AutoTrunc;
  trunc_params_out.frac_bits = fb;
  trunc_params_out.range_hint = out_acc;
  compiler::AbsBound ctx_abs = compiler::abs_from_range(ctx_r, ctx_r.is_signed);
  ctx_abs.kind = have_qkv_abs ? compiler::RangeKind::Proof : compiler::RangeKind::Hint;
  compiler::AbsBound w_abs = compiler::abs_from_range(wout_r, true);
  w_abs.kind = compiler::RangeKind::Proof;
  compiler::AbsBound out_abs = compiler::matmul_accum_abs(ctx_abs, w_abs, D);
  trunc_params_out.abs_hint = out_abs;
  if (out_abs.kind == compiler::RangeKind::Proof) {
    trunc_params_out.gap_hint =
        compiler::gap_from_abs(out_abs, fb, compiler::default_mask_bound(fb));
  }
  std::mt19937_64 rng_out(456);
  auto trunc_out_bundle =
      compiler::lower_truncation_gate(ctx->trunc_backend(), rng_out, trunc_params_out, Y_share.numel());
  {
    runtime::PhaseResources truncR;
    runtime::ProtoChanFromNet pch(*pfss_nc);
    truncR.party = party;
    truncR.net_chan = &ch;
    truncR.pfss_backend = &ctx->trunc_backend();
    truncR.pfss_chan = &pch;
    truncR.pfss_trunc = &pe->pfss_coeff_batch();  // share batch for trunc + coeff
    truncR.opens = &pe->open_collector();
    runtime::PfssPhasePlanner planner;
    planner.bind(&pe->pfss_coeff_batch(), truncR.pfss_trunc);
    truncR.pfss_planner = &planner;
    pe->begin_phase(runtime::PhaseExecutor::Phase::kOutProj);
    enter_phase();
    pe->add_task(std::make_unique<runtime::TruncTask>(
        &trunc_out_bundle,
        std::span<const uint64_t>(Y_share.data, Y_share.numel()),
        std::span<uint64_t>(Y_share.data, Y_share.numel())));
    pe->run(truncR);
      barrier(runtime::PfssLayerPlanner::BarrierPolicy{.drain_open = true,
                                                       .drain_pfss_coeff = true,
                                                       .drain_pfss_trunc = true});
    record_phase_plan(planner);
  }
}

}  // namespace nn
