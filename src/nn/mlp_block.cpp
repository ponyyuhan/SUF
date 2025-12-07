#include "nn/mlp_block.hpp"

#include <vector>
#include <random>
#include <limits>
#include "gates/tables/silu_spline_table.hpp"
#include "gates/silu_composite.hpp"
#include "runtime/pfss_superbatch.hpp"
#include "runtime/phase_tasks.hpp"
#include "runtime/pfss_phase_planner.hpp"
#include "proto/pfss_utils.hpp"
#include "compiler/matmul_truncation.hpp"
#include "runtime/phase_executor.hpp"

namespace nn {

using gates::make_silu_spec;
using gates::ref_silu_fixed;

static inline int64_t to_signed(uint64_t v) { return static_cast<int64_t>(v); }
static inline uint64_t to_ring(int64_t v) { return static_cast<uint64_t>(v); }

void mlp_forward(const MLPConfig& cfg,
                 const TensorView<uint64_t>& X_share,
                 const TensorView<int64_t>& W1_public,
                 const TensorView<int64_t>& W2_public,
                 TensorView<uint64_t> Y_share,
                 int party,
                 net::Chan& ch,
                 LayerContext* ctx,
                 runtime::PhaseExecutor* pe) {
  size_t B = X_share.shape[0];
  size_t T = X_share.shape[1];
  size_t D = cfg.D;
  size_t H = cfg.Hidden;
  MatmulParams mp;
  mp.frac_bits = cfg.frac_bits;
  mp.local_rescale = false;  // explicit rescale via truncation only
  mp.allow_legacy_shift = false;
  if (!ctx) {
    throw std::runtime_error("mlp_forward: LayerContext required (no local rescale fallback)");
  }
  compiler::RangeInterval x_range_hint = compiler::RangeInterval::whole(true);
  if (ctx && !ctx->graph.tensors().empty()) {
    x_range_hint = ctx->graph.tensors().back().range;
  }
  net::Chan* pfss_nc = (ctx && ctx->pfss_net_chan) ? ctx->pfss_net_chan : &ch;
  auto record_phase_plan = [&](runtime::PfssPhasePlanner& planner) {
    if (!ctx || !ctx->pfss_layer_planner) return;
    const auto& st = planner.stats();
    if (st.coeff_jobs == 0 && st.trunc_jobs == 0 && st.coeff_flushes == 0 && st.trunc_flushes == 0) return;
    ctx->pfss_layer_planner->record_phase(planner, pe->pfss_coeff_batch(), pe->pfss_trunc_batch());
    if (party == 0) {
      std::cerr << "[pfss-phase][mlp] coeff_jobs=" << st.coeff_jobs
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
    if (ctx && ctx->pfss_layer_planner) {
      runtime::ProtoChanFromNet pch_bar(*pfss_nc);
      ctx->pfss_layer_planner->barrier(
          party,
          ctx->trunc_ctx->backend(),
          pe->pfss_coeff_batch(),
          pe->pfss_trunc_batch(),
          pch_bar,
          &pe->open_collector(),
          &ch,
          pol);
    }
  };

  std::vector<uint64_t> hidden(B * T * H, 0);
  compiler::RangeInterval mat1_x_range;
  compiler::RangeInterval mat1_w_range;
  compiler::RangeInterval mat2_x_range;
  compiler::RangeInterval mat2_w_range;
  int row_l1_max1 = 0;
  int row_l1_max2 = 0;
  compiler::RangeInterval mat1_out_range = compiler::RangeInterval::whole(true);
  compiler::RangeInterval mat2_out_range = compiler::RangeInterval::whole(true);
  if (ctx) {
    compiler::Scale q_scale = make_scale(cfg.frac_bits, true);
    SecretTensor x_t = make_secret_tensor(ctx, X_share, q_scale, x_range_hint);

    compiler::MatmulAttrs mat1;
    mat1.M = B * T;
    mat1.K = D;
    mat1.N = H;
    mat1.w_transposed = mp.w_transposed;
    mat1.params = nullptr;  // public W path; tracked for rescale only.
    mat1.frac_bits = cfg.frac_bits;
    mat1.row_l1_max = row_l1_max(W1_public, mp.w_transposed);
    row_l1_max1 = mat1.row_l1_max;
    mat1.w_range = range_from_public_weights(W1_public);
    mat1.x_range = x_t.range;
    mat1_x_range = mat1.x_range;
    mat1_w_range = mat1.w_range;
    auto acc1 =
        record_matmul(ctx, x_t, mat1, make_scale(2 * cfg.frac_bits, true),
                      mat1.row_l1_max > 0
                          ? compiler::propagate_matmul_accum_rowl1(x_t.range, mat1.row_l1_max)
                          : compiler::propagate_matmul_accum(x_t.range, mat1.w_range, mat1.K),
                      view2(hidden.data(), B * T, H));

    compiler::RangeInterval hidden_range =
        compiler::propagate_matmul_out(x_t.range, mat1.w_range, mat1.K, cfg.frac_bits);
    mat1_out_range = hidden_range;
    compiler::RescaleAttrs r1;
    r1.matmul_op = acc1.producer_op;
    r1.from_frac = 2 * cfg.frac_bits;
    r1.to_frac = cfg.frac_bits;
    auto hidden_t = record_rescale(ctx, acc1, r1, q_scale, hidden_range,
                                   view2(hidden.data(), B * T, H));
    // SiLU is tightly bounded; record a clamp so downstream matmul can exploit the gap cert.
    compiler::RangeInterval silu_range = clamp_silu_range(cfg.frac_bits);
    auto silu_t = record_clamp(ctx, hidden_t, silu_range, q_scale);

    compiler::MatmulAttrs mat2;
    mat2.M = B * T;
    mat2.K = H;
    mat2.N = D;
    mat2.w_transposed = mp.w_transposed;
    mat2.params = nullptr;
    mat2.frac_bits = cfg.frac_bits;
    mat2.row_l1_max = row_l1_max(W2_public, mp.w_transposed);
    row_l1_max2 = mat2.row_l1_max;
    mat2.w_range = range_from_public_weights(W2_public);
    mat2.x_range = silu_t.range;
    mat2_x_range = mat2.x_range;
    mat2_w_range = mat2.w_range;
    auto acc2 =
        record_matmul(ctx, silu_t, mat2, make_scale(2 * cfg.frac_bits, true),
                      mat2.row_l1_max > 0
                          ? compiler::propagate_matmul_accum_rowl1(silu_t.range, mat2.row_l1_max)
                          : compiler::propagate_matmul_accum(silu_t.range, mat2.w_range, mat2.K),
                      Y_share);

    compiler::RangeInterval out_range =
        compiler::propagate_matmul_out(silu_t.range, mat2.w_range, mat2.K, cfg.frac_bits);
    mat2_out_range = out_range;
    compiler::RescaleAttrs r2;
    r2.matmul_op = acc2.producer_op;
    r2.from_frac = 2 * cfg.frac_bits;
    r2.to_frac = cfg.frac_bits;
    (void)record_rescale(ctx, acc2, r2, q_scale, out_range, Y_share);
  }
  matmul_publicW(view2(const_cast<uint64_t*>(X_share.data), B * T, D),
                 W1_public,
                 view2(hidden.data(), B * T, H),
                 mp);

  bool use_phase = (ctx && pe && ctx->trunc_ctx);
  std::vector<uint64_t> hidden_scaled = hidden;
  runtime::PhaseResources R;
  runtime::ProtoChanFromNet pch(*pfss_nc);
  runtime::PfssPhasePlanner pfss_phase_planner;
  if (use_phase) {
    R.party = party;
    R.pfss_backend = &ctx->trunc_ctx->backend();
    R.pfss_chan = &pch;
    R.net_chan = &ch;
    // Share a single PFSS batch for coeff/trunc to fuse flushes.
    R.pfss_coeff = &pe->pfss_coeff_batch();
    R.pfss_trunc = R.pfss_coeff;
    R.opens = &pe->open_collector();
    runtime::PfssSuperBatch::Limits pfss_lim;
    pfss_lim.max_pending_jobs = 1ull << 12;
    pfss_lim.max_pending_hatx_words = 1ull << 21;
    pfss_lim.max_flushes = 1ull << 9;
    pe->pfss_coeff_batch().set_limits(pfss_lim);
    runtime::OpenCollector::Limits open_lim;
    open_lim.max_pending_words = 1ull << 22;
    pe->open_collector().set_limits(open_lim);
    pe->set_max_flushes(1ull << 10);
    pfss_phase_planner.bind(R.pfss_coeff, R.pfss_trunc);
    R.pfss_planner = &pfss_phase_planner;
  }

  if (use_phase) {
    // Faithful truncation via composite (no local shift) batched in phase executor.
    std::mt19937_64 rng(0);
    auto make_plan = [&](size_t M, size_t K, size_t N, const compiler::RangeInterval& x_range,
                         const compiler::RangeInterval& w_range, int row_l1_max) {
      compiler::RangeInterval accum;
      if (row_l1_max > 0) {
        int64_t max_abs_x = std::max(std::llabs(x_range.lo), std::llabs(x_range.hi));
        __int128 prod = static_cast<__int128>(max_abs_x) * static_cast<__int128>(row_l1_max);
        int64_t max_cap = std::numeric_limits<int64_t>::max();
        if (prod > static_cast<__int128>(max_cap)) prod = max_cap;
        int64_t max_abs_acc = static_cast<int64_t>(prod);
        accum.lo = -max_abs_acc;
        accum.hi = max_abs_acc;
        accum.is_signed = true;
      } else {
        accum = compiler::matmul_accum_range(x_range, w_range, K);
      }
      compiler::GateParams p;
      p.frac_bits = cfg.frac_bits;
      p.kind = compiler::GateKind::AutoTrunc;
      p.range_hint = accum;
      compiler::MatmulTruncationPlan plan;
      plan.kind = compiler::select_trunc_kind(accum, cfg.frac_bits);
      plan.accum_range = accum;
      plan.batch = M * N;
      plan.bundle = compiler::lower_truncation_gate(ctx->trunc_ctx->backend(), rng, p, plan.batch);
      return plan;
    };

    auto plan1 = make_plan(B * T, D, H, mat1_x_range, mat1_w_range, row_l1_max1);
    auto trunc_task1 = std::make_unique<runtime::TruncTask>(
        &plan1.bundle, std::span<const uint64_t>(hidden.data(), hidden.size()),
        std::span<uint64_t>(hidden_scaled.data(), hidden_scaled.size()));
    std::mt19937_64 rng2(0);
    std::cerr << "mlp_forward: building silu task material frac_bits=" << cfg.frac_bits
              << " elems=" << hidden_scaled.size() << "\n";
    auto mat = gates::dealer_make_silu_task_material(
        ctx->trunc_ctx->backend(),
        cfg.frac_bits,
        rng2,
        3 * hidden_scaled.size(),
        hidden_scaled.size());
    runtime::CubicPolyBundle bundle{
        &mat.suf, &mat.keys.k0, &mat.keys.k1, &mat.trunc_f, &mat.trunc_2f, cfg.frac_bits};
    auto silu_task = std::make_unique<runtime::CubicPolyTask>(
        bundle,
        std::span<const uint64_t>(hidden_scaled.data(), hidden_scaled.size()),
        std::span<uint64_t>(hidden_scaled.data(), hidden_scaled.size()));
    // Update hidden range to the clamped SiLU range for downstream matmul planning.
    compiler::RangeInterval silu_range = clamp_silu_range(cfg.frac_bits);
    mat2_x_range = silu_range;

    // First wave: trunc hidden matmul accum to Qf.
    pe->begin_phase(runtime::PhaseExecutor::Phase::kLN2_MLP);
    if (ctx && ctx->pfss_layer_planner) ctx->pfss_layer_planner->enter_phase();
    pe->add_task(std::move(trunc_task1));
    pe->run(R);
    barrier(runtime::PfssLayerPlanner::BarrierPolicy{.drain_all = true});
    record_phase_plan(pfss_phase_planner);

    // Second wave: apply SiLU cubic on the truncated hidden.
    pe->begin_phase(runtime::PhaseExecutor::Phase::kLN2_MLP);
    if (ctx && ctx->pfss_layer_planner) ctx->pfss_layer_planner->enter_phase();
    pe->add_task(std::move(silu_task));
    pe->run(R);
    barrier(runtime::PfssLayerPlanner::BarrierPolicy{.drain_all = true});
    record_phase_plan(pfss_phase_planner);

    // Linear 2 on the updated hidden.
    matmul_publicW(view2(hidden_scaled.data(), B * T, H),
                   W2_public,
                   view2(Y_share.data, B * T, D),
                   mp);

    // Second wave: trunc output matmul accum.
    std::mt19937_64 rng_out(1);
    auto plan2 = make_plan(B * T, H, D, mat2_x_range, mat2_w_range, row_l1_max2);
    std::vector<uint64_t> y_scaled(Y_share.numel(), 0);
    pe->begin_phase(runtime::PhaseExecutor::Phase::kLN2_MLP);
    if (ctx && ctx->pfss_layer_planner) ctx->pfss_layer_planner->enter_phase();
    pe->add_task(std::make_unique<runtime::TruncTask>(
        &plan2.bundle, std::span<const uint64_t>(Y_share.data, Y_share.numel()),
        std::span<uint64_t>(y_scaled.data(), y_scaled.size())));
    pe->run(R);
    barrier(runtime::PfssLayerPlanner::BarrierPolicy{.drain_all = true});
    record_phase_plan(pfss_phase_planner);
    for (size_t i = 0; i < Y_share.numel() && i < y_scaled.size(); ++i) {
      Y_share.data[i] = y_scaled[i];
    }
  } else {
    throw std::runtime_error("mlp_forward: phase executor + truncation required (legacy path disabled)");
  }
}

}  // namespace nn
