#include "nn/mlp_block.hpp"

#include <vector>
#include <random>
#include "gates/tables/silu_spline_table.hpp"
#include "gates/silu_composite.hpp"
#include "runtime/pfss_superbatch.hpp"
#include "runtime/phase_tasks.hpp"
#include "proto/pfss_utils.hpp"
#include "compiler/matmul_truncation.hpp"
#include "runtime/phase_executor.hpp"

namespace nn {

using gates::make_silu_spec;
using gates::ref_silu_fixed;

static inline int64_t to_signed(uint64_t v) { return static_cast<int64_t>(v); }
static inline uint64_t to_ring(int64_t v) { return static_cast<uint64_t>(v); }

static inline void rescale_buffer(std::vector<uint64_t>& buf, int frac_bits) {
  if (frac_bits <= 0) return;
  for (auto& v : buf) {
    int64_t s = to_signed(v);
    v = to_ring(s >> frac_bits);
  }
}

static inline void rescale_view(const TensorView<uint64_t>& t, int frac_bits) {
  if (frac_bits <= 0) return;
  size_t n = t.numel();
  for (size_t i = 0; i < n; ++i) {
    int64_t s = to_signed(t.data[i]);
    t.data[i] = to_ring(s >> frac_bits);
  }
}

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
  mp.local_rescale = (ctx == nullptr);  // explicit rescale when ctx is provided.
  mp.allow_legacy_shift = (ctx == nullptr);

  std::vector<uint64_t> hidden(B * T * H, 0);
  compiler::RangeInterval mat1_x_range;
  compiler::RangeInterval mat1_w_range;
  compiler::RangeInterval mat2_x_range;
  compiler::RangeInterval mat2_w_range;
  if (ctx) {
    compiler::Scale q_scale = make_scale(cfg.frac_bits, true);
    compiler::RangeInterval x_range = compiler::RangeInterval::whole(true);
    SecretTensor x_t = make_secret_tensor(ctx, X_share, q_scale, x_range);

    compiler::MatmulAttrs mat1;
    mat1.M = B * T;
    mat1.K = D;
    mat1.N = H;
    mat1.w_transposed = mp.w_transposed;
    mat1.params = nullptr;  // public W path; tracked for rescale only.
    mat1.frac_bits = cfg.frac_bits;
    mat1.row_l1_max = row_l1_max(W1_public, mp.w_transposed);
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
    compiler::RescaleAttrs r1;
    r1.matmul_op = acc1.producer_op;
    r1.from_frac = 2 * cfg.frac_bits;
    r1.to_frac = cfg.frac_bits;
    auto hidden_t = record_rescale(ctx, acc1, r1, q_scale, hidden_range,
                                   view2(hidden.data(), B * T, H));

    compiler::MatmulAttrs mat2;
    mat2.M = B * T;
    mat2.K = H;
    mat2.N = D;
    mat2.w_transposed = mp.w_transposed;
    mat2.params = nullptr;
    mat2.frac_bits = cfg.frac_bits;
    mat2.row_l1_max = row_l1_max(W2_public, mp.w_transposed);
    mat2.w_range = range_from_public_weights(W2_public);
    mat2.x_range = hidden_t.range;
    mat2_x_range = mat2.x_range;
    mat2_w_range = mat2.w_range;
    auto acc2 =
        record_matmul(ctx, hidden_t, mat2, make_scale(2 * cfg.frac_bits, true),
                      mat2.row_l1_max > 0
                          ? compiler::propagate_matmul_accum_rowl1(hidden_t.range, mat2.row_l1_max)
                          : compiler::propagate_matmul_accum(hidden_t.range, mat2.w_range, mat2.K),
                      Y_share);

    compiler::RangeInterval out_range =
        compiler::propagate_matmul_out(hidden_t.range, mat2.w_range, mat2.K, cfg.frac_bits);
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

  bool has_trunc = (ctx && ctx->trunc_ctx);
  std::vector<uint64_t> hidden_scaled;
  if (has_trunc) {
    // Faithful truncation via composite (no local shift).
    std::mt19937_64 rng(0);
    auto plan1 = compiler::compile_matmul_truncation(ctx->trunc_ctx->backend(),
                                                     rng,
                                                     B * T,
                                                     D,
                                                     H,
                                                     cfg.frac_bits,
                                                     mat1_x_range,
                                                     mat1_w_range);
    runtime::ProtoChanFromNet pch(ch);
    runtime::run_truncation_now(party,
                                ctx->trunc_ctx->backend(),
                                pch,
                                plan1.bundle,
                                hidden,
                                hidden_scaled);
  } else {
    hidden_scaled = hidden;
    if (!mp.local_rescale) rescale_buffer(hidden_scaled, cfg.frac_bits);
  }

  bool use_phase_silu = (ctx && pe && ctx->trunc_ctx);
  if (use_phase_silu) {
    std::mt19937_64 rng(0);
    auto mat = gates::dealer_make_silu_task_material(
        ctx->trunc_ctx->backend(), cfg.frac_bits, rng, 3 * hidden_scaled.size());
    runtime::CubicPolyBundle bundle{
        &mat.suf, &mat.keys.k0, &mat.keys.k1, &mat.trunc_f, &mat.trunc_2f, cfg.frac_bits};

    runtime::PhaseResources R;
    runtime::ProtoChanFromNet pch(ch);
    R.party = party;
    R.pfss_backend = &ctx->trunc_ctx->backend();
    R.pfss_chan = &pch;
    R.net_chan = &ch;
    R.pfss = &pe->pfss_batch();
    R.opens = &pe->open_collector();

    pe->begin_phase(runtime::PhaseExecutor::Phase::kLN2_MLP);
    pe->add_task(std::make_unique<runtime::CubicPolyTask>(
        bundle,
        std::span<const uint64_t>(hidden_scaled.data(), hidden_scaled.size()),
        std::span<uint64_t>(hidden_scaled.data(), hidden_scaled.size())));
    pe->run(R);
    pe->pfss_batch().clear();
    pe->open_collector().clear();
  } else {
    auto silu_spec = make_silu_spec({cfg.frac_bits, 16});
    for (size_t i = 0; i < hidden_scaled.size(); ++i) {
      int64_t v = to_signed(hidden_scaled[i]);
      hidden_scaled[i] = to_ring(ref_silu_fixed(silu_spec, v));
    }
  }

  matmul_publicW(view2(hidden_scaled.data(), B * T, H),
                 W2_public,
                 view2(Y_share.data, B * T, D),
                 mp);

  if (has_trunc) {
    std::mt19937_64 rng(1);
    auto plan2 = compiler::compile_matmul_truncation(ctx->trunc_ctx->backend(),
                                                     rng,
                                                     B * T,
                                                     H,
                                                     D,
                                                     cfg.frac_bits,
                                                     mat2_x_range,
                                                     mat2_w_range);
    std::vector<uint64_t> y_scaled;
    runtime::ProtoChanFromNet pch(ch);
    std::vector<uint64_t> y_vec(Y_share.data, Y_share.data + Y_share.numel());
    runtime::run_truncation_now(party,
                                ctx->trunc_ctx->backend(),
                                pch,
                                plan2.bundle,
                                y_vec,
                                y_scaled);
    for (size_t i = 0; i < Y_share.numel() && i < y_scaled.size(); ++i) {
      Y_share.data[i] = y_scaled[i];
    }
  } else {
    if (!mp.local_rescale) rescale_view(Y_share, cfg.frac_bits);
  }
}

}  // namespace nn
