#include "nn/mlp_block.hpp"

#include <vector>
#include "gates/tables/silu_spline_table.hpp"

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
                 LayerContext* ctx) {
  size_t B = X_share.shape[0];
  size_t T = X_share.shape[1];
  size_t D = cfg.D;
  size_t H = cfg.Hidden;
  MatmulParams mp;
  mp.frac_bits = cfg.frac_bits;
  mp.local_rescale = (ctx == nullptr);  // explicit rescale when ctx is provided.

  std::vector<uint64_t> hidden(B * T * H, 0);
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
  rescale_buffer(hidden, cfg.frac_bits);

  auto silu_spec = make_silu_spec({cfg.frac_bits, 16});
  for (size_t i = 0; i < hidden.size(); ++i) {
    int64_t v = to_signed(hidden[i]);
    hidden[i] = to_ring(ref_silu_fixed(silu_spec, v));
  }

  matmul_publicW(view2(hidden.data(), B * T, H),
                 W2_public,
                 view2(Y_share.data, B * T, D),
                 mp);
  rescale_view(Y_share, cfg.frac_bits);
}

}  // namespace nn
