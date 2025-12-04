#include "nn/mlp_block.hpp"

#include <vector>
#include "gates/tables/silu_spline_table.hpp"

namespace nn {

using gates::make_silu_spec;
using gates::ref_silu_fixed;

static inline int64_t to_signed(uint64_t v) { return static_cast<int64_t>(v); }
static inline uint64_t to_ring(int64_t v) { return static_cast<uint64_t>(v); }

void mlp_forward(const MLPConfig& cfg,
                 const TensorView<uint64_t>& X_share,
                 const TensorView<int64_t>& W1_public,
                 const TensorView<int64_t>& W2_public,
                 TensorView<uint64_t> Y_share) {
  size_t B = X_share.shape[0];
  size_t T = X_share.shape[1];
  size_t D = cfg.D;
  size_t H = cfg.Hidden;
  MatmulParams mp;
  mp.frac_bits = cfg.frac_bits;

  std::vector<uint64_t> hidden(B * T * H, 0);
  matmul_publicW(view2(const_cast<uint64_t*>(X_share.data), B * T, D),
                 W1_public,
                 view2(hidden.data(), B * T, H),
                 mp);

  auto silu_spec = make_silu_spec({cfg.frac_bits, 16});
  for (size_t i = 0; i < hidden.size(); ++i) {
    int64_t v = to_signed(hidden[i]);
    hidden[i] = to_ring(ref_silu_fixed(silu_spec, v));
  }

  matmul_publicW(view2(hidden.data(), B * T, H),
                 W2_public,
                 view2(Y_share.data, B * T, D),
                 mp);
}

}  // namespace nn
