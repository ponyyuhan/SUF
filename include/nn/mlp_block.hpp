#pragma once

#include <cstdint>
#include "nn/matmul_publicW.hpp"
#include "nn/tensor_view.hpp"
#include "gates/silu_spline_gate.hpp"
#include "nn/layer_context.hpp"
#include "runtime/phase_executor.hpp"
#include "mpc/net.hpp"

namespace nn {

struct MLPConfig {
  size_t D = 0;
  size_t Hidden = 0;
  int frac_bits = 16;
};

void mlp_forward(const MLPConfig& cfg,
                 const TensorView<uint64_t>& X_share,
                 const TensorView<int64_t>& W1_public,
                 const TensorView<int64_t>& W2_public,
                 TensorView<uint64_t> Y_share,
                 int party,
                 net::Chan& ch,
                 LayerContext* ctx = nullptr,
                 runtime::PhaseExecutor* pe = nullptr);

}  // namespace nn
