#pragma once

#include <cstdint>
#include "nn/attention_block.hpp"
#include "nn/mlp_block.hpp"
#include "nn/tensor_view.hpp"
#include "nn/layer_context.hpp"

namespace nn {

struct TransformerConfig {
  AttentionConfig attn;
  MLPConfig mlp;
  int frac_bits = 16;
};

void transformer_layer_forward(const TransformerConfig& cfg,
                               int party,
                               net::Chan& ch,
                               const TensorView<uint64_t>& X_share,
                               const TensorView<int64_t>& Wqkv_public,
                               const TensorView<int64_t>& Wout_public,
                               const TensorView<int64_t>& W1_public,
                               const TensorView<int64_t>& W2_public,
                               KVCache& cache,
                               TensorView<uint64_t> Y_share,
                               LayerContext* ctx = nullptr);

}  // namespace nn
