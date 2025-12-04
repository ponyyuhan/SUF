#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include "nn/kv_cache.hpp"
#include "nn/matmul_publicW.hpp"
#include "nn/tensor_view.hpp"
#include "mpc/net.hpp"

namespace nn {

struct AttentionConfig {
  size_t D = 0;
  size_t H = 0;
  size_t Dh = 0;
  size_t S_max = 0;
  int frac_bits = 16;
};

void attention_forward(const AttentionConfig& cfg,
                       int party,
                       net::Chan& ch,
                       const TensorView<uint64_t>& X_share,
                       const TensorView<int64_t>& Wqkv_public,
                       const TensorView<int64_t>& Wout_public,
                       KVCache& cache,
                       TensorView<uint64_t> Y_share);

}  // namespace nn
