#pragma once

#include <cstdint>
#include <vector>
#include "nn/tensor_view.hpp"

namespace nn {

struct MatmulParams {
  int frac_bits = 0;
  bool w_transposed = false;
  const std::vector<int64_t>* bias = nullptr;
};

void matmul_publicW(const TensorView<uint64_t>& X_share,
                    const TensorView<int64_t>& W_public,
                    TensorView<uint64_t> Y_share,
                    const MatmulParams& params);

}  // namespace nn
