#pragma once

#include <cstdint>
#include <vector>
#include "nn/tensor_view.hpp"

namespace nn {

struct MatmulParams {
  int frac_bits = 0;  // stored frac bits of operands; output is unscaled if local_rescale=false.
  bool w_transposed = false;
  const std::vector<int64_t>* bias = nullptr;
  bool local_rescale = true;  // legacy shift; set false when explicit Rescale nodes are used.
  bool allow_legacy_shift = true;  // set false when LayerContext present; kept for debug path.
};

void matmul_publicW(const TensorView<uint64_t>& X_share,
                    const TensorView<int64_t>& W_public,
                    TensorView<uint64_t> Y_share,
                    const MatmulParams& params);

}  // namespace nn
