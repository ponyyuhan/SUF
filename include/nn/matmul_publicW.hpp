#pragma once

#include <cstdint>
#include <vector>
#include "nn/tensor_view.hpp"

namespace nn {

struct MatmulParams {
  int frac_bits = 0;  // stored frac bits of operands; output is unscaled if local_rescale=false.
  bool w_transposed = false;
  const std::vector<int64_t>* bias = nullptr;
  bool local_rescale = false;  // legacy shift; default off (explicit Rescale should be used).
  bool allow_legacy_shift = false;  // must be opt-in for legacy/debug paths.
  // Optional: GPU overlap stream to chain with PFSS kernels when using a GPU backend.
  void* overlap_stream = nullptr;
};

// Optional: obtain the default CUDA stream used by the GPU matmul
// implementation (non-blocking). Returns nullptr when CUDA is not enabled.
void* matmul_default_stream();

void matmul_publicW(const TensorView<uint64_t>& X_share,
                    const TensorView<int64_t>& W_public,
                    TensorView<uint64_t> Y_share,
                    const MatmulParams& params);

}  // namespace nn
