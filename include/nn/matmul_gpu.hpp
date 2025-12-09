#pragma once

#include <cstdint>
#include "nn/tensor_view.hpp"
#include "nn/matmul_publicW.hpp"

namespace nn {

// GPU-backed matmul against public weights. Returns true on success; on any
// CUDA error it returns false and leaves outputs unspecified.
bool matmul_publicW_gpu(const TensorView<uint64_t>& X_share,
                        const TensorView<int64_t>& W_public,
                        TensorView<uint64_t> Y_share,
                        const MatmulParams& params);

}  // namespace nn

