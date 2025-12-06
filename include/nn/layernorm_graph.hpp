#pragma once

#include "nn/layer_context.hpp"

namespace nn {

// Build a LayerNorm subgraph: mean/var/rsqrt/affine (gamma/beta optional, treated as public).
// Returns output SecretTensor; gamma/beta can be nullptr for identity.
SecretTensor build_layernorm_graph(LayerContext& ctx,
                                   const SecretTensor& x,
                                   const TensorView<int64_t>* gamma_public,
                                   const TensorView<int64_t>* beta_public,
                                   int length,
                                   int frac_bits);

}  // namespace nn
