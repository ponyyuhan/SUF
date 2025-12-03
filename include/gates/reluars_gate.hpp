#pragma once

#include "gates/relu_gate.hpp"

namespace gates {

// Placeholder: reuse ReLU SUF structure until ReluARS specifics are filled in.
inline suf::SUF<core::Z2n<64>> make_reluars_suf_placeholder() {
  return make_relu_suf_64();
}

}  // namespace gates
