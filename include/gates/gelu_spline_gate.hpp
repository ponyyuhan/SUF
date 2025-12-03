#pragma once

#include "core/ring.hpp"
#include "suf/suf_ir.hpp"

namespace gates {

// Placeholder GeLU spline SUF; fill with concrete spline coefficients as needed.
inline suf::SUF<core::Z2n<64>> make_gelu_spline_placeholder() {
  suf::SUF<core::Z2n<64>> F;
  F.n_bits = 64;
  F.r_out = 1;
  F.l_out = 0;
  F.degree = 1;
  F.alpha = {0ull, 0ull};  // degenerate until configured
  return F;
}

}  // namespace gates
