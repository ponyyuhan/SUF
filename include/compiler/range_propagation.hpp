#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "compiler/range_analysis.hpp"

namespace compiler {

// Lightweight helpers to propagate conservative ranges through common NN ops.
// Intended to be used by a compiler pass that annotates IR values before
// selecting GateKind (GapARS vs faithful) and emitting truncation SUFs.

inline RangeInterval propagate_add(const RangeInterval& a, const RangeInterval& b) {
  return add_range(a, b);
}

inline RangeInterval propagate_sub(const RangeInterval& a, const RangeInterval& b) {
  return sub_range(a, b);
}

inline RangeInterval propagate_mul_const(const RangeInterval& x,
                                         int64_t c,
                                         int frac_bits) {
  return mul_const_range(x, c, frac_bits);
}

inline RangeInterval propagate_axpy(const RangeInterval& x,
                                    const RangeInterval& y,
                                    int64_t a,
                                    int frac_bits) {
  return axpy_range(x, y, a, frac_bits);
}

// Elementwise product of two fixed-point values; result carries full precision
// (no implicit shift) and the caller can decide to shift down based on scale.
inline RangeInterval propagate_mul(const RangeInterval& a,
                                   const RangeInterval& b,
                                   int frac_bits) {
  (void)frac_bits;
  return mul_range(a, b);
}

// Matmul accumulator range before truncation/rescale.
inline RangeInterval propagate_matmul_accum(const RangeInterval& x,
                                            const RangeInterval& w,
                                            size_t K) {
  return matmul_accum_range(x, w, K);
}

// Matmul accumulator range using a column L1 bound on weights (public case).
inline RangeInterval propagate_matmul_accum_rowl1(const RangeInterval& x,
                                                  int64_t row_l1_max) {
  int64_t max_abs_x = std::max<int64_t>(std::abs(x.lo), std::abs(x.hi));
  __int128 bound = static_cast<__int128>(max_abs_x) * static_cast<__int128>(row_l1_max);
  auto clamp = [](__int128 v) -> int64_t {
    if (v > static_cast<__int128>(std::numeric_limits<int64_t>::max())) {
      return std::numeric_limits<int64_t>::max();
    }
    if (v < static_cast<__int128>(std::numeric_limits<int64_t>::min())) {
      return std::numeric_limits<int64_t>::min();
    }
    return static_cast<int64_t>(v);
  };
  int64_t b = clamp(bound);
  RangeInterval r;
  r.is_signed = true;
  r.lo = -b;
  r.hi = b;
  return r;
}

// Matmul output range after arithmetic right shift by frac_bits (typical rescale).
inline RangeInterval propagate_matmul_out(const RangeInterval& x,
                                          const RangeInterval& w,
                                          size_t K,
                                          int frac_bits) {
  return matmul_output_range(x, w, K, frac_bits);
}

// Given an accumulator range and desired frac_bits, select GateKind for the rescale.
inline GateKind select_rescale_kind_from_accum(const RangeInterval& accum_range,
                                               int frac_bits) {
  return select_trunc_kind(accum_range, frac_bits);
}

}  // namespace compiler
