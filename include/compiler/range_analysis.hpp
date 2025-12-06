#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "compiler/pfss_program_desc.hpp"

namespace compiler {

inline RangeInterval intersect(const RangeInterval& a, const RangeInterval& b) {
  RangeInterval out;
  out.is_signed = a.is_signed || b.is_signed;
  out.lo = std::max(a.lo, b.lo);
  out.hi = std::min(a.hi, b.hi);
  if (out.lo > out.hi) {
    out.lo = out.hi = 0;
  }
  return out;
}

inline RangeInterval add_range(const RangeInterval& a, const RangeInterval& b) {
  RangeInterval out;
  out.is_signed = a.is_signed || b.is_signed;
  out.lo = a.lo + b.lo;
  out.hi = a.hi + b.hi;
  return out;
}

inline RangeInterval sub_range(const RangeInterval& a, const RangeInterval& b) {
  RangeInterval out;
  out.is_signed = a.is_signed || b.is_signed;
  out.lo = a.lo - b.hi;
  out.hi = a.hi - b.lo;
  return out;
}

inline RangeInterval mul_range(const RangeInterval& a, const RangeInterval& b) {
  RangeInterval out;
  out.is_signed = a.is_signed || b.is_signed;
  auto clamp = []( __int128 v) -> int64_t {
    if (v > static_cast<__int128>(std::numeric_limits<int64_t>::max())) {
      return std::numeric_limits<int64_t>::max();
    }
    if (v < static_cast<__int128>(std::numeric_limits<int64_t>::min())) {
      return std::numeric_limits<int64_t>::min();
    }
    return static_cast<int64_t>(v);
  };
  __int128 v1 = static_cast<__int128>(a.lo) * static_cast<__int128>(b.lo);
  __int128 v2 = static_cast<__int128>(a.lo) * static_cast<__int128>(b.hi);
  __int128 v3 = static_cast<__int128>(a.hi) * static_cast<__int128>(b.lo);
  __int128 v4 = static_cast<__int128>(a.hi) * static_cast<__int128>(b.hi);
  out.lo = clamp(std::min(std::min(v1, v2), std::min(v3, v4)));
  out.hi = clamp(std::max(std::max(v1, v2), std::max(v3, v4)));
  return out;
}

// Compute the number of effective bits needed to represent the interval (unsigned).
inline int effective_bits_unsigned(const RangeInterval& r) {
  uint64_t hi_u = static_cast<uint64_t>(r.hi);
  int bits = 0;
  while (hi_u) { hi_u >>= 1; bits++; }
  return std::max(bits, 1);
}

// Decide GateKind for a rescale based on a conservative range.
inline GateKind select_trunc_kind(const RangeInterval& r, int frac_bits) {
  (void)frac_bits;  // current GapARS certificate is frac-agnostic and uses absolute bounds.
  constexpr uint64_t kGapBound = (uint64_t(1) << 62);  // |x_int| < 2^(n-2) for n=64
  if (!r.is_signed) return GateKind::FaithfulTR;

  auto abs64 = [](int64_t v) -> uint64_t {
    return (v < 0) ? static_cast<uint64_t>(-v) : static_cast<uint64_t>(v);
  };
  uint64_t abs_lo = abs64(r.lo);
  uint64_t abs_hi = abs64(r.hi);
  uint64_t abs_max = std::max(abs_lo, abs_hi);
  if (abs_max < kGapBound) return GateKind::GapARS;
  return GateKind::FaithfulARS;
}

// Lightweight GapARS certificate helper.
inline bool has_gap_cert(const RangeInterval& r) {
  constexpr uint64_t kGapBound = (uint64_t(1) << 62);
  if (!r.is_signed) return false;
  auto abs64 = [](int64_t v) -> uint64_t { return (v < 0) ? static_cast<uint64_t>(-v) : static_cast<uint64_t>(v); };
  uint64_t abs_lo = abs64(r.lo);
  uint64_t abs_hi = abs64(r.hi);
  uint64_t abs_max = std::max(abs_lo, abs_hi);
  return abs_max < kGapBound;
}

// Repeat-add an interval `count` times (conservative clamp to int64_t limits).
inline RangeInterval repeat_add(const RangeInterval& a, size_t count) {
  RangeInterval out;
  out.is_signed = a.is_signed;
  if (count == 0) {
    out.lo = out.hi = 0;
    return out;
  }
  auto clamp = []( __int128 v) -> int64_t {
    if (v > static_cast<__int128>(std::numeric_limits<int64_t>::max())) {
      return std::numeric_limits<int64_t>::max();
    }
    if (v < static_cast<__int128>(std::numeric_limits<int64_t>::min())) {
      return std::numeric_limits<int64_t>::min();
    }
    return static_cast<int64_t>(v);
  };
  __int128 lo = static_cast<__int128>(a.lo) * static_cast<__int128>(count);
  __int128 hi = static_cast<__int128>(a.hi) * static_cast<__int128>(count);
  out.lo = clamp(std::min(lo, hi));
  out.hi = clamp(std::max(lo, hi));
  return out;
}

// Arithmetic right-shift of the bounds (conservative).
inline RangeInterval shift_down(const RangeInterval& r, int f) {
  if (f <= 0) return r;
  auto arith = [f](int64_t v) -> int64_t {
    return static_cast<int64_t>(v >> f);
  };
  RangeInterval out;
  out.is_signed = r.is_signed;
  int64_t a = arith(r.lo);
  int64_t b = arith(r.hi);
  out.lo = std::min(a, b);
  out.hi = std::max(a, b);
  return out;
}

// Conservative accumulator range for a length-K dot product before truncation.
inline RangeInterval matmul_accum_range(const RangeInterval& x_range,
                                        const RangeInterval& w_range,
                                        size_t K) {
  RangeInterval prod = mul_range(x_range, w_range);
  return repeat_add(prod, K);
}

// Output range after arithmetic right shift by frac_bits (typical rescale).
inline RangeInterval matmul_output_range(const RangeInterval& x_range,
                                         const RangeInterval& w_range,
                                         size_t K,
                                         int frac_bits) {
  auto acc = matmul_accum_range(x_range, w_range, K);
  return shift_down(acc, frac_bits);
}

// Simple helpers for linear ops used in NN graph to propagate ranges.
inline RangeInterval axpy_range(const RangeInterval& x_range,
                                const RangeInterval& y_range,
                                int64_t a,
                                int frac_bits) {
  RangeInterval prod = mul_range(y_range, RangeInterval{a, a, true});
  RangeInterval scaled = shift_down(prod, frac_bits);
  return add_range(x_range, scaled);
}

inline RangeInterval mul_const_range(const RangeInterval& x_range,
                                     int64_t c,
                                     int frac_bits) {
  RangeInterval prod = mul_range(x_range, RangeInterval{c, c, true});
  return shift_down(prod, frac_bits);
}

}  // namespace compiler
