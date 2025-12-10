#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <optional>

#include "compiler/pfss_program_desc.hpp"

namespace compiler {

inline bool can_gapars(const GapCert& g, int nbits = 64) {
  if (g.kind != RangeKind::Proof || !g.is_signed) return false;
  __int128 lhs = static_cast<__int128>(g.max_abs) + static_cast<__int128>(g.mask_abs);
  __int128 rhs = static_cast<__int128>(uint64_t(1) << (nbits - 1));
  return lhs < rhs;
}

inline uint64_t default_mask_bound(int frac_bits) {
  if (frac_bits <= 0) return 1ull;
  if (frac_bits >= 63) return std::numeric_limits<uint64_t>::max();
  return uint64_t(1) << frac_bits;
}

inline std::optional<GapCert> gap_from_abs(const AbsBound& ab,
                                           int frac_bits,
                                           uint64_t mask_abs = 0) {
  if (ab.kind != RangeKind::Proof || !ab.is_signed) return std::nullopt;
  GapCert g;
  g.is_signed = ab.is_signed;
  g.frac_bits = frac_bits;
  g.max_abs = ab.max_abs;
  g.mask_abs = (mask_abs == 0) ? default_mask_bound(frac_bits) : mask_abs;
  g.kind = RangeKind::Proof;
  return g;
}

inline uint64_t sat_add_u64(uint64_t a, uint64_t b) {
  __int128 s = static_cast<__int128>(a) + static_cast<__int128>(b);
  if (s > static_cast<__int128>(std::numeric_limits<uint64_t>::max())) {
    return std::numeric_limits<uint64_t>::max();
  }
  return static_cast<uint64_t>(s);
}

inline uint64_t sat_mul_u64(uint64_t a, uint64_t b) {
  __int128 p = static_cast<__int128>(a) * static_cast<__int128>(b);
  if (p > static_cast<__int128>(std::numeric_limits<uint64_t>::max())) {
    return std::numeric_limits<uint64_t>::max();
  }
  return static_cast<uint64_t>(p);
}

inline uint64_t ceil_div_pow2(uint64_t v, int shift) {
  if (shift <= 0) return v;
  if (shift >= 63) return (v == 0) ? 0ull : 1ull;
  uint64_t mask = (uint64_t(1) << shift) - 1;
  return (v + mask) >> shift;
}

inline uint64_t shift_mask(uint64_t mask_abs, int shift) {
  if (shift == 0) return mask_abs;
  if (shift > 0) return ceil_div_pow2(mask_abs, shift);
  int s = -shift;
  if (s >= 63) return std::numeric_limits<uint64_t>::max();
  uint64_t mul = uint64_t(1) << s;
  return sat_mul_u64(mask_abs, mul);
}

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

inline AbsBound abs_from_range(const RangeInterval& r, bool is_signed) {
  auto abs64 = [](int64_t v) -> uint64_t { return (v < 0) ? static_cast<uint64_t>(-v) : static_cast<uint64_t>(v); };
  uint64_t abs_lo = abs64(r.lo);
  uint64_t abs_hi = abs64(r.hi);
  AbsBound out;
  out.is_signed = is_signed;
  out.max_abs = std::max(abs_lo, abs_hi);
  out.kind = RangeKind::Hint;
  return out;
}

inline AbsBound add_abs(const AbsBound& a, const AbsBound& b) {
  AbsBound out;
  out.is_signed = a.is_signed || b.is_signed;
  out.max_abs = sat_add_u64(a.max_abs, b.max_abs);
  out.kind = (a.kind == RangeKind::Proof && b.kind == RangeKind::Proof) ? RangeKind::Proof : RangeKind::Hint;
  return out;
}

inline AbsBound sub_abs(const AbsBound& a, const AbsBound& b) {
  return add_abs(a, b);
}

inline AbsBound shift_down_abs(const AbsBound& a, int shift) {
  AbsBound out = a;
  if (shift > 0) {
    out.max_abs = ceil_div_pow2(out.max_abs, shift);
  } else if (shift < 0) {
    int sh = -shift;
    if (sh >= 63) {
      out.max_abs = std::numeric_limits<uint64_t>::max();
    } else {
      out.max_abs = sat_mul_u64(out.max_abs, static_cast<uint64_t>(uint64_t(1) << sh));
    }
  }
  return out;
}

inline AbsBound mul_const_abs(const AbsBound& a, int64_t c, int frac_bits) {
  AbsBound out;
  out.is_signed = true;
  uint64_t c_abs = (c < 0) ? static_cast<uint64_t>(-c) : static_cast<uint64_t>(c);
  uint64_t prod = sat_mul_u64(a.max_abs, c_abs);
  out.max_abs = ceil_div_pow2(prod, frac_bits);
  out.kind = (a.kind == RangeKind::Proof) ? RangeKind::Proof : RangeKind::Hint;
  return out;
}

inline AbsBound hadamard_abs(const AbsBound& a, const AbsBound& b, int frac_bits) {
  AbsBound out;
  out.is_signed = a.is_signed || b.is_signed;
  uint64_t prod = sat_mul_u64(a.max_abs, b.max_abs);
  out.max_abs = ceil_div_pow2(prod, frac_bits);
  out.kind = (a.kind == RangeKind::Proof && b.kind == RangeKind::Proof) ? RangeKind::Proof : RangeKind::Hint;
  return out;
}

inline AbsBound axpy_abs(const AbsBound& x, const AbsBound& y, int64_t a, int frac_bits) {
  AbsBound scaled_y = mul_const_abs(y, a, frac_bits);
  return add_abs(x, scaled_y);
}

inline AbsBound matmul_accum_abs(const AbsBound& x, const AbsBound& w, size_t K) {
  AbsBound out;
  out.is_signed = true;
  uint64_t prod = sat_mul_u64(x.max_abs, w.max_abs);
  uint64_t total = sat_mul_u64(prod, static_cast<uint64_t>(K));
  out.max_abs = total;
  out.kind = (x.kind == RangeKind::Proof && w.kind == RangeKind::Proof) ? RangeKind::Proof : RangeKind::Hint;
  return out;
}

inline AbsBound matmul_rowl1_abs(const AbsBound& x, int64_t row_l1_max) {
  AbsBound out;
  out.is_signed = true;
  uint64_t l1 = (row_l1_max < 0) ? static_cast<uint64_t>(-row_l1_max)
                                 : static_cast<uint64_t>(row_l1_max);
  out.max_abs = sat_mul_u64(x.max_abs, l1);
  out.kind = (x.kind == RangeKind::Proof) ? RangeKind::Proof : RangeKind::Hint;
  return out;
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

inline RangeInterval clamp_range(const RangeInterval& in, int64_t lo, int64_t hi, bool is_signed = true) {
  RangeInterval r;
  r.is_signed = is_signed;
  r.lo = std::max(in.lo, lo);
  r.hi = std::min(in.hi, hi);
  if (r.lo > r.hi) {
    r.lo = r.hi = 0;
  }
  return r;
}

inline RangeInterval mul_const_range(const RangeInterval& x_range,
                                     int64_t c,
                                     int frac_bits) {
  RangeInterval prod = mul_range(x_range, RangeInterval{c, c, true});
  return shift_down(prod, frac_bits);
}

// Decide GateKind for a rescale based on proof-carrying bounds.
inline GateKind select_trunc_kind(const AbsBound& abs,
                                  int frac_bits,
                                  const std::optional<GapCert>& cert = std::nullopt,
                                  uint64_t mask_abs_hint = 0) {
  if (!abs.is_signed) return GateKind::FaithfulTR;
  GapCert g;
  g.is_signed = abs.is_signed;
  g.frac_bits = frac_bits;
  g.max_abs = abs.max_abs;
  g.mask_abs = (mask_abs_hint == 0) ? default_mask_bound(frac_bits) : mask_abs_hint;
  g.kind = abs.kind;
  if (cert) {
    g.is_signed = cert->is_signed;
    g.max_abs = cert->max_abs;
    g.mask_abs = (mask_abs_hint != 0) ? mask_abs_hint : cert->mask_abs;
    g.kind = cert->kind;
  }
  if (g.kind == RangeKind::Proof && can_gapars(g)) return GateKind::GapARS;
  return GateKind::FaithfulARS;
}

inline GateKind select_trunc_kind(const RangeInterval& r,
                                  int frac_bits,
                                  RangeKind kind = RangeKind::Hint,
                                  const std::optional<GapCert>& cert = std::nullopt,
                                  uint64_t mask_abs_hint = 0) {
  AbsBound abs = abs_from_range(r, r.is_signed);
  abs.kind = kind;
  return select_trunc_kind(abs, frac_bits, cert, mask_abs_hint);
}

}  // namespace compiler
