#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include "gates/piecewise_poly.hpp"

namespace gates {

inline double nexp_fn(double t) { return std::exp(-t); }
inline double nexp_deriv(double t) { return -std::exp(-t); }

inline CoeffPack make_nexp_cubic(double start,
                                 double end,
                                 int frac_bits_out,
                                 int64_t offset_fixed) {
  double h = end - start;
  double y0 = nexp_fn(start);
  double y1 = nexp_fn(end);
  double m0 = nexp_deriv(start);
  double m1 = nexp_deriv(end);

  double c0 = y0;
  double c1 = m0;
  double c2 = (3.0 * (y1 - y0) - (2.0 * m0 + m1) * h) / (h * h);
  double c3 = (2.0 * (y0 - y1) + (m0 + m1) * h) / (h * h * h);

  double scale = std::ldexp(1.0, frac_bits_out);
  auto enc = [&](double v) -> int64_t {
    return static_cast<int64_t>(std::llround(v * scale));
  };

  CoeffPack pack;
  pack.offset = offset_fixed;
  pack.coeffs = {enc(c0), enc(c1), enc(c2), enc(c3)};
  return pack;
}

inline PiecewisePolySpec make_nexp_piecewise_spec(int frac_bits = 16, int segments = 16) {
  PiecewisePolySpec spec;
  spec.frac_bits_in = frac_bits;
  spec.frac_bits_out = frac_bits;

  double lo = 0.0;
  double hi = 16.0;
  double step = (hi - lo) / static_cast<double>(segments);

  auto scale_x = [&](double x) -> int64_t {
    return static_cast<int64_t>(std::llround(x * std::ldexp(1.0, frac_bits)));
  };

  // t < 0 -> clamp to 0 (exp(0)=1).
  // In all current callers, t is already clamped to [0, hi], so we omit the
  // negative-domain interval to enable eff_bits packing.

  for (int i = 0; i < segments; ++i) {
    double start = lo + step * static_cast<double>(i);
    double end = (i == segments - 1) ? hi : (start + step);
    int64_t start_fixed = scale_x(start);
    int64_t end_fixed = scale_x(end);
    auto pack = make_nexp_cubic(start, end, frac_bits, start_fixed);
    append_interval_signed(spec, start_fixed, end_fixed, pack);
  }

  // t > hi -> clamp input to hi
  double tail = nexp_fn(hi);
  CoeffPack above;
  above.offset = 0;
  above.coeffs = {static_cast<int64_t>(std::llround(tail * std::ldexp(1.0, frac_bits)))};
  append_interval_signed(spec, scale_x(hi), std::numeric_limits<int64_t>::max(), above);

  return spec;
}

}  // namespace gates
