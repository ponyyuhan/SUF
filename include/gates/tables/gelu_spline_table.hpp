#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include "gates/piecewise_poly.hpp"

namespace gates {

// GeLU(x) = x * Phi(x), where Phi is the standard normal CDF.
inline double gelu_fn(double x) {
  constexpr double inv_sqrt2 = 0.70710678118654752440;  // 1/sqrt(2)
  return 0.5 * x * (1.0 + std::erf(x * inv_sqrt2));
}

inline double gelu_deriv(double x) {
  // d/dx [x*Phi(x)] = Phi(x) + x*phi(x)
  constexpr double inv_sqrt2 = 0.70710678118654752440;         // 1/sqrt(2)
  constexpr double inv_sqrt2pi = 0.39894228040143267794;       // 1/sqrt(2*pi)
  double phi = inv_sqrt2pi * std::exp(-0.5 * x * x);
  double Phi = 0.5 * (1.0 + std::erf(x * inv_sqrt2));
  return Phi + x * phi;
}

inline CoeffPack make_gelu_cubic_hermite_pack(double start,
                                              double end,
                                              int frac_bits_out,
                                              int64_t offset_fixed) {
  double h = end - start;
  double y0 = gelu_fn(start);
  double y1 = gelu_fn(end);
  double m0 = gelu_deriv(start);
  double m1 = gelu_deriv(end);

  // Cubic Hermite spline in t = x - start:
  // y = c0 + c1*t + c2*t^2 + c3*t^3
  double c0 = y0;
  double c1 = m0;
  double c2 = (3.0 * (y1 - y0) - (2.0 * m0 + m1) * h) / (h * h);
  double c3 = (2.0 * (y0 - y1) + (m0 + m1) * h) / (h * h * h);

  auto scale = [&](double v) -> int64_t {
    double s = std::ldexp(1.0, frac_bits_out);
    return static_cast<int64_t>(std::llround(v * s));
  };

  CoeffPack pack;
  pack.offset = offset_fixed;
  pack.coeffs = {scale(c0), scale(c1), scale(c2), scale(c3)};
  return pack;
}

inline PiecewisePolySpec make_gelu_spline_spec(int frac_bits = 16, int segments = 16) {
  PiecewisePolySpec spec;
  spec.frac_bits_in = frac_bits;
  spec.frac_bits_out = frac_bits;

  // Practical clamp range for fixed-point transformer activations.
  double lo = -8.0;
  double hi = 8.0;
  double step = (hi - lo) / static_cast<double>(segments);

  auto scale_x = [&](double x) -> int64_t {
    return static_cast<int64_t>(std::llround(x * std::ldexp(1.0, frac_bits)));
  };

  // x <= lo : clamp to 0
  CoeffPack below;
  below.offset = 0;
  below.coeffs = {0};
  append_interval_signed(spec, std::numeric_limits<int64_t>::min(), scale_x(lo), below);

  for (int i = 0; i < segments; ++i) {
    double start = lo + step * static_cast<double>(i);
    double end = (i == segments - 1) ? hi : (start + step);
    int64_t start_fixed = scale_x(start);
    int64_t end_fixed = scale_x(end);
    auto pack = make_gelu_cubic_hermite_pack(start, end, frac_bits, start_fixed);
    append_interval_signed(spec, start_fixed, end_fixed, pack);
  }

  // x >= hi : y = x (Phi≈1)
  CoeffPack above;
  above.offset = 0;
  above.coeffs = {0, static_cast<int64_t>(1ll << frac_bits)};
  append_interval_signed(spec, scale_x(hi), std::numeric_limits<int64_t>::max(), above);

  return spec;
}

}  // namespace gates

