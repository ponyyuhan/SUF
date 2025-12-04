#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include "gates/piecewise_poly.hpp"

namespace gates {

inline PiecewisePolySpec make_rsqrt_affine_init_spec(
    int frac_bits = 16,
    double eps = 1.0 / 1024.0,
    double vmax = 16.0) {
  PiecewisePolySpec spec;
  spec.frac_bits_in = frac_bits;
  spec.frac_bits_out = frac_bits;

  auto scale_x = [&](double x) -> int64_t {
    return static_cast<int64_t>(std::llround(x * std::ldexp(1.0, frac_bits)));
  };
  auto scale_y = [&](double y) -> int64_t {
    return static_cast<int64_t>(std::llround(y * std::ldexp(1.0, frac_bits)));
  };

  // x < eps => clamp to eps
  double clamp_low_val = 1.0 / std::sqrt(eps);
  CoeffPack below;
  below.offset = 0;
  below.coeffs = {scale_y(clamp_low_val)};
  append_interval_signed(spec, std::numeric_limits<int64_t>::min(), scale_x(eps), below);

  std::vector<double> bounds;
  bounds.push_back(eps);
  double cur = eps;
  while (cur < vmax) {
    cur *= 2.0;
    bounds.push_back(std::min(cur, vmax));
  }

  for (size_t i = 1; i < bounds.size(); ++i) {
    double L = bounds[i - 1];
    double U = bounds[i];
    double y0 = 1.0 / std::sqrt(L);
    double y1 = 1.0 / std::sqrt(U);
    double slope = (y1 - y0) / (U - L);
    CoeffPack pack;
    pack.offset = scale_x(L);
    pack.coeffs = {scale_y(y0), scale_y(slope)};
    append_interval_signed(spec, scale_x(L), scale_x(U), pack);
  }

  // x > vmax => clamp to vmax
  double tail = 1.0 / std::sqrt(vmax);
  CoeffPack above;
  above.offset = 0;
  above.coeffs = {scale_y(tail)};
  append_interval_signed(spec, scale_x(vmax), std::numeric_limits<int64_t>::max(), above);

  return spec;
}

}  // namespace gates
