#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include "gates/piecewise_poly.hpp"

namespace gates {

inline PiecewisePolySpec make_recip_affine_init_spec(
    int frac_bits = 16,
    double nmax = 1024.0) {
  PiecewisePolySpec spec;
  spec.frac_bits_in = frac_bits;
  spec.frac_bits_out = frac_bits;

  auto scale_x = [&](double x) -> int64_t {
    return static_cast<int64_t>(std::llround(x * std::ldexp(1.0, frac_bits)));
  };
  auto scale_y = [&](double y) -> int64_t {
    return static_cast<int64_t>(std::llround(y * std::ldexp(1.0, frac_bits)));
  };

  // x < 1 => clamp to 1. Callers ensure x is non-negative, so we only cover
  // [0,1) here (omit negative half to enable eff_bits packing).
  CoeffPack below;
  below.offset = 0;
  below.coeffs = {scale_y(1.0)};
  append_interval_signed(spec, /*start=*/0, scale_x(1.0), below);

  std::vector<double> bounds;
  bounds.push_back(1.0);
  double cur = 1.0;
  while (cur < nmax) {
    cur *= 2.0;
    bounds.push_back(std::min(cur, nmax));
  }

  for (size_t i = 1; i < bounds.size(); ++i) {
    double L = bounds[i - 1];
    double U = bounds[i];
    double y0 = 1.0 / L;
    double y1 = 1.0 / U;
    double slope = (y1 - y0) / (U - L);
    CoeffPack pack;
    pack.offset = scale_x(L);
    pack.coeffs = {scale_y(y0), scale_y(slope)};
    append_interval_signed(spec, scale_x(L), scale_x(U), pack);
  }

  // x >= nmax => clamp to 1/nmax
  double tail = 1.0 / nmax;
  CoeffPack above;
  above.offset = 0;
  above.coeffs = {scale_y(tail)};
  append_interval_signed(spec, scale_x(nmax), std::numeric_limits<int64_t>::max(), above);

  return spec;
}

}  // namespace gates
