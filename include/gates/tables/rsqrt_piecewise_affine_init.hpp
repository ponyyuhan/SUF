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

  const double scale = std::ldexp(1.0, frac_bits);
  auto scale_x = [&](double x) -> int64_t {
    return static_cast<int64_t>(std::llround(x * scale));
  };
  auto scale_y = [&](double y) -> int64_t {
    return static_cast<int64_t>(std::llround(y * scale));
  };

  auto add_interval = [&](uint64_t start, uint64_t end, const CoeffPack& pack) {
    if (start >= end) return;
    spec.intervals.push_back({start, end, pack});
  };

  // Clamp x < eps to eps. Force eps_qf >= 1 to keep boundaries increasing in low frac_bits cases.
  uint64_t eps_qf = static_cast<uint64_t>(scale_x(eps));
  if (eps_qf == 0) eps_qf = 1;
  double clamp_low_val = 1.0 / std::sqrt(std::max(eps, std::numeric_limits<double>::min()));
  CoeffPack below{0, {scale_y(clamp_low_val)}};
  add_interval(0, eps_qf, below);

  std::vector<double> bounds;
  bounds.push_back(eps);
  double cur = eps;
  while (cur < vmax) {
    cur *= 2.0;
    bounds.push_back(std::min(cur, vmax));
  }

  std::vector<uint64_t> bounds_q;
  bounds_q.reserve(bounds.size());
  bounds_q.push_back(eps_qf);
  for (size_t i = 1; i < bounds.size(); ++i) {
    uint64_t q = static_cast<uint64_t>(scale_x(bounds[i]));
    if (q <= bounds_q.back()) q = bounds_q.back() + 1;
    bounds_q.push_back(q);
  }

  auto qf_to_real = [&](uint64_t q) -> double {
    return static_cast<double>(static_cast<int64_t>(q)) / scale;
  };

  for (size_t i = 1; i < bounds.size(); ++i) {
    uint64_t Lq = bounds_q[i - 1];
    uint64_t Uq = bounds_q[i];
    if (Uq <= Lq) continue;
    double L = qf_to_real(Lq);
    double U = qf_to_real(Uq);
    double y0 = 1.0 / std::sqrt(L);
    double y1 = 1.0 / std::sqrt(U);
    double slope = (y1 - y0) / (U - L);
    CoeffPack pack;
    pack.offset = static_cast<int64_t>(Lq);
    pack.coeffs = {scale_y(y0), scale_y(slope)};
    add_interval(Lq, Uq, pack);
  }

  // x > vmax => clamp to vmax
  double tail = 1.0 / std::sqrt(vmax);
  CoeffPack above{0, {scale_y(tail)}};
  uint64_t vmax_q = bounds_q.empty() ? static_cast<uint64_t>(scale_x(vmax)) : bounds_q.back();
  if (vmax_q == 0) vmax_q = 1;
  add_interval(vmax_q, std::numeric_limits<uint64_t>::max(), above);

  return spec;
}

}  // namespace gates
