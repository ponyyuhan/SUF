#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include <iostream>

#include "gates/piecewise_poly.hpp"
#include "suf/suf_ir.hpp"

namespace suf {

inline uint64_t clamp_to_ring(__int128 v) {
  if (v > static_cast<__int128>(std::numeric_limits<int64_t>::max())) {
    v = static_cast<__int128>(std::numeric_limits<int64_t>::max());
  }
  if (v < static_cast<__int128>(std::numeric_limits<int64_t>::min())) {
    v = static_cast<__int128>(std::numeric_limits<int64_t>::min());
  }
  return static_cast<uint64_t>(static_cast<int64_t>(v));
}

inline __int128 ipow_signed(int64_t base, int exp) {
  __int128 acc = 1;
  __int128 b = static_cast<__int128>(base);
  for (int i = 0; i < exp; ++i) acc *= b;
  return acc;
}

// Expand a polynomial in (x - x0) into coefficients of x^k, scaling results
// back to Q{frac_bits} so downstream Horner can treat coeffs as Qf.
inline std::vector<uint64_t> expand_to_x_coeffs(const gates::CoeffPack& pack,
                                                int degree,
                                                int frac_bits) {
  double scale = std::ldexp(1.0, frac_bits);
  std::vector<double> coeffs_real(static_cast<size_t>(degree + 1), 0.0);
  for (size_t i = 0; i < coeffs_real.size(); ++i) {
    if (i < pack.coeffs.size()) {
      coeffs_real[i] = static_cast<double>(pack.coeffs[i]) / scale;
    }
  }
  double x0 = static_cast<double>(pack.offset) / scale;
  std::vector<double> accum(static_cast<size_t>(degree + 1), 0.0);
  auto binom = [](int k, int i) -> int {
    if (k == 0 || i == 0 || i == k) return 1;
    if (k == 1) return 1;
    if (k == 2) return (i == 1) ? 2 : 1;
    if (k == 3) return (i == 1 || i == 2) ? 3 : 1;
    int c = 1;
    for (int j = 1; j <= i; ++j) c = c * (k - j + 1) / j;
    return c;
  };
  for (int k = 0; k <= degree; ++k) {
    double ck = coeffs_real[static_cast<size_t>(k)];
    for (int i = 0; i <= k; ++i) {
      double term = ck * static_cast<double>(binom(k, i)) * std::pow(-x0, k - i);
      accum[static_cast<size_t>(i)] += term;
    }
  }
  std::vector<uint64_t> out(static_cast<size_t>(degree + 1), 0);
  for (size_t i = 0; i < out.size(); ++i) {
    long double v = static_cast<long double>(accum[i]) * static_cast<long double>(scale);
    if (v > static_cast<long double>(std::numeric_limits<int64_t>::max())) {
      v = static_cast<long double>(std::numeric_limits<int64_t>::max());
    }
    if (v < static_cast<long double>(std::numeric_limits<int64_t>::min())) {
      v = static_cast<long double>(std::numeric_limits<int64_t>::min());
    }
    out[i] = static_cast<uint64_t>(static_cast<int64_t>(std::llround(v)));
  }
  return out;
}

// Build a SUF for SiLU from an existing piecewise spline table. The resulting
// SUF emits per-interval polynomial coefficients (c0..c3) as additive shares,
// with degree=0 so runtime hooks can perform Horner + truncation explicitly.
inline SUF<uint64_t> build_silu_suf_from_piecewise(const gates::PiecewisePolySpec& spec) {
  SUF<uint64_t> F;
  F.n_bits = 64;
  // Emit 4 coeff words (c0..c3) as separate outputs.
  F.r_out = 4;
  F.l_out = 0;

  // Coeff payload only; Horner happens in postproc hook.
  F.degree = 0;

  std::vector<gates::PiecewiseInterval> intervals = spec.intervals;
  std::sort(intervals.begin(), intervals.end(),
            [](const gates::PiecewiseInterval& a, const gates::PiecewiseInterval& b) {
              return a.start < b.start;
            });
  if (intervals.empty()) return F;

  auto make_zero_piece = [&]() {
    suf::SufPiece<uint64_t> zero_piece;
    for (int i = 0; i < F.r_out; ++i) {
      suf::Poly<uint64_t> zero_poly;
      zero_poly.coeffs.assign(1, 0);
      zero_piece.polys.push_back(std::move(zero_poly));
    }
    return zero_piece;
  };

  auto make_coeff_piece = [&](const gates::CoeffPack& pack) {
    suf::SufPiece<uint64_t> piece;
    auto coeffs = expand_to_x_coeffs(pack, 3, spec.frac_bits_out);  // up to cubic
    coeffs.resize(static_cast<size_t>(F.r_out), 0);
    for (int i = 0; i < F.r_out; ++i) {
      suf::Poly<uint64_t> poly;
      poly.coeffs = {coeffs[static_cast<size_t>(i)]};  // degree 0 payload
      piece.polys.push_back(std::move(poly));
    }
    return piece;
  };

  F.alpha.clear();
  F.alpha.reserve(intervals.size() + 1);
  uint64_t cur = 0;
  bool seeded = false;

  for (const auto& iv : intervals) {
    uint64_t start = iv.start;
    uint64_t end = iv.end;
    if (end <= start) continue;

    if (!seeded) {
      // Ensure alpha begins at 0 for full-domain consistency.
      if (start > 0) {
        F.alpha.push_back(0ull);
        F.pieces.push_back(make_zero_piece());
        F.alpha.push_back(start);
      } else {
        F.alpha.push_back(start);
      }
      cur = start;
      seeded = true;
    }

    // Trim overlap; enforce monotonic starts.
    if (start < cur) start = cur;
    if (end <= start) continue;

    // Fill any gap with a zero piece to keep boundaries increasing.
    if (start > cur) {
      F.pieces.push_back(make_zero_piece());
      F.alpha.push_back(start);
      cur = start;
    }

    F.pieces.push_back(make_coeff_piece(iv.pack));
    F.alpha.push_back(end);
    cur = end;
  }

  // If nothing was emitted (all degenerate), fall back to a single zero interval.
  if (F.pieces.empty()) {
    F.alpha.clear();
    F.alpha.push_back(0ull);
    F.alpha.push_back(~uint64_t(0));
    F.pieces.push_back(make_zero_piece());
  }

  bool monotonic = true;
  for (size_t i = 1; i < F.alpha.size(); ++i) {
    if (F.alpha[i - 1] >= F.alpha[i]) {
      monotonic = false;
      break;
    }
  }
  if (!monotonic) {
    std::cerr << "build_silu_suf_from_piecewise produced non-monotonic alpha size="
              << F.alpha.size() << " frac_in=" << spec.frac_bits_in
              << " frac_out=" << spec.frac_bits_out << " intervals=" << intervals.size()
              << " vals=";
    size_t limit = std::min<size_t>(F.alpha.size(), 32);
    for (size_t i = 0; i < limit; ++i) {
      std::cerr << F.alpha[i];
      if (i + 1 < limit) std::cerr << ",";
    }
    std::cerr << "\n";
  }

  return F;
}

}  // namespace suf
