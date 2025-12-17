#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "gates/piecewise_poly.hpp"
#include "suf/suf_ir.hpp"
#include "suf/suf_silu_builders.hpp"

namespace suf {

inline uint64_t mask_n_bits(uint64_t v, int n_bits) {
  if (n_bits <= 0) return 0;
  if (n_bits >= 64) return v;
  return v & ((uint64_t(1) << n_bits) - 1);
}

// Build a SUF for GeLU from a piecewise spline table. Like SiLU, the SUF emits
// per-interval polynomial coefficients (c0..c3) as additive payload words.
inline SUF<uint64_t> build_gelu_suf_from_piecewise(const gates::PiecewisePolySpec& spec) {
  SUF<uint64_t> F;
  F.n_bits = 64;
  F.r_out = 4;  // emit c0..c3 (cubic)
  F.l_out = 0;
  F.degree = 0;  // payload-only; Horner handled by CubicPolyTask postproc

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
    auto coeffs = expand_to_x_coeffs(pack, 3, spec.frac_bits_out);
    coeffs.resize(static_cast<size_t>(F.r_out), 0);
    for (int i = 0; i < F.r_out; ++i) {
      suf::Poly<uint64_t> poly;
      poly.coeffs = {coeffs[static_cast<size_t>(i)]};
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

    if (start < cur) start = cur;
    if (end <= start) continue;

    if (start > cur) {
      F.pieces.push_back(make_zero_piece());
      F.alpha.push_back(start);
      cur = start;
    }

    F.pieces.push_back(make_coeff_piece(iv.pack));
    F.alpha.push_back(end);
    cur = end;
  }

  if (F.pieces.empty()) {
    F.alpha.clear();
    F.alpha.push_back(0ull);
    F.alpha.push_back(~uint64_t(0));
    F.pieces.push_back(make_zero_piece());
  }

  return F;
}

// Benchmark-oriented GeLU approximation: emit the evaluated GeLU value per interval as a single
// arithmetic payload word (degree=0). This avoids online Beaver mul/trunc for GeLU and is intended
// for end-to-end throughput comparisons, not accuracy studies.
inline SUF<uint64_t> build_gelu_suf_const_from_piecewise(const gates::PiecewisePolySpec& spec,
                                                         int eff_bits) {
  SUF<uint64_t> F;
  // The spline tables are defined over the full two's-complement u64 domain; keep n_bits=64
  // so the interval partition matches the table, but mask payload outputs down to `eff_bits`.
  F.n_bits = 64;
  F.r_out = 1;
  F.l_out = 0;
  F.degree = 0;  // payload-only constant per interval.

  std::vector<gates::PiecewiseInterval> intervals = spec.intervals;
  std::sort(intervals.begin(), intervals.end(),
            [](const gates::PiecewiseInterval& a, const gates::PiecewiseInterval& b) {
              return a.start < b.start;
            });
  if (intervals.empty()) return F;

  auto make_const_piece = [&](uint64_t start, uint64_t end) {
    // Pick a representative x in [start,end) and evaluate the reference spline there, then
    // use it as a constant approximation over the interval.
    uint64_t width = end - start;
    uint64_t mid_u = start + (width >> 1);
    int64_t x_mid = static_cast<int64_t>(mid_u);
    int64_t y_mid = gates::eval_piecewise_poly_ref(spec, x_mid);
    suf::SufPiece<uint64_t> piece;
    suf::Poly<uint64_t> poly;
    poly.coeffs = {mask_n_bits(static_cast<uint64_t>(y_mid), eff_bits)};
    piece.polys.push_back(std::move(poly));
    return piece;
  };

  auto make_zero_piece = [&]() {
    suf::SufPiece<uint64_t> zero_piece;
    suf::Poly<uint64_t> zero_poly;
    zero_poly.coeffs.assign(1, 0);
    zero_piece.polys.push_back(std::move(zero_poly));
    return zero_piece;
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

    if (start < cur) start = cur;
    if (end <= start) continue;

    if (start > cur) {
      F.pieces.push_back(make_zero_piece());
      F.alpha.push_back(start);
      cur = start;
    }

    F.pieces.push_back(make_const_piece(start, end));
    F.alpha.push_back(end);
    cur = end;
  }

  if (F.pieces.empty()) {
    F.alpha.clear();
    F.alpha.push_back(0ull);
    F.alpha.push_back(~uint64_t(0));
    F.pieces.push_back(make_zero_piece());
  }

  return F;
}

}  // namespace suf
