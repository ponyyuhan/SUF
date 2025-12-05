#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

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

// Expand a polynomial in (x - x0) into coefficients of x^k so it can be
// embedded directly as a SUF arithmetic channel.
inline std::vector<uint64_t> expand_to_x_coeffs(const gates::CoeffPack& pack, int degree) {
  std::vector<__int128> accum(static_cast<size_t>(degree + 1), 0);
  std::vector<int64_t> coeffs = pack.coeffs;
  if (static_cast<int>(coeffs.size()) < degree + 1) {
    coeffs.resize(static_cast<size_t>(degree + 1), 0);
  }
  int64_t x0 = pack.offset;
  for (int k = 0; k <= degree; ++k) {
    int64_t ck = coeffs[static_cast<size_t>(k)];
    __int128 c128 = static_cast<__int128>(ck);
    for (int i = 0; i <= k; ++i) {
      // binom(k,i) * (-x0)^{k-i}
      int choose = 1;
      if (k == 1) choose = 1;
      else if (k == 2) choose = (i == 0 || i == 2) ? 1 : 2;
      else if (k == 3) {
        choose = (i == 0 || i == 3) ? 1 : (i == 1 || i == 2 ? 3 : 1);
      } else {
        // fallback: simple multiplicative formula (small degrees only in practice)
        choose = 1;
        for (int j = 1; j <= i; ++j) {
          choose = choose * (k - j + 1) / j;
        }
      }
      __int128 term = c128 * static_cast<__int128>(choose) * ipow_signed(-x0, k - i);
      accum[static_cast<size_t>(i)] += term;
    }
  }
  std::vector<uint64_t> out(static_cast<size_t>(degree + 1), 0);
  for (size_t i = 0; i < out.size(); ++i) {
    out[i] = clamp_to_ring(accum[i]);
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

  F.alpha.clear();
  F.alpha.reserve(intervals.size() + 1);
  F.alpha.push_back(intervals.front().start);

  // Ensure coverage starts at 0; if not, prepend a zero poly interval.
  if (F.alpha.front() != 0) {
    suf::SufPiece<uint64_t> zero_piece;
    for (int i = 0; i < F.r_out; ++i) {
      suf::Poly<uint64_t> zero_poly;
      zero_poly.coeffs.assign(1, 0);
      zero_piece.polys.push_back(std::move(zero_poly));
    }
    F.pieces.push_back(std::move(zero_piece));
    F.alpha.insert(F.alpha.begin(), 0ull);
  }

  for (const auto& iv : intervals) {
    suf::SufPiece<uint64_t> piece;
    auto coeffs = expand_to_x_coeffs(iv.pack, 3);  // up to cubic
    coeffs.resize(static_cast<size_t>(F.r_out), 0);
    for (int i = 0; i < F.r_out; ++i) {
      suf::Poly<uint64_t> poly;
      poly.coeffs = {coeffs[static_cast<size_t>(i)]};  // degree 0 payload
      piece.polys.push_back(std::move(poly));
    }
    F.pieces.push_back(std::move(piece));
    F.alpha.push_back(iv.end);
  }
  return F;
}

}  // namespace suf
