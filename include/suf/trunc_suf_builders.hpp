#pragma once

#include <cstdint>
#include <limits>
#include <vector>

#include "suf/bool_expr.hpp"
#include "suf/predicates.hpp"
#include "suf/suf_ir.hpp"

namespace suf {

// Build a predicate-only new for truncation carry (and optional sign).
// - carry = 1[((x_low + r_low) mod 2^f) < r_low] = 1[x_low >= 2^f - r_low]
// - sign  = MSB(x) when requested
inline SUF<uint64_t> build_trunc_pred_suf(int frac_bits,
                                          uint64_t r_low,
                                          bool include_sign) {
  SUF<uint64_t> s;
  s.n_bits = 64;
  s.r_out = 1;   // placeholder arithmetic output (mask-only, to be filled by postproc)
  s.degree = 0;  // constant polynomial

  // Single piece covering the domain [0, 2^64).
  s.alpha = {0ull, std::numeric_limits<uint64_t>::max()};

  SufPiece<uint64_t> piece;
  Poly<uint64_t> poly;
  poly.coeffs = {0ull};  // constant 0; output mask is carried in r_out
  piece.polys.push_back(poly);

  // Carry predicate (optional when frac_bits == 0).
  if (frac_bits > 0) {
    uint64_t mask = (frac_bits >= 64) ? std::numeric_limits<uint64_t>::max()
                                      : ((uint64_t(1) << frac_bits) - 1);
    uint64_t two_f = (frac_bits >= 64) ? 0ull : (uint64_t(1) << frac_bits);
    uint64_t gamma = r_low & mask;
    // carry = NOT(x_mod < threshold) with threshold = 2^f - r_low; when r_low==0 carry is always 0.
    if (gamma == 0 || two_f == 0) {
      piece.bool_outs.push_back(BoolExpr{BConst{false}});
    } else {
      uint64_t threshold = two_f - gamma;
      int idx = static_cast<int>(s.primitive_preds.size());
      s.primitive_preds.push_back(Pred_X_mod2f_lt{frac_bits, threshold});
      BoolExpr base{BVar{idx}};
      piece.bool_outs.push_back(BoolExpr{BNot{std::make_unique<BoolExpr>(base)}});
    }
  }

  // Sign bit predicate if requested.
  if (include_sign) {
    int idx = static_cast<int>(s.primitive_preds.size());
    s.primitive_preds.push_back(Pred_MSB_x{});
    piece.bool_outs.push_back(BoolExpr{BVar{idx}});
  }

  s.l_out = static_cast<int>(piece.bool_outs.size());
  s.pieces.push_back(std::move(piece));
  return s;
}

inline SUF<uint64_t> build_trunc_faithful_suf(int frac_bits, uint64_t r_low) {
  return build_trunc_pred_suf(frac_bits, r_low, /*include_sign=*/false);
}

inline SUF<uint64_t> build_ars_faithful_suf(int frac_bits, uint64_t r_low) {
  return build_trunc_pred_suf(frac_bits, r_low, /*include_sign=*/true);
}

inline SUF<uint64_t> build_gapars_suf(int frac_bits, uint64_t r_low) {
  // GapARS uses the same predicate set (carry + sign) for now.
  return build_trunc_pred_suf(frac_bits, r_low, /*include_sign=*/true);
}

}  // namespace suf
