#pragma once

#include <vector>
#include "suf/polynomial.hpp"
#include "suf/bool_expr.hpp"
#include "suf/predicates.hpp"

namespace suf {

// One interval piece: [alpha_i, alpha_{i+1})
template<typename RingT>
struct SufPiece {
  std::vector<Poly<RingT>> polys;   // r arithmetic outputs
  std::vector<BoolExpr> bool_outs;  // ℓ boolean outputs
};

template<typename RingT>
struct SUF {
  int n_bits;                    // n
  int r_out;                     // # arithmetic outputs
  int l_out;                     // # boolean outputs
  int degree;                    // max d
  std::vector<uint64_t> alpha;   // boundaries: size m+1, 0<=...<=2^n

  std::vector<PrimitivePred> primitive_preds;  // global list before masking
  std::vector<SufPiece<RingT>> pieces;         // size m
};

}  // namespace suf
