#pragma once

#include "core/ring.hpp"
#include "suf/bool_expr.hpp"
#include "suf/suf_ir.hpp"

namespace gates {

// ReLU(x) = max(x,0) in two’s complement.
// Exposes arithmetic output y and helper bit w = 1[x>=0].
inline suf::SUF<uint64_t> make_relu_suf_u64() {
  suf::SUF<uint64_t> F;
  F.n_bits = 64;
  F.r_out = 1;
  F.l_out = 1;
  F.degree = 1;

  // boundaries split by sign: [0,2^63) and [2^63,2^64)
  F.alpha = {0ull, (1ull << 63), 0ull};  // last boundary wraps to represent 2^64

  // Primitive preds: MSB(x)
  F.primitive_preds.push_back(suf::Pred_MSB_x{});

  suf::SufPiece<uint64_t> nonneg;
  nonneg.polys = {suf::Poly<uint64_t>{{0ull, 1ull}}};  // 0 + 1*x
  nonneg.bool_outs = {suf::BoolExpr{suf::BConst{true}}};

  suf::SufPiece<uint64_t> neg;
  neg.polys = {suf::Poly<uint64_t>{{0ull}}};
  neg.bool_outs = {suf::BoolExpr{suf::BConst{false}}};

  F.pieces = {nonneg, neg};
  return F;
}

// ReLU(x) with only the arithmetic output (no helper bit). This is useful for
// internal max/reduction code paths that never consume the predicate output and
// would otherwise pay unnecessary Composite-FSS boolean blending overhead.
inline suf::SUF<uint64_t> make_relu_value_suf_u64() {
  auto F = make_relu_suf_u64();
  F.l_out = 0;
  for (auto& p : F.pieces) p.bool_outs.clear();
  return F;
}

}  // namespace gates
