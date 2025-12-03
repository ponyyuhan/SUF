#pragma once

#include <vector>
#include <cstdint>
#include <type_traits>
#include "suf/suf_ir.hpp"
#include "suf/bool_expr.hpp"

namespace suf {

inline bool eval_primitive(const PrimitivePred& p, uint64_t x) {
  return std::visit([&](auto&& n) -> bool {
    using T = std::decay_t<decltype(n)>;
    if constexpr (std::is_same_v<T, Pred_X_lt_const>) {
      return x < n.beta;
    } else if constexpr (std::is_same_v<T, Pred_X_mod2f_lt>) {
      if (n.f >= 64) return x < n.gamma;
      uint64_t mask = (uint64_t(1) << n.f) - 1;
      return (x & mask) < n.gamma;
    } else if constexpr (std::is_same_v<T, Pred_MSB_x>) {
      return ((x >> 63) & 1u) != 0;
    } else if constexpr (std::is_same_v<T, Pred_MSB_x_plus>) {
      uint64_t y = x + n.c;
      return ((y >> 63) & 1u) != 0;
    } else {
      return false;
    }
  }, p);
}

template<typename T>
inline T eval_poly_ref(const Poly<T>& p, T x) {
  if (p.coeffs.empty()) return T{};
  T acc = p.coeffs.back();
  for (int i = static_cast<int>(p.coeffs.size()) - 2; i >= 0; --i) {
    acc = acc * x + p.coeffs[static_cast<size_t>(i)];
  }
  return acc;
}

struct RefEvalOut {
  std::vector<uint64_t> arith;
  std::vector<bool> bools;
  int piece_idx = -1;
};

inline int select_interval(const std::vector<uint64_t>& alpha, uint64_t x) {
  for (size_t i = 0; i + 1 < alpha.size(); i++) {
    uint64_t start = alpha[i];
    uint64_t end = alpha[i + 1];
    if (x >= start && x < end) {
      return static_cast<int>(i);
    }
  }
  return static_cast<int>(alpha.size()) - 2;
}

inline RefEvalOut eval_suf_ref(const SUF<uint64_t>& s, uint64_t x) {
  RefEvalOut out;
  int idx = select_interval(s.alpha, x);
  if (idx < 0 || static_cast<size_t>(idx) >= s.pieces.size()) throw std::runtime_error("ref_eval: bad interval");
  const auto& piece = s.pieces[static_cast<size_t>(idx)];

  std::vector<bool> prim_bits;
  prim_bits.reserve(s.primitive_preds.size());
  for (const auto& p : s.primitive_preds) prim_bits.push_back(eval_primitive(p, x));

  out.arith.resize(piece.polys.size());
  for (size_t i = 0; i < piece.polys.size(); i++) {
    out.arith[i] = eval_poly_ref(piece.polys[i], static_cast<uint64_t>(x));
  }
  out.bools.resize(piece.bool_outs.size());
  for (size_t i = 0; i < piece.bool_outs.size(); i++) {
    out.bools[i] = eval_bool_expr(piece.bool_outs[i], prim_bits);
  }
  out.piece_idx = idx;
  return out;
}

}  // namespace suf
