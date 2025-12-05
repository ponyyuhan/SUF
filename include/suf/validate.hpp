#pragma once

#include <stdexcept>
#include <vector>
#include <cstdint>
#include <limits>
#include <type_traits>
#include "suf/suf_ir.hpp"
#include "suf/bool_expr.hpp"

namespace suf {

inline void ensure(bool ok, const char* msg) {
  if (!ok) throw std::runtime_error(msg);
}

inline void validate_pred(const PrimitivePred& p) {
  std::visit([](auto&& n) {
    using T = std::decay_t<decltype(n)>;
    if constexpr (std::is_same_v<T, Pred_X_mod2f_lt>) {
      ensure(n.f > 0 && n.f <= 64, "validate: f out of range");
      if (n.f < 64) {
        uint64_t mask = (uint64_t(1) << n.f) - 1;
        ensure(n.gamma <= mask, "validate: gamma too large");
      }
    } else if constexpr (std::is_same_v<T, Pred_X_lt_const>) {
      (void)n;
    } else if constexpr (std::is_same_v<T, Pred_MSB_x>) {
      (void)n;
    } else if constexpr (std::is_same_v<T, Pred_MSB_x_plus>) {
      (void)n;
    } else {
      ensure(false, "validate: unknown predicate");
    }
  }, p);
}

inline void validate_bool_expr(const BoolExpr& e, size_t num_preds) {
  std::visit([&](auto&& n) {
    using T = std::decay_t<decltype(n)>;
    if constexpr (std::is_same_v<T, BConst>) {
      (void)n;
    } else if constexpr (std::is_same_v<T, BVar>) {
      ensure(n.pred_idx >= 0 && static_cast<size_t>(n.pred_idx) < num_preds,
             "validate: BVar index out of range");
    } else if constexpr (std::is_same_v<T, BNot>) {
      ensure(n.a != nullptr, "validate: null ptr");
      validate_bool_expr(*n.a, num_preds);
    } else if constexpr (std::is_same_v<T, BXor> || std::is_same_v<T, BAnd> || std::is_same_v<T, BOr>) {
      ensure(n.a != nullptr && n.b != nullptr, "validate: null ptr");
      validate_bool_expr(*n.a, num_preds);
      validate_bool_expr(*n.b, num_preds);
    } else {
      ensure(false, "validate: unknown bool node");
    }
  }, e.node);
}

template<typename RingT>
inline void validate_poly(const Poly<RingT>& p, int max_deg) {
  ensure(!p.coeffs.empty(), "validate: empty polynomial");
  ensure(static_cast<int>(p.coeffs.size()) - 1 <= max_deg, "validate: degree too large");
}

// Validate new IR invariants; throws on failure.
template<typename RingT>
inline void validate_suf(const SUF<RingT>& s) {
  ensure(s.n_bits > 0 && s.n_bits <= 64, "validate: n_bits out of range");
  ensure(!s.alpha.empty(), "validate: alpha empty");
  ensure(s.alpha.size() >= 2, "validate: need at least two boundaries");
  ensure(s.pieces.size() + 1 == s.alpha.size(), "validate: piece/boundary mismatch");

  // boundaries strictly increasing and within domain
  for (size_t i = 1; i < s.alpha.size(); i++) {
    ensure(s.alpha[i - 1] < s.alpha[i], "validate: alpha not strictly increasing");
  }
  // last boundary <= 2^n
  if (s.n_bits < 64) {
    uint64_t maxv = (uint64_t(1) << s.n_bits);
    ensure(s.alpha.back() <= maxv, "validate: alpha exceeds domain");
  }

  for (const auto& p : s.primitive_preds) validate_pred(p);

  for (const auto& piece : s.pieces) {
    ensure(static_cast<int>(piece.polys.size()) == s.r_out, "validate: poly arity mismatch");
    ensure(static_cast<int>(piece.bool_outs.size()) == s.l_out, "validate: bool arity mismatch");
    for (const auto& poly : piece.polys) validate_poly(poly, s.degree);
    for (const auto& be : piece.bool_outs) validate_bool_expr(be, s.primitive_preds.size());
  }
}

}  // namespace suf
