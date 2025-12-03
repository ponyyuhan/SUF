#pragma once

#include <memory>
#include <type_traits>
#include <variant>
#include <vector>
#include "suf/predicates.hpp"

namespace suf {

// AST for boolean formulas built from primitive preds and connectives.
struct BoolExpr;

struct BConst { bool v; };
struct BVar { int pred_idx; };  // index into primitive predicate vector
struct BNot { std::unique_ptr<BoolExpr> a; };
struct BXor { std::unique_ptr<BoolExpr> a, b; };
struct BAnd { std::unique_ptr<BoolExpr> a, b; };
struct BOr { std::unique_ptr<BoolExpr> a, b; };

struct BoolExpr {
  std::variant<BConst, BVar, BNot, BXor, BAnd, BOr> node;
  BoolExpr() = default;
  explicit BoolExpr(std::variant<BConst, BVar, BNot, BXor, BAnd, BOr> n) : node(std::move(n)) {}
  BoolExpr(const BoolExpr& other);
  BoolExpr& operator=(const BoolExpr& other);
  BoolExpr(BoolExpr&&) noexcept = default;
  BoolExpr& operator=(BoolExpr&&) noexcept = default;
};

inline BoolExpr clone_bool_expr(const BoolExpr& e);

inline std::variant<BConst, BVar, BNot, BXor, BAnd, BOr>
clone_variant(const std::variant<BConst, BVar, BNot, BXor, BAnd, BOr>& v) {
  return std::visit([](auto const& n) -> std::variant<BConst, BVar, BNot, BXor, BAnd, BOr> {
    using T = std::decay_t<decltype(n)>;
    if constexpr (std::is_same_v<T, BConst> || std::is_same_v<T, BVar>) {
      return n;
    } else if constexpr (std::is_same_v<T, BNot>) {
      return BNot{std::make_unique<BoolExpr>(clone_bool_expr(*n.a))};
    } else if constexpr (std::is_same_v<T, BXor>) {
      return BXor{std::make_unique<BoolExpr>(clone_bool_expr(*n.a)),
                  std::make_unique<BoolExpr>(clone_bool_expr(*n.b))};
    } else if constexpr (std::is_same_v<T, BAnd>) {
      return BAnd{std::make_unique<BoolExpr>(clone_bool_expr(*n.a)),
                  std::make_unique<BoolExpr>(clone_bool_expr(*n.b))};
    } else {  // BOr
      return BOr{std::make_unique<BoolExpr>(clone_bool_expr(*n.a)),
                 std::make_unique<BoolExpr>(clone_bool_expr(*n.b))};
    }
  }, v);
}

inline BoolExpr clone_bool_expr(const BoolExpr& e) { return BoolExpr{clone_variant(e.node)}; }

inline BoolExpr::BoolExpr(const BoolExpr& other) : node(clone_variant(other.node)) {}

inline BoolExpr& BoolExpr::operator=(const BoolExpr& other) {
  if (this != &other) node = clone_variant(other.node);
  return *this;
}

inline bool eval_bool_expr(const BoolExpr& e, const std::vector<bool>& prim_bits);

inline bool eval_bool_expr(const BConst& n, const std::vector<bool>&) { return n.v; }
inline bool eval_bool_expr(const BVar& n, const std::vector<bool>& prim_bits) {
  int idx = n.pred_idx;
  if (idx < 0 || static_cast<size_t>(idx) >= prim_bits.size()) return false;
  return prim_bits[static_cast<size_t>(idx)];
}
inline bool eval_bool_expr(const BNot& n, const std::vector<bool>& prim_bits) {
  return !eval_bool_expr(*n.a, prim_bits);
}
inline bool eval_bool_expr(const BXor& n, const std::vector<bool>& prim_bits) {
  return eval_bool_expr(*n.a, prim_bits) ^ eval_bool_expr(*n.b, prim_bits);
}
inline bool eval_bool_expr(const BAnd& n, const std::vector<bool>& prim_bits) {
  return eval_bool_expr(*n.a, prim_bits) && eval_bool_expr(*n.b, prim_bits);
}
inline bool eval_bool_expr(const BOr& n, const std::vector<bool>& prim_bits) {
  return eval_bool_expr(*n.a, prim_bits) || eval_bool_expr(*n.b, prim_bits);
}

inline bool eval_bool_expr(const BoolExpr& e, const std::vector<bool>& prim_bits) {
  return std::visit([&](auto&& n) { return eval_bool_expr(n, prim_bits); }, e.node);
}

}  // namespace suf
