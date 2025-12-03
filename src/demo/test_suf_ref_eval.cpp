#include <cassert>
#include <iostream>
#include <random>
#include <limits>
#include "suf/suf_ir.hpp"
#include "suf/validate.hpp"
#include "suf/ref_eval.hpp"

using namespace suf;

static SUF<uint64_t> make_exhaustive_suf() {
  SUF<uint64_t> s;
  s.n_bits = 8;
  s.r_out = 1;
  s.l_out = 2;
  s.degree = 1;
  s.alpha = {0, 100, 200, 256};

  // predicates: p0 = x<128, p1 = x mod 16 < 7
  s.primitive_preds.push_back(Pred_X_lt_const{128});
  s.primitive_preds.push_back(Pred_X_mod2f_lt{4, 7});

  auto make_piece = [&](uint64_t c0) {
    SufPiece<uint64_t> p;
    Poly<uint64_t> poly;
    poly.coeffs = {c0, 1};  // y = c0 + x
    p.polys.push_back(poly);
    // bool0 = p0 AND !p1, bool1 = p1
    BoolExpr b0{BAnd{std::make_unique<BoolExpr>(BoolExpr{BVar{0}}),
                     std::make_unique<BoolExpr>(BoolExpr{BNot{std::make_unique<BoolExpr>(BoolExpr{BVar{1}})}})}};
    BoolExpr b1{BVar{1}};
    p.bool_outs.push_back(std::move(b0));
    p.bool_outs.push_back(std::move(b1));
    return p;
  };
  s.pieces.push_back(make_piece(1));
  s.pieces.push_back(make_piece(2));
  s.pieces.push_back(make_piece(3));
  return s;
}

static void test_exhaustive() {
  auto s = make_exhaustive_suf();
  validate_suf(s);
  for (uint64_t x = 0; x < 256; x++) {
    auto out = eval_suf_ref(s, x);
    int expected_piece = (x < 100) ? 0 : (x < 200 ? 1 : 2);
    assert(out.piece_idx == expected_piece);
    uint64_t expected_poly = (expected_piece + 1) + x;
    assert(out.arith.size() == 1 && out.arith[0] == expected_poly);
    bool p0 = x < 128;
    bool p1 = (x & 0xF) < 7;
    bool b0 = p0 && !p1;
    bool b1 = p1;
    assert(out.bools.size() == 2);
    assert(out.bools[0] == b0);
    assert(out.bools[1] == b1);
  }
  std::cout << "Exhaustive SUF ref-eval test passed (n=8)\n";
}

static void test_random() {
  SUF<uint64_t> s;
  s.n_bits = 64;
  s.r_out = 2;
  s.l_out = 1;
  s.degree = 2;
  s.alpha = {0, uint64_t(1) << 63, std::numeric_limits<uint64_t>::max()};
  s.primitive_preds.push_back(Pred_MSB_x{});
  Poly<uint64_t> p0; p0.coeffs = {1, 2, 3}; // 3x^2 +2x +1
  Poly<uint64_t> p1; p1.coeffs = {5, 0};    // 5
  SufPiece<uint64_t> piece0;
  piece0.polys = {p0, p1};
  piece0.bool_outs.push_back(BoolExpr{BVar{0}});
  SufPiece<uint64_t> piece1 = piece0;
  s.pieces = {piece0, piece1};
  validate_suf(s);

  std::mt19937_64 rng(123);
  for (int i = 0; i < 1000; i++) {
    uint64_t x = rng();
    auto out = eval_suf_ref(s, x);
    uint64_t poly0 = eval_poly_ref(p0, x);
    uint64_t poly1 = eval_poly_ref(p1, x);
    assert(out.arith[0] == poly0);
    assert(out.arith[1] == poly1);
    bool msb = ((x >> 63) & 1u) != 0;
    assert(out.bools[0] == msb);
  }
  std::cout << "Random SUF ref-eval test passed (n=64)\n";
}

int main() {
  test_exhaustive();
  test_random();
  return 0;
}
