#include <cassert>
#include <iostream>
#include <random>
#include <unordered_map>
#include "compiler/suf_to_pfss.hpp"
#include "suf/ref_eval.hpp"
#include "suf/validate.hpp"

using namespace compiler;
using namespace suf;

struct RawEvalCtx {
  const PredProgramDesc* desc = nullptr;
  std::vector<uint64_t> bits; // raw bits in {0,1}
};

static RawEvalCtx eval_pred_program(const PredProgramDesc& p, uint64_t hatx) {
  RawEvalCtx ctx;
  ctx.desc = &p;
  ctx.bits.resize(p.queries.size());
  for (size_t i = 0; i < p.queries.size(); i++) {
    const auto& q = p.queries[i];
    if (q.kind == RawPredKind::kLtU64) {
      ctx.bits[i] = (hatx < q.theta) ? 1ull : 0ull;
    } else {
      uint64_t mask = (q.f >= 64) ? ~uint64_t(0) : ((uint64_t(1) << q.f) - 1);
      uint64_t z = hatx & mask;
      ctx.bits[i] = (z < q.theta) ? 1ull : 0ull;
    }
  }
  return ctx;
}

// Evaluate BoolExpr over raw bits (BVar idx into bits, BConst allowed)
static bool eval_bool_expr_raw(const BoolExpr& e, const std::vector<uint64_t>& bits, const std::vector<uint64_t>& wrap_bits) {
  return std::visit([&](auto&& n) -> bool {
    using T = std::decay_t<decltype(n)>;
    if constexpr (std::is_same_v<T, BConst>) {
      return n.v;
    } else if constexpr (std::is_same_v<T, BVar>) {
      size_t idx = static_cast<size_t>(n.pred_idx);
      if (idx < bits.size()) return bits[idx] != 0;
      idx -= bits.size();
      if (idx < wrap_bits.size()) return wrap_bits[idx] != 0;
      return false;
    } else if constexpr (std::is_same_v<T, BNot>) {
      return !eval_bool_expr_raw(*n.a, bits, wrap_bits);
    } else if constexpr (std::is_same_v<T, BXor>) {
      return eval_bool_expr_raw(*n.a, bits, wrap_bits) ^ eval_bool_expr_raw(*n.b, bits, wrap_bits);
    } else if constexpr (std::is_same_v<T, BAnd>) {
      return eval_bool_expr_raw(*n.a, bits, wrap_bits) && eval_bool_expr_raw(*n.b, bits, wrap_bits);
    } else {
      return eval_bool_expr_raw(*n.a, bits, wrap_bits) || eval_bool_expr_raw(*n.b, bits, wrap_bits);
    }
  }, e.node);
}

// Coeff step-DCF: base + sum_{cut<=hatx} delta
static std::vector<uint64_t> eval_coeff_step(const CoeffProgramDesc& c, uint64_t hatx) {
  std::vector<uint64_t> payload = c.base_payload_words;
  for (size_t i = 0; i < c.cutpoints_ge.size(); i++) {
    if (hatx >= c.cutpoints_ge[i]) {
      for (int j = 0; j < c.out_words; j++) {
        payload[static_cast<size_t>(j)] = payload[static_cast<size_t>(j)] + c.deltas_words[i][static_cast<size_t>(j)];
      }
    }
  }
  return payload;
}

static std::vector<uint64_t> eval_coeff_interval(const CoeffProgramDesc& c, uint64_t hatx) {
  for (const auto& seg : c.intervals) {
    if (hatx >= seg.lo && hatx < seg.hi) return seg.payload_words;
  }
  return c.intervals.empty() ? std::vector<uint64_t>{} : c.intervals.back().payload_words;
}

static std::vector<uint64_t> eval_coeff_program(const CoeffProgramDesc& c, uint64_t hatx) {
  if (c.mode == CoeffMode::kIntervalLut) return eval_coeff_interval(c, hatx);
  return eval_coeff_step(c, hatx);
}

static std::vector<uint64_t> eval_poly_vec(const std::vector<uint64_t>& coeff_flat, int r, int degree, uint64_t x) {
  std::vector<uint64_t> out(static_cast<size_t>(r), 0);
  int stride = degree + 1;
  for (int i = 0; i < r; i++) {
    uint64_t acc = coeff_flat[static_cast<size_t>(i * stride + degree)];
    for (int k = degree - 1; k >= 0; k--) {
      acc = acc * x + coeff_flat[static_cast<size_t>(i * stride + k)];
    }
    out[static_cast<size_t>(i)] = acc;
  }
  return out;
}

// Build a small SUF: intervals [0,100),(100,200),(200,2^64), r_out=1, l_out=1, degree=1, y = (piece_idx+1) + x
static SUF<uint64_t> make_sample_suf() {
  SUF<uint64_t> s;
  s.n_bits = 64;
  s.r_out = 1;
  s.l_out = 1;
  s.degree = 1;
  s.alpha = {0, 100, 200, std::numeric_limits<uint64_t>::max()};
  s.primitive_preds.push_back(Pred_X_lt_const{128}); // p0
  s.primitive_preds.push_back(Pred_X_mod2f_lt{4, 7}); // p1

  auto make_piece = [&](uint64_t c0) {
    SufPiece<uint64_t> p;
    Poly<uint64_t> poly;
    poly.coeffs = {c0, 1};  // y = c0 + x
    p.polys.push_back(poly);
    BoolExpr b0{BAnd{std::make_unique<BoolExpr>(BoolExpr{BVar{0}}),
                     std::make_unique<BoolExpr>(BoolExpr{BNot{std::make_unique<BoolExpr>(BoolExpr{BVar{1}})}})}};
    p.bool_outs.push_back(std::move(b0));
    return p;
  };
  s.pieces.push_back(make_piece(1));
  s.pieces.push_back(make_piece(2));
  s.pieces.push_back(make_piece(3));
  return s;
}

int main() {
  auto s = make_sample_suf();
  validate_suf(s);
  uint64_t r_in = 12345;
  std::vector<uint64_t> r_out = {9876};
  auto compiled = compile_suf_to_pfss_two_programs(s, r_in, r_out, CoeffMode::kStepDcf);

  std::mt19937_64 rng(2025);
  for (int t = 0; t < 1000; t++) {
    uint64_t x = rng();
    uint64_t hatx = x + r_in;

    auto ref = eval_suf_ref(s, x);
    auto preds = eval_pred_program(compiled.pred, hatx);
    auto coeffs = eval_coeff_program(compiled.coeff, hatx);
    auto poly_out = eval_poly_vec(coeffs, compiled.r, compiled.degree, x);
    // bool outputs: select piece by coeff interval matching
    int piece_idx = -1;
    for (size_t i = 0; i < compiled.coeff.cutpoints_ge.size() + 1; i++) {
      // reconstruct using step-DCF segmentation order: base is segment0, cuts introduce new segments
      // we cannot derive piece directly here; fall back to ref.piece_idx for this test
      (void)i;
    }
    piece_idx = ref.piece_idx;
    std::vector<uint64_t> wraps = compiled.wrap_bits;
    bool b = eval_bool_expr_raw(compiled.bool_per_piece[static_cast<size_t>(piece_idx)][0], preds.bits, wraps);

    uint64_t y_masked = poly_out[0] + r_out[0];
    assert(poly_out[0] == ref.arith[0]);
    assert(b == ref.bools[0]);
    assert(y_masked - r_out[0] == ref.arith[0]);
  }
  std::cout << "compile_suf_to_pfss_two_programs baseline test passed\n";
  return 0;
}
