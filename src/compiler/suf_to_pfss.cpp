#include "compiler/suf_to_pfss.hpp"

#include <algorithm>
#include <numeric>
#include <tuple>
#include <unordered_map>
#include <iostream>
#include "suf/mask_rewrite.hpp"
#include "suf/mask_rewrite_eval.hpp"
#include "suf/validate.hpp"

namespace compiler {

struct QueryKey {
  RawPredKind kind;
  uint8_t f;
  uint64_t theta;
  bool operator==(const QueryKey& o) const {
    return kind == o.kind && f == o.f && theta == o.theta;
  }
};
struct QueryKeyHash {
  size_t operator()(const QueryKey& k) const noexcept {
    return std::hash<uint64_t>{}((uint64_t)k.kind ^ (uint64_t(k.f) << 8) ^ k.theta);
  }
};

static int add_query(const RawPredQuery& q, std::vector<RawPredQuery>& out, std::unordered_map<QueryKey,int,QueryKeyHash>& mp) {
  QueryKey key{q.kind, q.f, q.theta};
  auto it = mp.find(key);
  if (it != mp.end()) return it->second;
  int idx = static_cast<int>(out.size());
  out.push_back(q);
  mp[key] = idx;
  return idx;
}

static suf::BoolExpr make_wrap_expr(uint64_t wrap_idx, int idx_raw, int idx_raw2) {
  // w = wrap ? ((!a)|b) : ((!a)&b)
  suf::BoolExpr a{ suf::BVar{idx_raw} };
  suf::BoolExpr b{ suf::BVar{idx_raw2} };
  suf::BoolExpr na{ suf::BNot{ std::make_unique<suf::BoolExpr>(a) } };
  suf::BoolExpr and_expr{ suf::BAnd{ std::make_unique<suf::BoolExpr>(na), std::make_unique<suf::BoolExpr>(b) } };
  suf::BoolExpr or_expr{ suf::BOr{ std::make_unique<suf::BoolExpr>(na), std::make_unique<suf::BoolExpr>(b) } };
  // Use negative indices to refer to wrap bits; evaluator interprets them with offset.
  suf::BoolExpr wrap_var{ suf::BVar{ -1 - static_cast<int>(wrap_idx) } };
  suf::BoolExpr not_wrap{ suf::BNot{ std::make_unique<suf::BoolExpr>(wrap_var) } };
  suf::BoolExpr term0{ suf::BAnd{ std::make_unique<suf::BoolExpr>(not_wrap), std::make_unique<suf::BoolExpr>(and_expr) } };
  suf::BoolExpr term1{ suf::BAnd{ std::make_unique<suf::BoolExpr>(wrap_var), std::make_unique<suf::BoolExpr>(or_expr) } };
  suf::BoolExpr res{ suf::BOr{ std::make_unique<suf::BoolExpr>(term0), std::make_unique<suf::BoolExpr>(term1) } };
  return res;
}

static suf::BoolExpr rewrite_pred(const suf::PrimitivePred& p,
                                  uint64_t r_in,
                                  std::vector<RawPredQuery>& queries,
                                  std::unordered_map<QueryKey,int,QueryKeyHash>& mp,
                                  std::vector<uint64_t>& wrap_bits) {
  return std::visit([&](auto&& n) -> suf::BoolExpr {
    using T = std::decay_t<decltype(n)>;
    if constexpr (std::is_same_v<T, suf::Pred_X_lt_const>) {
      auto rec = suf::rewrite_lt_u64(r_in, n.beta);
      int a = add_query(RawPredQuery{RawPredKind::kLtU64, 64, rec.theta0}, queries, mp);
      int b = add_query(RawPredQuery{RawPredKind::kLtU64, 64, rec.theta1}, queries, mp);
      size_t wrap_idx = wrap_bits.size();
      wrap_bits.push_back(rec.wrap);
      return make_wrap_expr(wrap_idx, a, b);
    } else if constexpr (std::is_same_v<T, suf::Pred_X_mod2f_lt>) {
      auto rec = suf::rewrite_ltlow(r_in, n.f, n.gamma);
      int a = add_query(RawPredQuery{RawPredKind::kLtLow, static_cast<uint8_t>(rec.f), rec.theta0}, queries, mp);
      int b = add_query(RawPredQuery{RawPredKind::kLtLow, static_cast<uint8_t>(rec.f), rec.theta1}, queries, mp);
      size_t wrap_idx = wrap_bits.size();
      wrap_bits.push_back(rec.wrap);
      return make_wrap_expr(wrap_idx, a, b);
    } else if constexpr (std::is_same_v<T, suf::Pred_MSB_x>) {
      auto rec = suf::rewrite_msb_add(r_in, 0);
      int a = add_query(RawPredQuery{RawPredKind::kLtU64, 64, rec.theta0}, queries, mp);
      int b = add_query(RawPredQuery{RawPredKind::kLtU64, 64, rec.theta1}, queries, mp);
      size_t wrap_idx = wrap_bits.size();
      wrap_bits.push_back(rec.wrap);
      return make_wrap_expr(wrap_idx, a, b);
    } else { // MSB(x+c)
      auto rec = suf::rewrite_msb_add(r_in, n.c);
      int a = add_query(RawPredQuery{RawPredKind::kLtU64, 64, rec.theta0}, queries, mp);
      int b = add_query(RawPredQuery{RawPredKind::kLtU64, 64, rec.theta1}, queries, mp);
      size_t wrap_idx = wrap_bits.size();
      wrap_bits.push_back(rec.wrap);
      return make_wrap_expr(wrap_idx, a, b);
    }
  }, p);
}

static suf::BoolExpr rewrite_bool_expr(const suf::BoolExpr& e,
                                       const std::vector<suf::PrimitivePred>& preds,
                                       uint64_t r_in,
                                       std::vector<RawPredQuery>& queries,
                                       std::unordered_map<QueryKey,int,QueryKeyHash>& mp,
                                       std::vector<uint64_t>& wrap_bits) {
  return std::visit([&](auto&& n) -> suf::BoolExpr {
    using T = std::decay_t<decltype(n)>;
    if constexpr (std::is_same_v<T, suf::BConst>) return suf::BoolExpr{n};
    else if constexpr (std::is_same_v<T, suf::BVar>) {
      int idx = n.pred_idx;
      if (idx < 0 || static_cast<size_t>(idx) >= preds.size()) return suf::BoolExpr{suf::BConst{false}};
      return rewrite_pred(preds[static_cast<size_t>(idx)], r_in, queries, mp, wrap_bits);
    } else if constexpr (std::is_same_v<T, suf::BNot>) {
      return suf::BoolExpr{ suf::BNot{ std::make_unique<suf::BoolExpr>(rewrite_bool_expr(*n.a, preds, r_in, queries, mp, wrap_bits)) } };
    } else if constexpr (std::is_same_v<T, suf::BXor>) {
      return suf::BoolExpr{ suf::BXor{
        std::make_unique<suf::BoolExpr>(rewrite_bool_expr(*n.a, preds, r_in, queries, mp, wrap_bits)),
        std::make_unique<suf::BoolExpr>(rewrite_bool_expr(*n.b, preds, r_in, queries, mp, wrap_bits))} };
    } else if constexpr (std::is_same_v<T, suf::BAnd>) {
      return suf::BoolExpr{ suf::BAnd{
        std::make_unique<suf::BoolExpr>(rewrite_bool_expr(*n.a, preds, r_in, queries, mp, wrap_bits)),
        std::make_unique<suf::BoolExpr>(rewrite_bool_expr(*n.b, preds, r_in, queries, mp, wrap_bits))} };
    } else { // BOr
      return suf::BoolExpr{ suf::BOr{
        std::make_unique<suf::BoolExpr>(rewrite_bool_expr(*n.a, preds, r_in, queries, mp, wrap_bits)),
        std::make_unique<suf::BoolExpr>(rewrite_bool_expr(*n.b, preds, r_in, queries, mp, wrap_bits))} };
    }
  }, e.node);
}

static suf::BoolExpr remap_bool_expr_pred_indices(const suf::BoolExpr& e,
                                                  const std::vector<int>& new_of_old) {
  return std::visit([&](auto&& n) -> suf::BoolExpr {
    using T = std::decay_t<decltype(n)>;
    if constexpr (std::is_same_v<T, suf::BConst>) {
      return suf::BoolExpr{n};
    } else if constexpr (std::is_same_v<T, suf::BVar>) {
      int idx = n.pred_idx;
      if (idx >= 0 && static_cast<size_t>(idx) < new_of_old.size()) {
        return suf::BoolExpr{suf::BVar{new_of_old[static_cast<size_t>(idx)]}};
      }
      // Negative indices refer to wrap bits; leave untouched.
      return suf::BoolExpr{n};
    } else if constexpr (std::is_same_v<T, suf::BNot>) {
      return suf::BoolExpr{suf::BNot{
          std::make_unique<suf::BoolExpr>(remap_bool_expr_pred_indices(*n.a, new_of_old))}};
    } else if constexpr (std::is_same_v<T, suf::BXor>) {
      return suf::BoolExpr{suf::BXor{
          std::make_unique<suf::BoolExpr>(remap_bool_expr_pred_indices(*n.a, new_of_old)),
          std::make_unique<suf::BoolExpr>(remap_bool_expr_pred_indices(*n.b, new_of_old))}};
    } else if constexpr (std::is_same_v<T, suf::BAnd>) {
      return suf::BoolExpr{suf::BAnd{
          std::make_unique<suf::BoolExpr>(remap_bool_expr_pred_indices(*n.a, new_of_old)),
          std::make_unique<suf::BoolExpr>(remap_bool_expr_pred_indices(*n.b, new_of_old))}};
    } else {  // BOr
      return suf::BoolExpr{suf::BOr{
          std::make_unique<suf::BoolExpr>(remap_bool_expr_pred_indices(*n.a, new_of_old)),
          std::make_unique<suf::BoolExpr>(remap_bool_expr_pred_indices(*n.b, new_of_old))}};
    }
  }, e.node);
}

static std::vector<uint64_t> flatten_coeffs(const suf::SufPiece<uint64_t>& piece, int degree, int r) {
  std::vector<uint64_t> flat(static_cast<size_t>(r * (degree + 1)), 0);
  for (int i = 0; i < r; i++) {
    const auto& poly = piece.polys[static_cast<size_t>(i)];
    for (int k = 0; k <= degree && static_cast<size_t>(k) < poly.coeffs.size(); k++) {
      flat[static_cast<size_t>(i * (degree + 1) + k)] = poly.coeffs[static_cast<size_t>(k)];
    }
  }
  return flat;
}

static void build_coeff_step(const suf::SUF<uint64_t>& F, uint64_t r_in, CoeffProgramDesc& out) {
  // Rotate intervals, split wrap, sort by start.
  struct Seg { uint64_t start; std::vector<uint64_t> payload; };
  std::vector<Seg> segs;
  for (size_t i = 0; i + 1 < F.alpha.size(); i++) {
    uint64_t a = F.alpha[i];
    uint64_t b = F.alpha[i + 1];
    auto payload = flatten_coeffs(F.pieces[i], F.degree, F.r_out);
    uint64_t s = a + r_in;
    uint64_t e = b + r_in;
    if (s < e) {
      segs.push_back(Seg{s, payload});
    } else {
      segs.push_back(Seg{s, payload});
      segs.push_back(Seg{0, payload});
    }
  }
  std::sort(segs.begin(), segs.end(), [](const Seg& x, const Seg& y) { return x.start < y.start; });
  if (segs.empty()) return;
  out.base_payload_words = segs[0].payload;
  const std::vector<uint64_t>* prev = &segs[0].payload;
  for (size_t i = 1; i < segs.size(); i++) {
    std::vector<uint64_t> delta(static_cast<size_t>(out.out_words), 0);
    bool all_zero = true;
    for (int j = 0; j < out.out_words; j++) {
      uint64_t d = segs[i].payload[static_cast<size_t>(j)] - (*prev)[static_cast<size_t>(j)];
      delta[static_cast<size_t>(j)] = d;
      all_zero = all_zero && (d == 0);
    }
    // Drop redundant cutpoints when the payload does not change. This happens
    // for predicate-only SUFs (e.g., truncation carry/sign), where wrap-splitting
    // introduces a synthetic boundary but coefficients remain constant.
    if (all_zero) continue;
    out.cutpoints_ge.push_back(segs[i].start);
    out.deltas_words.push_back(std::move(delta));
    prev = &segs[i].payload;
  }
}

CompiledSUFGate compile_suf_to_pfss_two_programs(
    const suf::SUF<uint64_t>& F,
    uint64_t r_in,
    const std::vector<uint64_t>& r_out,
    CoeffMode coeff_mode,
    GateKind gate_kind) {
  if (gate_kind == GateKind::SiLUSpline) {
    bool monotonic = true;
    for (size_t i = 1; i < F.alpha.size(); ++i) {
      if (F.alpha[i - 1] >= F.alpha[i]) {
        monotonic = false;
        break;
      }
    }
    std::cerr << "compile_suf_to_pfss_two_programs SiLU ptr=" << static_cast<const void*>(&F)
              << " size=" << F.alpha.size() << " monotonic=" << (monotonic ? "yes" : "no")
              << " first=" << (F.alpha.empty() ? 0ull : F.alpha.front())
              << " last=" << (F.alpha.empty() ? 0ull : F.alpha.back()) << " vals=";
    size_t limit = std::min<size_t>(F.alpha.size(), 32);
    for (size_t i = 0; i < limit; ++i) {
      std::cerr << F.alpha[i];
      if (i + 1 < limit) std::cerr << ",";
    }
    std::cerr << "\n";
  }
  auto make_zero_piece = [&](int r, int l) {
    suf::SufPiece<uint64_t> p;
    p.polys.resize(r);
    for (auto& poly : p.polys) poly.coeffs = {0};
    p.bool_outs.resize(l);
    for (auto& b : p.bool_outs) b = suf::BoolExpr{suf::BConst{false}};
    return p;
  };
  auto repair_nonmonotonic = [&](const suf::SUF<uint64_t>& src) {
    suf::SUF<uint64_t> fixed = src;
    struct Seg { uint64_t start; uint64_t end; suf::SufPiece<uint64_t> piece; };
    std::vector<Seg> segs;
    for (size_t i = 0; i + 1 < src.alpha.size() && i < src.pieces.size(); ++i) {
      uint64_t s = src.alpha[i];
      uint64_t e = src.alpha[i + 1];
      if (e == s) continue;
      if (e > s) {
        segs.push_back(Seg{s, e, src.pieces[i]});
      } else {
        segs.push_back(Seg{s, ~uint64_t(0), src.pieces[i]});
        if (e > 0) segs.push_back(Seg{0ull, e, src.pieces[i]});
      }
    }
    std::sort(segs.begin(), segs.end(), [](const Seg& a, const Seg& b) { return a.start < b.start; });
    fixed.alpha.clear();
    fixed.pieces.clear();
    if (segs.empty()) {
      fixed.alpha = {0ull, ~uint64_t(0)};
      fixed.pieces.push_back(make_zero_piece(src.r_out, src.l_out));
    } else {
      fixed.alpha.push_back(segs.front().start);
      uint64_t cur = segs.front().start;
      for (const auto& seg : segs) {
        uint64_t start = seg.start;
        uint64_t end = seg.end;
        if (start < cur) start = cur;
        if (end <= start) continue;
        if (start > cur) {
          fixed.pieces.push_back(make_zero_piece(src.r_out, src.l_out));
          fixed.alpha.push_back(start);
          cur = start;
        }
        fixed.pieces.push_back(seg.piece);
        fixed.alpha.push_back(end);
        cur = end;
      }
    }
    return fixed;
  };

  suf::SUF<uint64_t> repaired;
  const suf::SUF<uint64_t>* F_ptr = &F;
  bool monotonic = true;
  for (size_t i = 1; i < F.alpha.size(); ++i) {
    if (F.alpha[i - 1] >= F.alpha[i]) {
      monotonic = false;
      break;
    }
  }
  if (!monotonic) {
    repaired = repair_nonmonotonic(F);
    F_ptr = &repaired;
  }
  const auto& Fn = *F_ptr;

  try {
    validate_suf(Fn);
  } catch (const std::exception& e) {
    std::string alpha_dbg;
    if (!Fn.alpha.empty()) {
      alpha_dbg = " alpha0=" + std::to_string(Fn.alpha.front()) +
                  " alpha_last=" + std::to_string(Fn.alpha.back()) +
                  " alpha_size=" + std::to_string(Fn.alpha.size());
      if (Fn.alpha.size() <= 32) {
        alpha_dbg += " alpha=[";
        for (size_t i = 0; i < Fn.alpha.size(); ++i) {
          alpha_dbg += std::to_string(Fn.alpha[i]);
          if (i + 1 < Fn.alpha.size()) alpha_dbg += ",";
        }
        alpha_dbg += "]";
      }
    }
    throw std::runtime_error("validate_suf failed for gate_kind=" +
                             std::to_string(static_cast<int>(gate_kind)) + ": " + e.what() +
                             alpha_dbg);
  }
  CompiledSUFGate out;
  out.r_in = r_in;
  out.r_out = r_out;
  out.degree = Fn.degree;
  out.r = Fn.r_out;
  out.ell = Fn.l_out;
  // Default layout names to avoid ambiguity downstream.
  out.layout.arith_ports.resize(out.r);
  for (int i = 0; i < out.r; i++) {
    out.layout.arith_ports[static_cast<size_t>(i)] = "y" + std::to_string(i);
  }
  out.layout.bool_ports.resize(out.ell);
  for (int i = 0; i < out.ell; i++) {
    out.layout.bool_ports[static_cast<size_t>(i)] = "b" + std::to_string(i);
  }
  // Gate-specific extras: keep empty unless filled by a higher-level builder.
  out.extra_u64.clear();
  out.gate_kind = gate_kind;

  std::vector<RawPredQuery> queries;
  std::unordered_map<QueryKey,int,QueryKeyHash> qmap;
  std::vector<uint64_t> wrap_bits;

  // rewrite bool outputs per piece
  for (const auto& piece : Fn.pieces) {
    std::vector<suf::BoolExpr> rewritten;
    for (const auto& b : piece.bool_outs) {
      rewritten.push_back(rewrite_bool_expr(b, Fn.primitive_preds, r_in, queries, qmap, wrap_bits));
    }
    out.bool_per_piece.push_back(std::move(rewritten));
  }

  PredProgramDesc pd;
  pd.n = Fn.n_bits;
  pd.out_mode = PredOutMode::kU64PerBit;
  pd.queries = std::move(queries);
  out.pred = std::move(pd);
  out.wrap_bits = wrap_bits;

  // Canonicalize raw predicate query ordering to help packed backends and GPU kernels:
  // group by (kind, bits_in) and sort by theta. This is semantics-preserving as long as
  // we remap the BoolExpr indices accordingly.
  if (!out.pred.queries.empty()) {
    std::vector<int> order(out.pred.queries.size());
    std::iota(order.begin(), order.end(), 0);
    auto sort_key = [&](int i) {
      const auto& q = out.pred.queries[static_cast<size_t>(i)];
      int kind = static_cast<int>(q.kind);
      int bits_in = (q.kind == RawPredKind::kLtLow) ? static_cast<int>(q.f) : 64;
      return std::tuple<int, int, uint64_t>(kind, bits_in, q.theta);
    };
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      return sort_key(a) < sort_key(b);
    });
    bool already_sorted = true;
    for (size_t i = 0; i < order.size(); ++i) {
      if (order[i] != static_cast<int>(i)) { already_sorted = false; break; }
    }
    if (!already_sorted) {
      std::vector<RawPredQuery> sorted;
      sorted.reserve(out.pred.queries.size());
      std::vector<int> new_of_old(out.pred.queries.size(), -1);
      for (size_t new_idx = 0; new_idx < order.size(); ++new_idx) {
        int old_idx = order[new_idx];
        sorted.push_back(out.pred.queries[static_cast<size_t>(old_idx)]);
        new_of_old[static_cast<size_t>(old_idx)] = static_cast<int>(new_idx);
      }
      // Remap BoolExpr indices to refer to the new query order.
      for (auto& piece : out.bool_per_piece) {
        for (auto& expr : piece) {
          expr = remap_bool_expr_pred_indices(expr, new_of_old);
        }
      }
      out.pred.queries = std::move(sorted);
    }
  }

  CoeffProgramDesc cd;
  cd.n = Fn.n_bits;
  cd.mode = coeff_mode;
  cd.out_words = Fn.r_out * (Fn.degree + 1);
  if (coeff_mode == CoeffMode::kStepDcf) {
    build_coeff_step(Fn, r_in, cd);
  } else {
    // interval LUT: rotate and split wrap
    for (size_t i = 0; i + 1 < Fn.alpha.size(); i++) {
      uint64_t a = Fn.alpha[i];
      uint64_t b = Fn.alpha[i + 1];
      auto payload = flatten_coeffs(Fn.pieces[i], Fn.degree, Fn.r_out);
      uint64_t s = a + r_in;
      uint64_t e = b + r_in;
      if (s < e) {
        cd.intervals.push_back(IntervalPayload{s, e, payload});
      } else {
        cd.intervals.push_back(IntervalPayload{s, 0, payload});
        cd.intervals.push_back(IntervalPayload{0, e, payload});
      }
    }
    std::sort(cd.intervals.begin(), cd.intervals.end(), [](const IntervalPayload& x, const IntervalPayload& y){ return x.lo < y.lo; });
  }
  out.coeff = std::move(cd);

  // Infer a safe eff_bits hint for hatx staging/packing when the coefficient
  // program does not depend on x (no cutpoints / single interval). In that
  // case, the only x-dependence comes from low-bit predicates.
  bool coeff_depends_on_x = false;
  if (out.coeff.mode == CoeffMode::kStepDcf) {
    coeff_depends_on_x = !out.coeff.cutpoints_ge.empty();
  } else {  // Interval LUT
    coeff_depends_on_x = out.coeff.intervals.size() > 1;
  }
  if (!coeff_depends_on_x) {
    bool has_full_compare = false;
    int max_low_bits = 0;
    for (const auto& q : out.pred.queries) {
      if (q.kind == RawPredKind::kLtU64) {
        has_full_compare = true;
        break;
      }
      if (q.kind == RawPredKind::kLtLow) {
        max_low_bits = std::max<int>(max_low_bits, static_cast<int>(q.f));
      }
    }
    if (!has_full_compare && max_low_bits > 0 && max_low_bits < out.pred.n) {
      out.pred.eff_bits = max_low_bits;
      out.coeff.eff_bits = max_low_bits;
    }
  }
  return out;
}

}  // namespace compiler
