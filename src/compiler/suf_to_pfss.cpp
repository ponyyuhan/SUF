#include "compiler/suf_to_pfss.hpp"

#include <algorithm>
#include <unordered_map>
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

static suf::BoolExpr make_wrap_expr(uint64_t wrap_idx_offset, int idx_raw, int idx_raw2) {
  // w = wrap ? ((!a)|b) : ((!a)&b)
  suf::BoolExpr a{ suf::BVar{idx_raw} };
  suf::BoolExpr b{ suf::BVar{idx_raw2} };
  suf::BoolExpr na{ suf::BNot{ std::make_unique<suf::BoolExpr>(a) } };
  suf::BoolExpr and_expr{ suf::BAnd{ std::make_unique<suf::BoolExpr>(na), std::make_unique<suf::BoolExpr>(b) } };
  suf::BoolExpr or_expr{ suf::BOr{ std::make_unique<suf::BoolExpr>(na), std::make_unique<suf::BoolExpr>(b) } };
  // Use negative indices to refer to wrap bits; evaluator interprets them with offset.
  suf::BoolExpr wrap_var{ suf::BVar{ -1 - static_cast<int>(wrap_idx_offset) } };
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
      return make_wrap_expr(queries.size() + wrap_idx, a, b);
    } else if constexpr (std::is_same_v<T, suf::Pred_X_mod2f_lt>) {
      auto rec = suf::rewrite_ltlow(r_in, n.f, n.gamma);
      int a = add_query(RawPredQuery{RawPredKind::kLtLow, static_cast<uint8_t>(rec.f), rec.theta0}, queries, mp);
      int b = add_query(RawPredQuery{RawPredKind::kLtLow, static_cast<uint8_t>(rec.f), rec.theta1}, queries, mp);
      size_t wrap_idx = wrap_bits.size();
      wrap_bits.push_back(rec.wrap);
      return make_wrap_expr(queries.size() + wrap_idx, a, b);
    } else if constexpr (std::is_same_v<T, suf::Pred_MSB_x>) {
      auto rec = suf::rewrite_msb_add(r_in, 0);
      int a = add_query(RawPredQuery{RawPredKind::kLtU64, 64, rec.theta0}, queries, mp);
      int b = add_query(RawPredQuery{RawPredKind::kLtU64, 64, rec.theta1}, queries, mp);
      size_t wrap_idx = wrap_bits.size();
      wrap_bits.push_back(rec.wrap);
      return make_wrap_expr(queries.size() + wrap_idx, a, b);
    } else { // MSB(x+c)
      auto rec = suf::rewrite_msb_add(r_in, n.c);
      int a = add_query(RawPredQuery{RawPredKind::kLtU64, 64, rec.theta0}, queries, mp);
      int b = add_query(RawPredQuery{RawPredKind::kLtU64, 64, rec.theta1}, queries, mp);
      size_t wrap_idx = wrap_bits.size();
      wrap_bits.push_back(rec.wrap);
      return make_wrap_expr(queries.size() + wrap_idx, a, b);
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
  for (size_t i = 1; i < segs.size(); i++) {
    out.cutpoints_ge.push_back(segs[i].start);
    std::vector<uint64_t> delta(out.out_words, 0);
    for (int j = 0; j < out.out_words; j++) {
      delta[static_cast<size_t>(j)] = segs[i].payload[static_cast<size_t>(j)] - segs[i - 1].payload[static_cast<size_t>(j)];
    }
    out.deltas_words.push_back(std::move(delta));
  }
}

CompiledSUFGate compile_suf_to_pfss_two_programs(
    const suf::SUF<uint64_t>& F,
    uint64_t r_in,
    const std::vector<uint64_t>& r_out,
    CoeffMode coeff_mode) {
  validate_suf(F);
  CompiledSUFGate out;
  out.r_in = r_in;
  out.r_out = r_out;
  out.degree = F.degree;
  out.r = F.r_out;
  out.ell = F.l_out;
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

  std::vector<RawPredQuery> queries;
  std::unordered_map<QueryKey,int,QueryKeyHash> qmap;
  std::vector<uint64_t> wrap_bits;

  // rewrite bool outputs per piece
  for (const auto& piece : F.pieces) {
    std::vector<suf::BoolExpr> rewritten;
    for (const auto& b : piece.bool_outs) {
      rewritten.push_back(rewrite_bool_expr(b, F.primitive_preds, r_in, queries, qmap, wrap_bits));
    }
    out.bool_per_piece.push_back(std::move(rewritten));
  }

  PredProgramDesc pd;
  pd.n = F.n_bits;
  pd.out_mode = PredOutMode::kU64PerBit;
  pd.queries = std::move(queries);
  out.pred = std::move(pd);
  out.wrap_bits = wrap_bits;

  CoeffProgramDesc cd;
  cd.n = F.n_bits;
  cd.mode = coeff_mode;
  cd.out_words = F.r_out * (F.degree + 1);
  if (coeff_mode == CoeffMode::kStepDcf) {
    build_coeff_step(F, r_in, cd);
  } else {
    // interval LUT: rotate and split wrap
    for (size_t i = 0; i + 1 < F.alpha.size(); i++) {
      uint64_t a = F.alpha[i];
      uint64_t b = F.alpha[i + 1];
      auto payload = flatten_coeffs(F.pieces[i], F.degree, F.r_out);
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
  return out;
}

}  // namespace compiler
