#pragma once

#include <random>
#include <utility>
#include <vector>
#include "compiler/suf_collect.hpp"
#include "pfss/pfss.hpp"
#include "pfss/program_desc.hpp"
#include "core/ring.hpp"
#include "suf/suf_ir.hpp"

namespace compiler {

// What the dealer gives each party for one SUF gate instance/type.
struct CompiledSUFKeys {
  pfss::Key pred_key0, pred_key1;
  pfss::Key coeff_key0, coeff_key1;

  // masks (shared). r_in is on input wire; r_out per arithmetic output.
  std::vector<uint64_t> r_out_share0, r_out_share1;
  uint64_t r_in_share0 = 0, r_in_share1 = 0;

  // metadata
  int n_bits = 0;
  int r_out = 0;
  int l_out = 0;
  int degree = 0;
  int num_primitive_preds = 0;
  int packed_pred_words = 0;  // e.g. ceil(T/64)
};

// Helper: split a ring element into two additive shares
inline std::pair<uint64_t, uint64_t> split_u64(std::mt19937_64& g, uint64_t x) {
  uint64_t s0 = g();
  uint64_t s1 = x - s0;
  return {s0, s1};
}

// Build coefficient LUT pieces in masked hat{x}-domain (split wrap-around intervals).
inline pfss_desc::PiecewiseVectorDesc
compile_coeff_piecewise(const suf::SUF<core::Z2n<64>>& F, uint64_t r_in) {
  // Payload per interval: concat coefficients for each arithmetic output.
  pfss_desc::PiecewiseVectorDesc desc;
  desc.n_bits = F.n_bits;

  const bool full64 = (F.n_bits == 64);
  uint64_t mod_mask = full64 ? ~0ull : ((1ull << F.n_bits) - 1);
  auto norm = [&](uint64_t v) { return full64 ? v : (v & mod_mask); };
  auto shift = [&](uint64_t a) { return norm(a + r_in); };

  if (F.alpha.size() != F.pieces.size() + 1) return desc;  // malformed

  for (size_t i = 0; i < F.pieces.size(); i++) {
    uint64_t A = norm(F.alpha[i]);
    uint64_t B = norm(F.alpha[i + 1]);
    uint64_t L = shift(A);
    uint64_t U = shift(B);

    std::vector<uint64_t> payload;
    payload.reserve(static_cast<size_t>(F.r_out) * static_cast<size_t>(F.degree + 1));
    for (int j = 0; j < F.r_out; j++) {
      const auto& poly = F.pieces[i].polys[static_cast<size_t>(j)];
      for (auto c : poly.coeffs) payload.push_back(c.v);
    }

    bool wrap = (U < L);
    if (!wrap) {
      desc.pieces.push_back({L, U, payload});
    } else {
      desc.pieces.push_back({0, U, payload});
      desc.pieces.push_back({L, full64 ? ~0ull : mod_mask, payload});
    }
  }
  return desc;
}

template<typename PredPayloadT, typename CoeffPayloadT>
inline CompiledSUFKeys dealer_compile_suf_gate(
    pfss::Backend<PredPayloadT>& pred_backend,
    pfss::Backend<CoeffPayloadT>& coeff_backend,
    const pfss::PublicParams& pp_pred,
    const pfss::PublicParams& pp_coeff,
    const suf::SUF<core::Z2n<64>>& F,
    std::mt19937_64& g) {
  // 1) sample masks
  auto [r0, r1] = split_u64(g, g());  // r_in random; represented as shares
  uint64_t r_in = r0 + r1;

  std::vector<uint64_t> r_out0(F.r_out), r_out1(F.r_out);
  for (int j = 0; j < F.r_out; j++) {
    auto [s0, s1] = split_u64(g, g());
    r_out0[static_cast<size_t>(j)] = s0;
    r_out1[static_cast<size_t>(j)] = s1;
  }

  // 2) compile predicate desc in masked coordinate (dealer-only)
  auto pred_bits = compile_primitive_pred_bits(F, r_in);
  pfss::ProgramDesc pred_prog;
  pred_prog.kind = "predicates";
  pred_prog.dealer_only_desc = pfss_desc::serialize_pred_bits(pred_bits);

  // 3) compile coefficient piecewise LUT desc
  auto coeff_desc = compile_coeff_piecewise(F, r_in);
  pfss::ProgramDesc coeff_prog;
  coeff_prog.kind = "coeff_lut";
  coeff_prog.dealer_only_desc = pfss_desc::serialize_piecewise(coeff_desc);

  // 4) ProgGen -> keys
  auto [pk0, pk1] = pred_backend.prog_gen(pp_pred, pred_prog);
  auto [ck0, ck1] = coeff_backend.prog_gen(pp_coeff, coeff_prog);

  CompiledSUFKeys out;
  out.pred_key0 = std::move(pk0);
  out.pred_key1 = std::move(pk1);
  out.coeff_key0 = std::move(ck0);
  out.coeff_key1 = std::move(ck1);
  out.r_in_share0 = r0;
  out.r_in_share1 = r1;
  out.r_out_share0 = std::move(r_out0);
  out.r_out_share1 = std::move(r_out1);
  out.n_bits = F.n_bits;
  out.r_out = F.r_out;
  out.l_out = F.l_out;
  out.degree = F.degree;
  out.num_primitive_preds = static_cast<int>(F.primitive_preds.size());
  out.packed_pred_words = (out.num_primitive_preds + 63) / 64;
  return out;
}

}  // namespace compiler
