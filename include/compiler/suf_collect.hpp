#pragma once

#include <vector>
#include <variant>
#include "core/ring.hpp"
#include "compiler/mask_rewrite.hpp"
#include "pfss/program_desc.hpp"
#include "suf/suf_ir.hpp"
#include "suf/predicates.hpp"

namespace compiler {

// Collect primitive predicate bits needed, and rewrite each primitive into one RotInterval over hat{x}.
inline std::vector<pfss_desc::PredBitDesc>
compile_primitive_pred_bits(const suf::SUF<core::Z2n<64>>& F, uint64_t r_in) {
  std::vector<pfss_desc::PredBitDesc> out;
  out.reserve(F.primitive_preds.size());

  for (const auto& pr : F.primitive_preds) {
    using namespace suf;
    if (std::holds_alternative<Pred_X_lt_const>(pr)) {
      auto beta = std::get<Pred_X_lt_const>(pr).beta;
      auto R = rotate_prefix(F.n_bits, r_in, beta);  // hat{x} in rotated interval
      out.push_back({F.n_bits, R.ranges});
    } else if (std::holds_alternative<Pred_X_mod2f_lt>(pr)) {
      auto p = std::get<Pred_X_mod2f_lt>(pr);
      uint64_t delta = (p.f == 64) ? r_in : (r_in & ((1ull << p.f) - 1));
      auto R = rotate_lowbits(p.f, delta, p.gamma);
      out.push_back({p.f, R.ranges});
    } else if (std::holds_alternative<Pred_MSB_x>(pr)) {
      uint64_t beta = (F.n_bits == 64) ? (1ull << 63) : (1ull << (F.n_bits - 1));
      auto R = rotate_prefix(F.n_bits, r_in, beta);  // gives 1[x<beta]
      out.push_back({F.n_bits, R.ranges});
    } else if (std::holds_alternative<Pred_MSB_x_plus>(pr)) {
      auto c = std::get<Pred_MSB_x_plus>(pr).c;
      uint64_t beta = (F.n_bits == 64) ? (1ull << 63) : (1ull << (F.n_bits - 1));
      uint64_t r_eff = r_in - c;  // because test is on (x+c)
      auto R = rotate_prefix(F.n_bits, r_eff, beta);
      out.push_back({F.n_bits, R.ranges});
    }
  }
  return out;
}

}  // namespace compiler
