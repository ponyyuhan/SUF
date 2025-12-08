#include "compiler/rescale_pass.hpp"

namespace compiler {

TruncationPassResult run_truncation_pass(const TruncationPassConfig& cfg,
                                         TruncationPassContext& ctx) {
  for (const auto& m : cfg.matmuls) {
    if (!m.params) continue;
    bool gap_ok = m.prefer_gapars;
    if (!gap_ok && m.gap_cert) gap_ok = can_gapars(*m.gap_cert);
    if (!gap_ok && m.accum_abs.kind == RangeKind::Proof && m.params) {
      auto g = gap_from_abs(m.accum_abs, m.params->frac_bits);
      gap_ok = g && can_gapars(*g);
    }
    wire_matmul_truncation(*m.params,
                           ctx,
                           m.M,
                           m.K,
                           m.N,
                           m.x_range,
                           m.w_range,
                           gap_ok,
                           m.gap_cert);
  }
  return ctx.finalize();
}

}  // namespace compiler
