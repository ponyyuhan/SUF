#include "compiler/rescale_pass.hpp"

namespace compiler {

TruncationPassResult run_truncation_pass(const TruncationPassConfig& cfg,
                                         TruncationPassContext& ctx) {
  for (const auto& m : cfg.matmuls) {
    if (!m.params) continue;
    wire_matmul_truncation(*m.params, ctx, m.M, m.K, m.N, m.x_range, m.w_range);
  }
  return ctx.finalize();
}

}  // namespace compiler
