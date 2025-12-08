#include "compiler/passes.hpp"
#include "compiler/range_propagation.hpp"
#include "compiler/rescale_pass.hpp"
#include "nn/matmul_beaver.hpp"

namespace compiler {

// Minimal skeleton: given MatmulBeaverParams and operand ranges, attach a plan
// using the provided TruncationPassContext. In a full IR pass, the call sites
// would be generated automatically when encountering a rescale node.
void wire_matmul_truncation(nn::MatmulBeaverParams& params,
                            TruncationPassContext& ctx,
                            size_t M,
                            size_t K,
                            size_t N,
                            const RangeInterval& x_range,
                            const RangeInterval& w_range,
                            bool prefer_gapars,
                            std::optional<GapCert> gap_cert) {
  const auto& plan =
      ctx.add_matmul_plan(M, K, N, params.frac_bits, x_range, w_range, prefer_gapars, gap_cert);
  attach_matmul_plan(params, ctx.backend(), plan, x_range, w_range);
}

}  // namespace compiler
