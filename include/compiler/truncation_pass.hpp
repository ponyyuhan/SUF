#pragma once

#include <random>

#include "compiler/range_analysis.hpp"
#include "compiler/matmul_truncation.hpp"
#include "compiler/truncation_lowering.hpp"

namespace compiler {

// Simple end-to-end helper that (1) picks GateKind from a range bound, and
// (2) produces the composite keys/hooks bundle for runtime execution.
inline TruncationLoweringResult compile_truncation_with_range(proto::PfssBackend& backend,
                                                              std::mt19937_64& rng,
                                                              const RangeInterval& range,
                                                              int frac_bits,
                                                              size_t batch_N = 1) {
  GateParams params;
  params.frac_bits = frac_bits;
  params.range_hint = range;
  params.abs_hint = abs_from_range(range, range.is_signed);
  params.kind = select_trunc_kind(params.abs_hint, frac_bits);
  return lower_truncation_gate(backend, rng, params, batch_N);
}

// Convenience: produce a matmul-specific plan (accum range + lowering) using operand ranges.
inline MatmulTruncationPlan plan_matmul_rescale(proto::PfssBackendBatch& backend,
                                                std::mt19937_64& rng,
                                                size_t M,
                                                size_t K,
                                                size_t N,
                                                int frac_bits,
                                                const RangeInterval& x_range,
                                                const RangeInterval& w_range) {
  return compile_matmul_truncation(backend, rng, M, K, N, frac_bits, x_range, w_range);
}

}  // namespace compiler
