#pragma once

#include <random>

#include "compiler/range_analysis.hpp"
#include "compiler/truncation_lowering.hpp"
#include "proto/pfss_backend_batch.hpp"

namespace compiler {

// Bundle capturing truncation lowering + metadata for a single matmul.
struct MatmulTruncationPlan {
  TruncationLoweringResult bundle;
  GateKind kind = GateKind::FaithfulARS;
  RangeInterval accum_range;
  size_t batch = 0;

  MatmulTruncationPlan() = default;
  MatmulTruncationPlan(const MatmulTruncationPlan&) = delete;
  MatmulTruncationPlan& operator=(const MatmulTruncationPlan&) = delete;
  MatmulTruncationPlan(MatmulTruncationPlan&&) noexcept = default;
  MatmulTruncationPlan& operator=(MatmulTruncationPlan&&) noexcept = default;
};

// Compile trunc/ARS plan for an MxK * KxN matmul given operand ranges.
inline MatmulTruncationPlan compile_matmul_truncation(proto::PfssBackendBatch& backend,
                                                      std::mt19937_64& rng,
                                                      size_t M,
                                                      size_t K,
                                                      size_t N,
                                                      int frac_bits,
                                                      const RangeInterval& x_range = RangeInterval::whole(true),
                                                      const RangeInterval& w_range = RangeInterval::whole(true),
                                                      bool prefer_gapars = false) {
  RangeInterval accum = matmul_accum_range(x_range, w_range, K);
  GateParams params;
  params.frac_bits = frac_bits;
  params.kind = select_trunc_kind(accum, frac_bits);
  if (prefer_gapars && has_gap_cert(accum)) {
    params.kind = GateKind::GapARS;
  }
  size_t total = M * N;
  MatmulTruncationPlan plan;
  plan.kind = params.kind;
  plan.accum_range = accum;
  plan.batch = total;
  plan.bundle = lower_truncation_gate(backend, rng, params, total);
  return plan;
}

}  // namespace compiler
