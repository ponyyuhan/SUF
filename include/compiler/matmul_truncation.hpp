#pragma once

#include <random>
#include <optional>

#include "compiler/range_analysis.hpp"
#include "compiler/truncation_lowering.hpp"
#include "proto/pfss_backend_batch.hpp"

namespace compiler {

// Bundle capturing truncation lowering + metadata for a single matmul.
struct MatmulTruncationPlan {
  TruncationLoweringResult bundle;
  GateKind kind = GateKind::FaithfulARS;
  RangeInterval accum_range;
  AbsBound accum_abs;
  std::optional<GapCert> gap_cert;
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
                                                      bool prefer_gapars = false,
                                                      std::optional<GapCert> gap_cert = std::nullopt) {
  RangeInterval accum = matmul_accum_range(x_range, w_range, K);
  AbsBound accum_abs = matmul_accum_abs(abs_from_range(x_range, x_range.is_signed),
                                        abs_from_range(w_range, true),
                                        K);
  GateParams params;
  params.frac_bits = frac_bits;
  if (prefer_gapars) {
    accum_abs.kind = RangeKind::Proof;
  }
  // Matmul outputs: opt in to per-element masks for better masking hygiene.
  params.per_element_masks = true;
  params.abs_hint = accum_abs;
  params.gap_hint = gap_cert;
  params.kind = select_trunc_kind(accum_abs, frac_bits, gap_cert);
  if (prefer_gapars && gap_cert && gap_cert->kind != RangeKind::Proof) {
    params.kind = GateKind::FaithfulARS;
  }
  size_t total = M * N;
  MatmulTruncationPlan plan;
  plan.kind = params.kind;
  plan.accum_range = accum;
  plan.accum_abs = accum_abs;
  plan.gap_cert = gap_cert;
  plan.batch = total;
  plan.bundle = lower_truncation_gate(backend, rng, params, total);
  return plan;
}

}  // namespace compiler
