#pragma once

#include <random>
#include <vector>

#include "compiler/matmul_truncation.hpp"
#include "compiler/range_propagation.hpp"
#include "nn/matmul_beaver.hpp"
#include "proto/pfss_backend_batch.hpp"

namespace compiler {

// Aggregate output of a truncation pass: bundles for PFSS batching and attached plans.
struct TruncationPassResult {
  std::vector<const TruncationLoweringResult*> bundles;
};

// Minimal pass context that owns truncation plans so MatmulBeaverParams can
// reference them without relying on runtime auto_truncate. Intended as a
// stepping stone toward a full IR rewrite pass.
class TruncationPassContext {
 public:
  explicit TruncationPassContext(proto::PfssBackendBatch& backend, uint64_t seed = 0)
      : backend_(backend), rng_(seed) {}

  // Register a matmul rescale; returns a reference that stays valid for the
  // lifetime of this context.
  const MatmulTruncationPlan& add_matmul_plan(size_t M,
                                              size_t K,
                                              size_t N,
                                              int frac_bits,
                                              const RangeInterval& x_range,
                                              const RangeInterval& w_range) {
    matmul_plans_.push_back(
        compile_matmul_truncation(backend_, rng_, M, K, N, frac_bits, x_range, w_range));
    const auto& plan = matmul_plans_.back();
    bundles_.push_back(&plan.bundle);
    return plan;
  }

  TruncationPassResult finalize() const {
    TruncationPassResult r;
    r.bundles = bundles_;
    return r;
  }

  proto::PfssBackendBatch& backend() { return backend_; }
  const proto::PfssBackendBatch& backend() const { return backend_; }

 private:
  proto::PfssBackendBatch& backend_;
  std::mt19937_64 rng_;
  std::vector<MatmulTruncationPlan> matmul_plans_;
  std::vector<const TruncationLoweringResult*> bundles_;
};

// Attach a precomputed plan to MatmulBeaverParams (disables auto_truncate).
inline void attach_matmul_plan(nn::MatmulBeaverParams& params,
                               proto::PfssBackendBatch& backend,
                               const MatmulTruncationPlan& plan,
                               const RangeInterval& x_range,
                               const RangeInterval& w_range) {
  params.trunc_backend = &backend;
  params.trunc_plan = &plan;
  params.trunc_bundle = nullptr;
  params.require_truncation = true;
  params.x_range = x_range;
  params.w_range = w_range;
}

// Helper defined in truncation_pass.cpp to compile and attach a plan for a
// single matmul using the provided pass context.
void wire_matmul_truncation(nn::MatmulBeaverParams& params,
                            TruncationPassContext& ctx,
                            size_t M,
                            size_t K,
                            size_t N,
                            const RangeInterval& x_range,
                            const RangeInterval& w_range);

}  // namespace compiler
