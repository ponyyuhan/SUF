#pragma once

#include <vector>

#include "compiler/range_propagation.hpp"
#include "compiler/truncation_pass_runner.hpp"
#include "nn/matmul_beaver.hpp"

namespace compiler {

// Describes a matmul site that currently performs a local shift rescale.
struct MatmulRescaleSite {
  nn::MatmulBeaverParams* params = nullptr;  // target to patch
  size_t M = 0;
  size_t K = 0;
  size_t N = 0;
  RangeInterval x_range = RangeInterval::whole(true);
  RangeInterval w_range = RangeInterval::whole(true);
};

// Future extension: linop/activation rescale sites could be added here.
struct TruncationPassConfig {
  std::vector<MatmulRescaleSite> matmuls;
};

// Run a truncation pass: for each rescale site, compute GateKind from ranges,
// attach a precomputed plan, and collect bundles for PFSS batching.
TruncationPassResult run_truncation_pass(const TruncationPassConfig& cfg,
                                         TruncationPassContext& ctx);

}  // namespace compiler
