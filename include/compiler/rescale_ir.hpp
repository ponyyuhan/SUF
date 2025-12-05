#pragma once

#include <vector>

#include "compiler/range_analysis.hpp"
#include "compiler/truncation_pass.hpp"

namespace compiler {

// Minimal IR node for a rescale (truncate/ARS) op.
struct RescaleNode {
  int frac_bits = 0;
  RangeInterval range;   // conservative input range
};

// Lower a sequence of rescale nodes into truncation gate bundles.
inline std::vector<TruncationLoweringResult> lower_rescales(proto::PfssBackend& backend,
                                                            std::mt19937_64& rng,
                                                            const std::vector<RescaleNode>& nodes,
                                                            size_t batch_N = 1) {
  std::vector<TruncationLoweringResult> out;
  out.reserve(nodes.size());
  for (const auto& n : nodes) {
    out.push_back(compile_truncation_with_range(backend, rng, n.range, n.frac_bits, batch_N));
  }
  return out;
}

}  // namespace compiler
