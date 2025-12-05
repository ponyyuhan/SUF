#pragma once

#include "compiler/truncation_pass_runner.hpp"

namespace compiler {

// Placeholder for future pass registry; currently exposes only truncation wiring.
struct CompilerPasses {
  TruncationPassContext* trunc_ctx = nullptr;
};

}  // namespace compiler
