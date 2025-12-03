#pragma once

#include <vector>
#include <cstdint>
#include "suf/suf_ir.hpp"
#include "compiler/compiled_suf_gate.hpp"
#include "compiler/pfss_program_desc.hpp"

namespace compiler {

// Baseline compiler: produces step-DCF style coeff program by default.
CompiledSUFGate compile_suf_to_pfss_two_programs(
    const suf::SUF<uint64_t>& F,
    uint64_t r_in,
    const std::vector<uint64_t>& r_out,
    CoeffMode coeff_mode = CoeffMode::kStepDcf);

}  // namespace compiler
