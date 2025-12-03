#pragma once

#include <vector>
#include <cstdint>
#include "suf/bool_expr.hpp"
#include "compiler/pfss_program_desc.hpp"

namespace compiler {

struct CompiledSUFGate {
  uint64_t r_in = 0;
  std::vector<uint64_t> r_out;  // size r

  PredProgramDesc pred;
  CoeffProgramDesc coeff;

  // Bool outputs per interval, already rewritten to raw predicate indices
  std::vector<std::vector<suf::BoolExpr>> bool_per_piece;

  // Wrap bits produced during rewrite (one per primitive that needed it), as plain 0/1 to be secret-shared by dealer.
  std::vector<uint64_t> wrap_bits;

  int degree = 0;
  int r = 0;
  int ell = 0;
};

}  // namespace compiler
