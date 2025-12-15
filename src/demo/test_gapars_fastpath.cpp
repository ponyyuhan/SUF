#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>

#include "compiler/truncation_lowering.hpp"
#include "proto/backend_clear.hpp"

static size_t count_lt_u64(const compiler::CompiledSUFGate& g) {
  size_t n = 0;
  for (const auto& q : g.pred.queries) {
    if (q.kind == compiler::RawPredKind::kLtU64) n++;
  }
  return n;
}

int main() {
  proto::ClearBackend backend;
  std::mt19937_64 rng(0x1234);

  // Case 1: proof-grade small bound -> GapARS should compile without any 64-bit compares.
  compiler::GateParams p;
  p.kind = compiler::GateKind::AutoTrunc;
  p.frac_bits = 8;
  p.range_hint = compiler::RangeInterval{-128, 127, true};
  p.abs_hint.is_signed = true;
  p.abs_hint.max_abs = 128;
  p.abs_hint.kind = compiler::RangeKind::Proof;
  auto b0 = compiler::lower_truncation_gate(backend, rng, p, /*batch_N=*/1);
  const auto& c0 = b0.keys.k0.compiled;
  assert(c0.gate_kind == compiler::GateKind::GapARS);
  // GapARS uses only low-bit predicates (no full-width wrap/MSB comparisons).
  assert(count_lt_u64(c0) == 0);

  // Case 2: hint-only bounds -> must fall back to faithful ARS (needs full-width compares).
  compiler::GateParams p2 = p;
  p2.abs_hint.kind = compiler::RangeKind::Hint;
  auto b1 = compiler::lower_truncation_gate(backend, rng, p2, /*batch_N=*/1);
  const auto& c1 = b1.keys.k0.compiled;
  assert(c1.gate_kind == compiler::GateKind::FaithfulARS);
  // FaithfulARS uses both wrap and an extra compare for MSB rewriting.
  assert(count_lt_u64(c1) == 2);

  std::cout << "gapars fastpath ok\n";
  return 0;
}
