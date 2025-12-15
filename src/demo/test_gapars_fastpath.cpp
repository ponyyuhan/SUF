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

  // Case 1: provably non-negative -> GapARS should compile without MSB(x) predicates.
  compiler::GateParams p;
  p.kind = compiler::GateKind::AutoTrunc;
  p.frac_bits = 8;
  p.range_hint = compiler::RangeInterval{0, static_cast<int64_t>(1ll << p.frac_bits), true};
  p.abs_hint.is_signed = true;
  p.abs_hint.max_abs = static_cast<uint64_t>(1ull << p.frac_bits);
  p.abs_hint.kind = compiler::RangeKind::Proof;
  auto b0 = compiler::lower_truncation_gate(backend, rng, p, /*batch_N=*/1);
  const auto& c0 = b0.keys.k0.compiled;
  assert(c0.gate_kind == compiler::GateKind::GapARS);
  // Only needs the wrap query in the full 64-bit domain.
  assert(count_lt_u64(c0) == 1);

  // Case 2: range crosses zero -> fall back to faithful ARS (needs MSB predicate).
  compiler::GateParams p2 = p;
  p2.range_hint = compiler::RangeInterval{-128, 127, true};
  p2.abs_hint.max_abs = 128;
  auto b1 = compiler::lower_truncation_gate(backend, rng, p2, /*batch_N=*/1);
  const auto& c1 = b1.keys.k0.compiled;
  assert(c1.gate_kind == compiler::GateKind::FaithfulARS);
  // FaithfulARS uses both wrap and an extra compare for MSB rewriting.
  assert(count_lt_u64(c1) == 2);

  std::cout << "gapars fastpath ok\n";
  return 0;
}

