#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>

#include "compiler/truncation_lowering.hpp"
#include "proto/reference_backend.hpp"

int main() {
  std::mt19937_64 rng(123);
  proto::ReferenceBackend backend;

  compiler::GateParams p;
  p.kind = compiler::GateKind::FaithfulTR;
  p.frac_bits = 12;
  const size_t N = 256;

  auto lowered = compiler::lower_truncation_gate(backend, rng, p, N);
  int pred_eff = lowered.keys.k0.compiled.pred.eff_bits;
  int coeff_eff = lowered.keys.k0.compiled.coeff.eff_bits;

  if (pred_eff != p.frac_bits || coeff_eff != p.frac_bits) {
    throw std::runtime_error("trunc eff_bits not propagated: pred_eff=" +
                             std::to_string(pred_eff) +
                             " coeff_eff=" + std::to_string(coeff_eff) +
                             " expected=" + std::to_string(p.frac_bits));
  }
  if (!lowered.keys.k0.compiled.coeff.cutpoints_ge.empty() ||
      !lowered.keys.k0.compiled.coeff.deltas_words.empty()) {
    throw std::runtime_error("trunc coeff program should be input-independent (no cutpoints)");
  }
  return 0;
}

