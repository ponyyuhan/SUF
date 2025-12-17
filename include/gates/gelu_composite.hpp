#pragma once

#include <limits>
#include <random>
#include <vector>

#include "compiler/truncation_lowering.hpp"
#include "gates/composite_fss.hpp"
#include "gates/silu_composite.hpp"
#include "gates/tables/gelu_spline_table.hpp"
#include "nn/layer_context.hpp"
#include "proto/pfss_backend_batch.hpp"
#include "suf/suf_gelu_builders.hpp"

namespace gates {

// Task-friendly bundle for CubicPolyTask (coeff PFSS + two trunc bundles).
struct GeluTaskMaterial {
  suf::SUF<uint64_t> suf;
  gates::CompositeKeyPair keys;
  compiler::TruncationLoweringResult trunc_f;
  compiler::TruncationLoweringResult trunc_2f;
  gates::PiecewisePolySpec spec;
};

inline GeluTaskMaterial dealer_make_gelu_task_material(proto::PfssBackendBatch& backend,
                                                       int frac_bits,
                                                       std::mt19937_64& rng,
                                                       size_t triple_need = 0,
                                                       size_t batch_N = 1,
                                                       int segments = 16) {
  auto spec = gates::make_gelu_spline_spec(frac_bits, segments);
  auto suf_gate = suf::build_gelu_suf_from_piecewise(spec);

  const uint64_t r_in = rng();
  std::vector<uint64_t> r_out(static_cast<size_t>(suf_gate.r_out), 0ull);
  auto kp = gates::composite_gen_backend_with_masks(
      suf_gate, backend, rng, r_in, r_out, batch_N, compiler::GateKind::GeLUSpline);
  kp.k0.compiled.gate_kind = compiler::GateKind::GeLUSpline;
  kp.k1.compiled.gate_kind = compiler::GateKind::GeLUSpline;

  compiler::GateParams p;
  p.kind = compiler::GateKind::AutoTrunc;
  p.range_hint = nn::clamp_gelu_range(frac_bits);
  p.abs_hint = compiler::abs_from_range(p.range_hint, /*is_signed=*/true);
  p.abs_hint.kind = compiler::RangeKind::Proof;
  p.gap_hint = compiler::gap_from_abs(p.abs_hint, frac_bits);
  p.frac_bits = frac_bits;
  auto trunc_f = compiler::lower_truncation_gate(backend, rng, p, batch_N);
  p.frac_bits = 2 * frac_bits;
  p.abs_hint.max_abs = (p.frac_bits >= 63) ? std::numeric_limits<uint64_t>::max()
                                           : static_cast<uint64_t>(1ull << p.frac_bits);
  p.gap_hint = compiler::gap_from_abs(p.abs_hint, p.frac_bits);
  auto trunc_2f = compiler::lower_truncation_gate(backend, rng, p, batch_N);

  if (triple_need > 0) {
    gates::ensure_beaver_triples(kp, triple_need, rng);
  }

  GeluTaskMaterial out;
  out.spec = std::move(spec);
  out.suf = std::move(suf_gate);
  out.keys = std::move(kp);
  out.trunc_f = std::move(trunc_f);
  out.trunc_2f = std::move(trunc_2f);
  return out;
}

// Task-friendly bundle for a degree-0 GeLU approximation that emits the evaluated value directly
// as a single arithmetic payload word (so CubicPolyTask can early-return after PFSS).
struct GeluConstTaskMaterial {
  suf::SUF<uint64_t> suf;
  gates::CompositeKeyPair keys;
  gates::PiecewisePolySpec spec;
};

inline GeluConstTaskMaterial dealer_make_gelu_const_task_material(proto::PfssBackendBatch& backend,
                                                                  int eff_bits,
                                                                  int frac_bits,
                                                                  std::mt19937_64& rng,
                                                                  size_t batch_N = 1,
                                                                  int segments = 256) {
  auto spec = gates::make_gelu_spline_spec(frac_bits, segments);
  auto suf_gate = suf::build_gelu_suf_const_from_piecewise(spec, eff_bits);

  uint64_t r_in = rng();
  if (eff_bits > 0 && eff_bits < 64) r_in &= ((uint64_t(1) << eff_bits) - 1);
  std::vector<uint64_t> r_out(static_cast<size_t>(suf_gate.r_out), 0ull);
  auto kp = gates::composite_gen_backend_with_masks(
      suf_gate, backend, rng, r_in, r_out, batch_N, compiler::GateKind::GeLUSpline);
  kp.k0.compiled.gate_kind = compiler::GateKind::GeLUSpline;
  kp.k1.compiled.gate_kind = compiler::GateKind::GeLUSpline;

  GeluConstTaskMaterial out;
  out.spec = std::move(spec);
  out.suf = std::move(suf_gate);
  out.keys = std::move(kp);
  return out;
}

}  // namespace gates
