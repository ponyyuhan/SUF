#pragma once

#include <memory>
#include <random>
#include <vector>
#include <limits>

#include "compiler/suf_to_pfss.hpp"
#include "gates/composite_fss.hpp"
#include "gates/reciprocal_gate.hpp"
#include "runtime/phase_tasks.hpp"
#include "runtime/pfss_superbatch.hpp"
#include "proto/pfss_backend_batch.hpp"
#include "gates/tables/recip_piecewise_affine_init.hpp"
#include "suf/suf_silu_builders.hpp"  // reuse piecewise→SUF helper
#include "runtime/bench_accounting.hpp"

namespace gates {

// Bundle for a task-friendly reciprocal evaluator: coeff PFSS for affine init +
// NR trunc bundles + triples.
struct RecipTaskMaterial {
  suf::SUF<uint64_t> suf;         // piecewise affine init as SUF coeff program
  gates::CompositeKeyPair keys;   // includes triples for NR
  compiler::TruncationLoweringResult trunc_fb;   // Q2f -> Qf
  gates::PiecewisePolySpec init_spec;
  int frac_bits = 16;
  int nr_iters = 1;
};

inline void ensure_recips_triples(gates::CompositeKeyPair& kp,
                                  size_t per_iter_need,
                                  int nr_iters,
                                  size_t batch_N,
                                  std::mt19937_64& rng) {
  // One mul for the affine init plus per-iter muls, per element in the batch.
  batch_N = std::max<size_t>(batch_N, 1);
  const size_t per_elem = per_iter_need * static_cast<size_t>(nr_iters) + 1;
  size_t need = per_elem * batch_N;
  const size_t before0 = kp.k0.triples.size();
  const size_t before1 = kp.k1.triples.size();
  auto fill = [&](std::vector<proto::BeaverTriple64Share>& dst0,
                  std::vector<proto::BeaverTriple64Share>& dst1) {
    while (dst0.size() < need || dst1.size() < need) {
      uint64_t a = rng(), b = rng(), c = proto::mul_mod(a, b);
      uint64_t a0 = rng();
      uint64_t a1 = a - a0;
      uint64_t b0 = rng();
      uint64_t b1 = b - b0;
      uint64_t c0 = rng();
      uint64_t c1 = c - c0;
      dst0.push_back({a0, b0, c0});
      dst1.push_back({a1, b1, c1});
    }
  };
  fill(kp.k0.triples, kp.k1.triples);
  const size_t after0 = kp.k0.triples.size();
  const size_t after1 = kp.k1.triples.size();
  if (after0 > before0 || after1 > before1) {
    const uint64_t delta0 = static_cast<uint64_t>(after0 - before0);
    const uint64_t delta1 = static_cast<uint64_t>(after1 - before1);
    const uint64_t bytes = std::max(delta0, delta1) * static_cast<uint64_t>(sizeof(proto::BeaverTriple64Share));
    runtime::bench::add_offline_bytes(runtime::bench::OfflineBytesKind::BeaverTriple, bytes);
  }
}

inline RecipTaskMaterial dealer_make_recip_task_material(proto::PfssBackendBatch& backend,
                                                         int frac_bits,
                                                         int nr_iters,
                                                         std::mt19937_64& rng,
                                                         size_t batch_N = 1) {
  double nmax = 1024.0;
  auto spec = make_recip_affine_init_spec(frac_bits, nmax);
  auto suf_gate = suf::build_silu_suf_from_piecewise(spec);
  // eff_bits hint: ignore the +inf sentinel interval end (max int64).
  uint64_t sentinel = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  uint64_t sentinel_u64 = std::numeric_limits<uint64_t>::max();
  uint64_t max_end = 0;
  for (const auto& iv : spec.intervals) {
    if (iv.end == sentinel || iv.end == sentinel_u64) continue;
    max_end = std::max(max_end, iv.end);
  }
  auto bits_mag = [](uint64_t v) {
    int bits = 0;
    while (v) { v >>= 1; bits++; }
    return std::max(bits, 1);
  };
  int eff_bits = std::min(bits_mag(max_end) + 1, 64);
  const uint64_t r_in = rng();
  std::vector<uint64_t> r_out(static_cast<size_t>(suf_gate.r_out), 0ull);
  auto kp = gates::composite_gen_backend_with_masks(
      suf_gate, backend, rng, r_in, r_out, batch_N, compiler::GateKind::Reciprocal,
      /*pred_eff_bits_hint=*/eff_bits,
      /*coeff_eff_bits_hint=*/eff_bits);
  kp.k0.compiled.gate_kind = compiler::GateKind::Reciprocal;
  kp.k1.compiled.gate_kind = compiler::GateKind::Reciprocal;
  std::fill(kp.k0.r_out_share.begin(), kp.k0.r_out_share.end(), 0ull);
  std::fill(kp.k1.r_out_share.begin(), kp.k1.r_out_share.end(), 0ull);

  compiler::GateParams p;
  p.kind = compiler::GateKind::AutoTrunc;  // recip input is positive and bounded
  p.frac_bits = frac_bits;
  int64_t recip_hi = static_cast<int64_t>(1ll << frac_bits);
  int64_t recip_lo = static_cast<int64_t>(
      std::llround((1.0 / std::max(nmax, 1.0)) * std::ldexp(1.0, frac_bits)));
  p.range_hint = compiler::RangeInterval{recip_lo, recip_hi, true};
  p.abs_hint = compiler::abs_from_range(p.range_hint, /*is_signed=*/true);
  p.abs_hint.kind = compiler::RangeKind::Proof;
  p.gap_hint = compiler::gap_from_abs(p.abs_hint, p.frac_bits);
  auto trunc_fb = compiler::lower_truncation_gate(backend, rng, p, batch_N);
  std::fill(trunc_fb.keys.k0.r_out_share.begin(), trunc_fb.keys.k0.r_out_share.end(), 0ull);
  std::fill(trunc_fb.keys.k1.r_out_share.begin(), trunc_fb.keys.k1.r_out_share.end(), 0ull);

  // Two muls per NR iter: y*x and y*(2 - xy). Each mul needs a trunc back to Qf.
  ensure_recips_triples(kp, /*per_iter_need=*/2, nr_iters, batch_N, rng);

  RecipTaskMaterial out;
  out.suf = std::move(suf_gate);
  out.keys = std::move(kp);
  out.trunc_fb = std::move(trunc_fb);
  out.init_spec = spec;
  out.frac_bits = frac_bits;
  out.nr_iters = nr_iters;
  return out;
}

// Helper to build a task bundle consumable by a future ReciprocalTask.
inline runtime::RecipTaskBundle make_recip_bundle(const RecipTaskMaterial& mat) {
  runtime::RecipTaskBundle b{};
  b.suf = &mat.suf;
  b.key0 = &mat.keys.k0;
  b.key1 = &mat.keys.k1;
  // Reciprocal uses Qf truncs only; reuse trunc_fb for both slots.
  b.trunc_fb = &mat.trunc_fb;
  b.init_spec = &mat.init_spec;
  b.frac_bits = mat.frac_bits;
  b.nr_iters = mat.nr_iters;
  return b;
}

}  // namespace gates
