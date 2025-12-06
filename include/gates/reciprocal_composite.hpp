#pragma once

#include <memory>
#include <random>
#include <vector>

#include "compiler/suf_to_pfss.hpp"
#include "gates/composite_fss.hpp"
#include "gates/reciprocal_gate.hpp"
#include "runtime/phase_tasks.hpp"
#include "runtime/pfss_superbatch.hpp"
#include "proto/pfss_backend_batch.hpp"
#include "gates/tables/recip_piecewise_affine_init.hpp"
#include "suf/suf_silu_builders.hpp"  // reuse piecewise→SUF helper

namespace gates {

// Bundle for a task-friendly reciprocal evaluator: coeff PFSS for affine init +
// NR trunc bundles + triples.
struct RecipTaskMaterial {
  suf::SUF<uint64_t> suf;         // piecewise affine init as SUF coeff program
  gates::CompositeKeyPair keys;   // includes triples for NR
  compiler::TruncationLoweringResult trunc_fb;   // Q2f -> Qf
  int frac_bits = 16;
  int nr_iters = 1;
};

inline void ensure_recips_triples(gates::CompositeKeyPair& kp,
                                  size_t per_iter_need,
                                  int nr_iters,
                                  std::mt19937_64& rng) {
  // One mul for the affine init plus per-iter muls.
  size_t need = per_iter_need * static_cast<size_t>(nr_iters) + 1;
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
}

inline RecipTaskMaterial dealer_make_recip_task_material(proto::PfssBackendBatch& backend,
                                                         int frac_bits,
                                                         int nr_iters,
                                                         std::mt19937_64& rng) {
  auto spec = make_recip_affine_init_spec(frac_bits, /*nmax=*/1024.0);
  auto suf_gate = suf::build_silu_suf_from_piecewise(spec);
  std::vector<uint64_t> r_out(static_cast<size_t>(suf_gate.r_out), 0ull);
  auto kp = gates::composite_gen_backend_with_masks(suf_gate, backend, rng, rng(), r_out);
  kp.k0.compiled.gate_kind = compiler::GateKind::Reciprocal;
  kp.k1.compiled.gate_kind = compiler::GateKind::Reciprocal;

  compiler::GateParams p;
  p.kind = compiler::GateKind::GapARS;  // recip input is positive and bounded
  p.frac_bits = frac_bits;
  auto trunc_fb = compiler::lower_truncation_gate(backend, rng, p);

  // Two muls per NR iter: y*x and y*(2 - xy). Each mul needs a trunc back to Qf.
  ensure_recips_triples(kp, /*per_iter_need=*/2, nr_iters, rng);

  RecipTaskMaterial out;
  out.suf = std::move(suf_gate);
  out.keys = std::move(kp);
  out.trunc_fb = std::move(trunc_fb);
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
  b.frac_bits = mat.frac_bits;
  b.nr_iters = mat.nr_iters;
  return b;
}

}  // namespace gates
