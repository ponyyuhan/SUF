#pragma once

#include <memory>
#include <random>
#include <vector>

#include "compiler/suf_to_pfss.hpp"
#include "gates/composite_fss.hpp"
#include "gates/silu_spline_gate.hpp"
#include "runtime/pfss_superbatch.hpp"
#include "runtime/phase_tasks.hpp"
#include "suf/suf_silu_builders.hpp"
#include "proto/pfss_backend_batch.hpp"

namespace gates {

inline void ensure_beaver_triples(gates::CompositeKeyPair& keys,
                                  size_t need,
                                  std::mt19937_64& rng) {
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
  fill(keys.k0.triples, keys.k1.triples);
}

// Task-friendly bundle for CubicPolyTask (coeff PFSS + two trunc bundles).
struct SiluTaskMaterial {
  suf::SUF<uint64_t> suf;
  gates::CompositeKeyPair keys;
  compiler::TruncationLoweringResult trunc_f;
  compiler::TruncationLoweringResult trunc_2f;
};

inline SiluTaskMaterial dealer_make_silu_task_material(proto::PfssBackendBatch& backend,
                                                       int frac_bits,
                                                       std::mt19937_64& rng,
                                                       size_t triple_need = 0,
                                                       size_t batch_N = 1) {
  auto spec = make_silu_spec({frac_bits, 16});
  auto suf_gate = suf::build_silu_suf_from_piecewise(spec);
  auto kp = gates::composite_gen_backend(suf_gate, backend, rng, batch_N);
  // zero output masks so coeff payload is direct.
  std::fill(kp.k0.r_out_share.begin(), kp.k0.r_out_share.end(), 0ull);
  std::fill(kp.k1.r_out_share.begin(), kp.k1.r_out_share.end(), 0ull);
  kp.k0.compiled.gate_kind = compiler::GateKind::SiLUSpline;
  kp.k1.compiled.gate_kind = compiler::GateKind::SiLUSpline;

  compiler::GateParams p;
  p.kind = compiler::GateKind::FaithfulARS;
  p.frac_bits = frac_bits;
  auto trunc_f = compiler::lower_truncation_gate(backend, rng, p, batch_N);
  p.frac_bits = 2 * frac_bits;
  auto trunc_2f = compiler::lower_truncation_gate(backend, rng, p, batch_N);

  if (triple_need > 0) {
    ensure_beaver_triples(kp, triple_need, rng);
  }

  SiluTaskMaterial out;
  out.suf = std::move(suf_gate);
  out.keys = std::move(kp);
  out.trunc_f = std::move(trunc_f);
  out.trunc_2f = std::move(trunc_2f);
  return out;
}

// Dealer keygen for SiLU using the composite SUF pipeline.
struct SiluCompositeKeys {
  suf::SUF<uint64_t> suf;
  gates::CompositeKeyPair keys;
};

inline SiluCompositeKeys dealer_make_silu_composite_keys(proto::PfssBackend& backend,
                                                         const SiLUGateParams& params,
                                                         std::mt19937_64& rng) {
  auto spec = make_silu_spec(params);
  auto suf_gate = suf::build_silu_suf_from_piecewise(spec);
  auto kp = gates::composite_gen_backend(suf_gate, backend, rng);
  SiluCompositeKeys out;
  out.suf = std::move(suf_gate);
  out.keys = std::move(kp);
  // Coeff payload only; downstream hook performs Horner, so mask outputs are zeroed.
  std::fill(out.keys.k0.r_out_share.begin(), out.keys.k0.r_out_share.end(), 0ull);
  std::fill(out.keys.k1.r_out_share.begin(), out.keys.k1.r_out_share.end(), 0ull);
  // Mark gate kind so downstream postproc knows this is a SiLU payload (already Horner-evaluated).
  out.keys.k0.compiled.gate_kind = compiler::GateKind::SiLUSpline;
  out.keys.k1.compiled.gate_kind = compiler::GateKind::SiLUSpline;
  return out;
}

struct PreparedSiluJob {
  runtime::PfssHandle handle;
  nn::TensorView<uint64_t> dst;
  size_t elems = 0;
  std::unique_ptr<gates::PostProcHook> hook_storage;
  std::vector<uint64_t> hatx_public;
};

// Enqueue a SiLU batch into PfssSuperBatch; returns a handle for finalize.
inline PreparedSiluJob prepare_silu_batch(SiluCompositeKeys& ks,
                                          const gates::CompositePartyKey& k_party,
                                          int frac_bits,
                                          proto::PfssBackendBatch& backend,
                                          std::mt19937_64& rng,
                                          std::vector<uint64_t> hatx_public,
                                          nn::TensorView<uint64_t> out,
                                          runtime::PfssSuperBatch& batch) {
  runtime::PreparedCompositeJob job;
  job.suf = &ks.suf;
  job.key = &k_party;
  auto hook = std::make_unique<gates::HornerCubicHook>(frac_bits, k_party.r_in_share);
  hook->backend = &backend;
  job.hook = hook.get();
  // Keep a local copy of hatx for Horner postproc.
  std::vector<uint64_t> hatx_copy = hatx_public;
  job.hatx_public = std::move(hatx_public);
  job.out = out;  // let PfssSuperBatch finalize directly into destination
  job.token = static_cast<size_t>(-1);
  // Ensure enough Beaver triples for Horner (3 muls per element).
  size_t need_triples = 3 * hatx_copy.size();
  while (ks.keys.k0.triples.size() < need_triples ||
         ks.keys.k1.triples.size() < need_triples) {
    uint64_t a = rng(), b = rng(), c = proto::mul_mod(a, b);
    uint64_t a0 = rng();
    uint64_t a1 = a - a0;
    uint64_t b0 = rng();
    uint64_t b1 = b - b0;
    uint64_t c0 = rng();
    uint64_t c1 = c - c0;
    ks.keys.k0.triples.push_back({a0, b0, c0});
    ks.keys.k1.triples.push_back({a1, b1, c1});
  }
  auto handle = batch.enqueue_composite(std::move(job));
  PreparedSiluJob prep;
  prep.handle = handle;
  prep.dst = out;
  prep.elems = out.numel();
  prep.hook_storage = std::move(hook);
  prep.hatx_public = std::move(hatx_copy);
  return prep;
}

// After flush_and_finalize, copy the masked outputs from PFSS view into dst.
inline void finalize_silu_batch(const runtime::PfssSuperBatch& batch, const PreparedSiluJob& prep) {
  if (prep.dst.data == nullptr || prep.elems == 0) return;
  auto view = batch.view(prep.handle);
  size_t n = std::min(prep.elems, view.arith_words);
  for (size_t i = 0; i < n; ++i) {
    prep.dst.data[i] = view.arith[i];
  }
}

}  // namespace gates
