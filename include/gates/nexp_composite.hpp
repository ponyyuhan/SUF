#pragma once

#include <memory>
#include <random>
#include <vector>

#include "compiler/suf_to_pfss.hpp"
#include "gates/composite_fss.hpp"
#include "gates/nexp_gate.hpp"
#include "runtime/pfss_superbatch.hpp"
#include "runtime/phase_tasks.hpp"
#include "suf/suf_silu_builders.hpp"  // reusable generic builder for piecewise specs
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

struct NexpTaskMaterial {
  suf::SUF<uint64_t> suf;
  gates::CompositeKeyPair keys;
  compiler::TruncationLoweringResult trunc_f;
  compiler::TruncationLoweringResult trunc_2f;
};

inline runtime::CubicPolyBundle make_nexp_cubic_bundle(const NexpTaskMaterial& mat,
                                                       int frac_bits) {
  runtime::CubicPolyBundle b;
  b.suf = &mat.suf;
  b.key0 = &mat.keys.k0;
  b.key1 = &mat.keys.k1;
  b.trunc_f = &mat.trunc_f;
  b.trunc_2f = &mat.trunc_2f;
  b.frac_bits = frac_bits;
  return b;
}

struct NexpCompositeKeys {
  suf::SUF<uint64_t> suf;
  gates::CompositeKeyPair keys;
};

inline NexpCompositeKeys dealer_make_nexp_composite_keys(proto::PfssBackend& backend,
                                                         const NExpGateParams& params,
                                                         std::mt19937_64& rng) {
  auto spec = make_nexp_spec(params);
  // build_silu_suf_from_piecewise works for any piecewise poly spec expanded to x-polys.
  auto suf_gate = suf::build_silu_suf_from_piecewise(spec);
  std::vector<uint64_t> r_out(static_cast<size_t>(suf_gate.r_out), 0ull);
  auto kp = gates::composite_gen_backend_with_masks(
      suf_gate, backend, rng, /*r_in=*/0ull, r_out, /*batch_N=*/1, compiler::GateKind::NExp);
  NexpCompositeKeys out;
  out.suf = std::move(suf_gate);
  out.keys = std::move(kp);
  out.keys.k0.compiled.gate_kind = compiler::GateKind::NExp;
  out.keys.k1.compiled.gate_kind = compiler::GateKind::NExp;
  return out;
}

inline NexpTaskMaterial dealer_make_nexp_task_material(proto::PfssBackendBatch& backend,
                                                       const NExpGateParams& params,
                                                       std::mt19937_64& rng,
                                                       size_t triple_need = 0,
                                                       size_t batch_N = 1) {
  auto spec = make_nexp_spec(params);
  auto suf_gate = suf::build_silu_suf_from_piecewise(spec);
  std::vector<uint64_t> r_out(static_cast<size_t>(suf_gate.r_out), 0ull);
  auto kp = gates::composite_gen_backend_with_masks(
      suf_gate, backend, rng, /*r_in=*/0ull, r_out, batch_N, compiler::GateKind::NExp);
  kp.k0.compiled.gate_kind = compiler::GateKind::NExp;
  kp.k1.compiled.gate_kind = compiler::GateKind::NExp;

  compiler::GateParams p;
  // nExp input is clamped to [0,16]; downstream products stay well within GapARS margin.
  p.kind = compiler::GateKind::GapARS;
  p.frac_bits = params.frac_bits;
  p.range_hint = compiler::RangeInterval{0, static_cast<int64_t>(1ll << params.frac_bits), true};
  p.abs_hint = compiler::abs_from_range(p.range_hint, /*is_signed=*/true);
  p.abs_hint.kind = compiler::RangeKind::Proof;
  p.gap_hint = compiler::gap_from_abs(p.abs_hint, p.frac_bits);
  auto trunc_f = compiler::lower_truncation_gate(backend, rng, p, batch_N);
  p.frac_bits = 2 * params.frac_bits;
  p.abs_hint.max_abs = static_cast<uint64_t>(1ull << p.frac_bits);
  p.gap_hint = compiler::gap_from_abs(p.abs_hint, p.frac_bits);
  auto trunc_2f = compiler::lower_truncation_gate(backend, rng, p, batch_N);

  if (triple_need > 0) {
    ensure_beaver_triples(kp, triple_need, rng);
  }

  NexpTaskMaterial out;
  out.suf = std::move(suf_gate);
  out.keys = std::move(kp);
  out.trunc_f = std::move(trunc_f);
  out.trunc_2f = std::move(trunc_2f);
  return out;
}

struct PreparedNexpJob {
  runtime::PfssHandle handle;
  nn::TensorView<uint64_t> dst;
  size_t elems = 0;
  std::unique_ptr<gates::PostProcHook> hook_storage;
  std::vector<uint64_t> hatx_public;
};

inline PreparedNexpJob prepare_nexp_batch(NexpCompositeKeys& ks,
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
  std::vector<uint64_t> hatx_copy = hatx_public;
  job.hatx_public = std::move(hatx_public);
  job.out = out;
  job.token = static_cast<size_t>(-1);
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
  PreparedNexpJob prep;
  prep.handle = handle;
  prep.dst = out;
  prep.elems = out.numel();
  prep.hook_storage = std::move(hook);
  prep.hatx_public = std::move(hatx_copy);
  return prep;
}

inline void finalize_nexp_batch(const runtime::PfssSuperBatch& batch, const PreparedNexpJob& prep) {
  if (prep.dst.data == nullptr || prep.elems == 0) return;
  auto view = batch.view(prep.handle);
  size_t n = std::min(prep.elems, view.arith_words);
  for (size_t i = 0; i < n; ++i) {
    prep.dst.data[i] = view.arith[i];
  }
}

}  // namespace gates
