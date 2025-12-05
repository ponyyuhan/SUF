#pragma once

#include <random>
#include <vector>

#include "compiler/suf_to_pfss.hpp"
#include "gates/composite_fss.hpp"
#include "gates/silu_spline_gate.hpp"
#include "runtime/pfss_superbatch.hpp"
#include "suf/suf_silu_builders.hpp"

namespace gates {

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
  // Mark gate kind so downstream postproc knows this is a SiLU payload (already Horner-evaluated).
  out.keys.k0.compiled.gate_kind = compiler::GateKind::SiLUSpline;
  out.keys.k1.compiled.gate_kind = compiler::GateKind::SiLUSpline;
  return out;
}

struct PreparedSiluJob {
  runtime::PfssHandle handle;
  nn::TensorView<uint64_t> dst;
  size_t elems = 0;
};

// Enqueue a SiLU batch into PfssSuperBatch; returns a handle for finalize.
inline PreparedSiluJob prepare_silu_batch(const SiluCompositeKeys& ks,
                                          const gates::CompositePartyKey& k_party,
                                          std::vector<uint64_t> hatx_public,
                                          nn::TensorView<uint64_t> out,
                                          runtime::PfssSuperBatch& batch) {
  runtime::PreparedCompositeJob job;
  job.suf = &ks.suf;
  job.key = &k_party;
  job.hook = nullptr;  // Horner already inside composite eval.
  job.hatx_public = std::move(hatx_public);
  job.out = {};  // defer writing via handle/view to showcase new API
  job.token = static_cast<size_t>(-1);
  auto handle = batch.enqueue_composite(std::move(job));
  PreparedSiluJob prep;
  prep.handle = handle;
  prep.dst = out;
  prep.elems = out.numel();
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
