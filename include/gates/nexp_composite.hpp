#pragma once

#include <random>
#include <vector>

#include "compiler/suf_to_pfss.hpp"
#include "gates/composite_fss.hpp"
#include "gates/nexp_gate.hpp"
#include "runtime/pfss_superbatch.hpp"
#include "suf/suf_silu_builders.hpp"  // reusable generic builder for piecewise specs

namespace gates {

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
  auto kp = gates::composite_gen_backend(suf_gate, backend, rng);
  NexpCompositeKeys out;
  out.suf = std::move(suf_gate);
  out.keys = std::move(kp);
  out.keys.k0.compiled.gate_kind = compiler::GateKind::NExp;
  out.keys.k1.compiled.gate_kind = compiler::GateKind::NExp;
  return out;
}

struct PreparedNexpJob {
  runtime::PfssHandle handle;
  nn::TensorView<uint64_t> dst;
  size_t elems = 0;
};

inline PreparedNexpJob prepare_nexp_batch(const NexpCompositeKeys& ks,
                                          const gates::CompositePartyKey& k_party,
                                          std::vector<uint64_t> hatx_public,
                                          nn::TensorView<uint64_t> out,
                                          runtime::PfssSuperBatch& batch) {
  runtime::PreparedCompositeJob job;
  job.suf = &ks.suf;
  job.key = &k_party;
  job.hook = nullptr;
  job.hatx_public = std::move(hatx_public);
  job.out = {};
  job.token = static_cast<size_t>(-1);
  auto handle = batch.enqueue_composite(std::move(job));
  PreparedNexpJob prep;
  prep.handle = handle;
  prep.dst = out;
  prep.elems = out.numel();
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
