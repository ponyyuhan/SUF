#pragma once

#include <random>
#include <vector>

#include "gates/nexp_composite.hpp"
#include "gates/reciprocal_gate.hpp"
#include "gates/composite_fss.hpp"
#include "runtime/pfss_superbatch.hpp"
#include "proto/pfss_backend_batch.hpp"

namespace gates {

struct SoftmaxCompositePrep {
  PreparedNexpJob exp_job;
  runtime::PfssHandle recip_handle;
  std::vector<uint64_t> recip_hatx;
};

// Placeholder: reuse nExp composite for exp part; reciprocal still legacy for now.
inline SoftmaxCompositePrep prepare_softmax_exp(NexpCompositeKeys& ks,
                                                const gates::CompositePartyKey& k_party,
                                                int frac_bits,
                                                proto::PfssBackendBatch& backend,
                                                std::mt19937_64& rng,
                                                std::vector<uint64_t> hatx_public,
                                                nn::TensorView<uint64_t> exp_out,
                                                runtime::PfssSuperBatch& batch) {
  SoftmaxCompositePrep prep;
  prep.exp_job = prepare_nexp_batch(ks, k_party, frac_bits,
                                    backend, rng, std::move(hatx_public), exp_out, batch);
  return prep;
}

inline void finalize_softmax_exp(const runtime::PfssSuperBatch& batch,
                                 SoftmaxCompositePrep& prep) {
  finalize_nexp_batch(batch, prep.exp_job);
}

}  // namespace gates
