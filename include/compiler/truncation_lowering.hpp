#pragma once

#include <memory>
#include <random>

#include "compiler/pfss_program_desc.hpp"
#include "gates/composite_fss.hpp"
#include "gates/postproc_hooks.hpp"

namespace compiler {

// Convenience struct bundling everything needed to run a trunc/ARS gate through Composite:
// - compiled new + keys
// - per-party postproc hooks configured with frac bits and mask shares
struct TruncationLoweringResult {
  suf::SUF<uint64_t> suf;                // predicate-only new
  gates::CompositeKeyPair keys;          // PFSS keys + masks
  std::unique_ptr<gates::PostProcHook> hook0;  // postproc for party 0
  std::unique_ptr<gates::PostProcHook> hook1;  // postproc for party 1
};

// Lower a rescale/trunc/ARS GateKind into a new + composite keys + postproc hooks.
inline TruncationLoweringResult lower_truncation_gate(proto::PfssBackend& backend,
                                                      std::mt19937_64& rng,
                                                      const GateParams& params,
                                                      size_t batch_N = 1) {
  if (params.kind != GateKind::FaithfulTR &&
      params.kind != GateKind::FaithfulARS &&
      params.kind != GateKind::GapARS) {
    throw std::runtime_error("lower_truncation_gate: GateKind must be TR/ARS/GapARS");
  }
  TruncationLoweringResult res;
  res.keys = gates::composite_gen_trunc_gate(backend, rng, params.frac_bits, params.kind, batch_N, &res.suf);

  // Build postproc hooks wired with layout + masks.
  if (params.kind == GateKind::FaithfulTR) {
    auto h0 = std::make_unique<gates::FaithfulTruncPostProc>();
    auto h1 = std::make_unique<gates::FaithfulTruncPostProc>();
    h0->f = h1->f = params.frac_bits;
    h0->r_hi_share = res.keys.k0.r_hi_share;
    h1->r_hi_share = res.keys.k1.r_hi_share;
    h0->r_in = res.keys.k0.compiled.r_in;
    h1->r_in = res.keys.k1.compiled.r_in;
    res.hook0 = std::move(h0);
    res.hook1 = std::move(h1);
  } else if (params.kind == GateKind::FaithfulARS || params.kind == GateKind::GapARS) {
    auto h0 = std::make_unique<gates::FaithfulArsPostProc>();
    auto h1 = std::make_unique<gates::FaithfulArsPostProc>();
    h0->f = h1->f = params.frac_bits;
    h0->r_hi_share = res.keys.k0.r_hi_share;
    h1->r_hi_share = res.keys.k1.r_hi_share;
    h0->r_in = res.keys.k0.compiled.r_in;
    h1->r_in = res.keys.k1.compiled.r_in;
    res.hook0 = std::move(h0);
    res.hook1 = std::move(h1);
  }
  return res;
}

}  // namespace compiler
