#pragma once

#include <memory>
#include <random>
#include <limits>

#include "compiler/pfss_program_desc.hpp"
#include "compiler/range_analysis.hpp"
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
  struct PerElement {
    suf::SUF<uint64_t> suf;
    gates::CompositeKeyPair keys;
    std::unique_ptr<gates::PostProcHook> hook0;
    std::unique_ptr<gates::PostProcHook> hook1;
  };
  std::vector<PerElement> per_elems;  // optional per-element compiled keys/hooks
};

// Lower a rescale/trunc/ARS GateKind into a new + composite keys + postproc hooks.
inline TruncationLoweringResult lower_truncation_gate(proto::PfssBackend& backend,
                                                      std::mt19937_64& rng,
                                                      const GateParams& params,
                                                      size_t batch_N = 1) {
  AbsBound abs = params.abs_hint;
  // If no explicit abs_hint was provided, derive a conservative hint from range.
  if (abs.max_abs == std::numeric_limits<uint64_t>::max() &&
      params.range_hint.lo != std::numeric_limits<int64_t>::min() &&
      params.range_hint.hi != std::numeric_limits<int64_t>::max()) {
    abs = abs_from_range(params.range_hint, params.range_hint.is_signed);
  }
  GateKind kind = params.kind;
  if (kind == GateKind::AutoTrunc) {
    std::optional<GapCert> gap = params.gap_hint;
    if (gap && gap->mask_abs == std::numeric_limits<uint64_t>::max()) {
      gap->mask_abs = default_mask_bound(params.frac_bits);
    }
    kind = select_trunc_kind(abs, params.frac_bits, gap);
  }
  if (kind != GateKind::FaithfulTR &&
      kind != GateKind::FaithfulARS &&
      kind != GateKind::GapARS) {
    throw std::runtime_error("lower_truncation_gate: GateKind must be TR/ARS/GapARS/AutoTrunc");
  }
  TruncationLoweringResult res;
  // Legacy single-mask path.
  res.keys = gates::composite_gen_trunc_gate(backend, rng, params.frac_bits, kind, batch_N, &res.suf);

  // Build postproc hooks wired with layout + masks.
  if (kind == GateKind::FaithfulTR) {
    auto h0 = std::make_unique<gates::FaithfulTruncPostProc>();
    auto h1 = std::make_unique<gates::FaithfulTruncPostProc>();
    h0->f = h1->f = params.frac_bits;
    h0->r_hi_share = res.keys.k0.r_hi_share;
    h1->r_hi_share = res.keys.k1.r_hi_share;
    h0->r_in = res.keys.k0.compiled.r_in;
    h1->r_in = res.keys.k1.compiled.r_in;
    res.hook0 = std::move(h0);
    res.hook1 = std::move(h1);
  } else if (kind == GateKind::FaithfulARS || kind == GateKind::GapARS) {
    if (kind == GateKind::GapARS) {
      auto h0_gap = std::make_unique<gates::GapArsPostProc>();
      auto h1_gap = std::make_unique<gates::GapArsPostProc>();
      h0_gap->f = h1_gap->f = params.frac_bits;
      h0_gap->r_hi_share = res.keys.k0.r_hi_share;
      h1_gap->r_hi_share = res.keys.k1.r_hi_share;
      h0_gap->r_in = res.keys.k0.compiled.r_in;
      h1_gap->r_in = res.keys.k1.compiled.r_in;
      res.hook0 = std::move(h0_gap);
      res.hook1 = std::move(h1_gap);
    } else {
      auto h0_f = std::make_unique<gates::FaithfulArsPostProc>();
      auto h1_f = std::make_unique<gates::FaithfulArsPostProc>();
      h0_f->f = h1_f->f = params.frac_bits;
      h0_f->r_hi_share = res.keys.k0.r_hi_share;
      h1_f->r_hi_share = res.keys.k1.r_hi_share;
      h0_f->r_in = res.keys.k0.compiled.r_in;
      h1_f->r_in = res.keys.k1.compiled.r_in;
      res.hook0 = std::move(h0_f);
      res.hook1 = std::move(h1_f);
    }
  }
  // Optional per-element path: generate distinct masks/compiled keys per element.
  if (params.per_element_masks && batch_N > 1) {
    res.per_elems.resize(batch_N);
    for (size_t i = 0; i < batch_N; ++i) {
      auto& pe = res.per_elems[i];
      pe.keys = gates::composite_gen_trunc_gate(backend, rng, params.frac_bits, kind, 1, &pe.suf);
      if (kind == GateKind::FaithfulTR) {
        auto h0 = std::make_unique<gates::FaithfulTruncPostProc>();
        auto h1 = std::make_unique<gates::FaithfulTruncPostProc>();
        h0->f = h1->f = params.frac_bits;
        h0->r_hi_share = pe.keys.k0.r_hi_share;
        h1->r_hi_share = pe.keys.k1.r_hi_share;
        h0->r_in = pe.keys.k0.compiled.r_in;
        h1->r_in = pe.keys.k1.compiled.r_in;
        pe.hook0 = std::move(h0);
        pe.hook1 = std::move(h1);
      } else {
        auto h0 = std::make_unique<gates::FaithfulArsPostProc>();
        auto h1 = std::make_unique<gates::FaithfulArsPostProc>();
        h0->f = h1->f = params.frac_bits;
        h0->r_hi_share = pe.keys.k0.r_hi_share;
        h1->r_hi_share = pe.keys.k1.r_hi_share;
        h0->r_in = pe.keys.k0.compiled.r_in;
        h1->r_in = pe.keys.k1.compiled.r_in;
        pe.hook0 = std::move(h0);
        pe.hook1 = std::move(h1);
      }
    }
  }
  return res;
}

}  // namespace compiler
