# PfssSuperBatch Design Per Gate (plan, current impl untouched)

This folder sketches how to migrate existing SUF gates onto the shared `PfssSuperBatch` prepare/finalize API without breaking the current execution path. Nothing here is wired yet; it is a reference plan for subsequent implementation.

## Common Prepare/Finalize Shape

- **Prepare**: given gate params and masked inputs (`hatx`), build or reuse composite keys, enqueue predicate/coeff eval into `PfssSuperBatch`, and record slices/strides for outputs. No local post-processing beyond enqueuing.
- **Finalize**: after a phase flush, read the batch outputs (arith + bool), apply gate-specific postproc (e.g., Horner already done by composite, optional extra corrections), write masked outputs back to tensor views.
- **Channel/backend**: use `ProtoChanFromNet` adapter so gate code does not depend on `net::Chan` directly.

## Gate Notes

- **SiLU (spline)**  
  - Build SUF from the piecewise spline table (existing `make_silu_spec`).  
  - Composite path already supports polynomial coeffs + selector preds; no postproc hook needed.  
  - **Implemented helper:** `gates::dealer_make_silu_composite_keys` (builds SUF via `build_silu_suf_from_piecewise`) plus `prepare_silu_batch` / `finalize_silu_batch` which use `PfssHandle` + `view()` to read outputs after a phase flush.

- **nExp**  
  - Same as SiLU (piecewise poly).  
  - Ensure domain clamp matches existing gate (input clipped to [0, 16]) in the SUF builder.

- **Reciprocal / Rsqrt**  
  - Use existing affine-init tables (Horner + optional NR iterations).  
  - Two options:  
    1) Expand NR iterations into SUF postproc hook (preferred) so PFSS only covers predicate/selectors.  
    2) Keep NR inside composite poly (simpler, less batching gain).  
  - Prepare: enqueue composite; Finalize: apply NR refinements if done outside composite.

- **Softmax pieces (max, nExp, recip)**  
  - Max predicates can be batched as separate composite jobs producing selector bits.  
  - nExp and recip as above.  
  - Finalize multiplies prob*V via existing matmul/linops.

- **LayerNorm block**  
  - Rsqrt part can be batched; means/variances stay local.  
  - Prepare: enqueue rsqrt composite job for variance.  

## Phase Executor Usage

- Each logical phase (LN1, QKV+score, Softmax, OutProj, LN2+MLP) enqueues gate jobs into `PfssSuperBatch`.  
- At phase barrier, call `flush_and_finalize` once, then run gate-specific finalizers to scatter outputs.

## Key APIs to Add (proposed)

- `prepare_silu_batch(const SiluParams&, span<const uint64_t> hatx, TensorView<uint64_t> out, PfssSuperBatch&)`  
- `prepare_nexp_batch(...)`, `prepare_recip_batch(...)`, `prepare_rsqrt_batch(...)`  
- Optional `struct PreparedGate { size_t job_idx; TensorView<uint64_t> out; }` to track slices.

## Migration Plan

1. Implement SiLU prepare/finalize as template gate.  
2. Extend to nExp and recip/rsqrt (with optional NR in postproc).  
3. Wire phase executor into attention/MLP to flush PFSS once per phase.  
4. Keep legacy CPU path behind a flag for fallback during transition.

This design keeps current gate implementations intact while providing a clear path to move them onto the shared batching surface.
