# SUF Milestone Task Audit (milestone1–8, milestone11_gpu, revise_m11, super)

This document summarizes which tasks in:

- `milestone1.md` … `milestone8.md`
- `milestone11_gpu.md`
- `revise_m11.md`
- `super.md`

are **completed** vs **incomplete** in the current codebase, and provides **detailed, repo-specific guidance** for finishing incomplete items.

Audit basis:

- Code + build targets in this repo (`include/`, `src/`, `CMakeLists.txt`)
- Test/bench executables under `src/demo/` and `src/bench/`
- Local verification commands listed below

> Notes
>
> - Some milestone docs are “design plans” rather than strict acceptance specs; for those, this audit treats **explicit Deliverables/DoD** and **clearly actionable bullets** as tasks, and labels long-horizon “performance polish / future optimization” as `Partial` by default.
> - GPU tasks are marked `Partial` unless they can be validated in a CUDA build with a real device.

## Quick Verification Commands

CPU build + tests:

```bash
ninja -C build_ninja
ctest --test-dir build_ninja --output-on-failure
```

Key benchmarks (CPU path):

```bash
ninja -C build_ninja bench_layer_breakdown
./build_ninja/bench_layer_breakdown --model bert-base --seq-len 4 --batch-size 1 --n-iters 1
```

CUDA notes:

- CUDA tests/benches auto-skip when `cudaGetDeviceCount()==0` (no device).
- Useful GPU smoke:
  - `ctest --test-dir build_ninja -R test_pfss_gpu --output-on-failure`
  - `ctest --test-dir build_ninja -R test_cuda_packed_pfss --output-on-failure`

---

## Status Legend

- `Done`: implemented + exercised by tests/bench (or strongly covered indirectly)
- `Partial`: implemented but missing a required optimization path, proof tightening, or hard acceptance criterion
- `TODO`: not implemented

---

## `milestone1.md` — Canonical runtime primitives + Tape determinism

### Step 1.1 — Canonical core/proto boundary

- `Done`: canonical pack/unpack + ring ops centralized in `include/core/serialization.hpp` and re-exported via `include/proto/common.hpp`.
  - Verify: `rg "pack_u64_le|unpack_u64_le" include/core include/proto`

### Step 1.2 — Batched Beaver multiplication

- `Done`: `proto::BeaverMul64::mul_batch` + wrapper `proto::BeaverMul64Batch` in `include/proto/beaver_mul64.hpp`.
  - Verified in: `src/demo/test_cuda_beaver_mul.cpp` (CUDA path) + extensive usage across tasks.

### Step 1.3 — Standardize bit operations on shares

- `Done`: additive-bit contract + `BitRingOps` (`NOT/AND/XOR/OR/SEL`) and `lut8_select` in `include/proto/bit_ring_ops.hpp`.
  - Verified in: `src/demo/test_postproc_hooks.cpp` (SEL + LUT usage in postproc).

### Step 1.4 — Canonical batching utilities / packing

- `Done`: key packing helpers exist (`include/proto/pack_utils.hpp`) and are exercised via:
  - `src/demo/test_myl7_bit_order.cpp`
  - `src/demo/test_pred_semantics.cpp`
  - composite equivalence tests (see Milestone 8 section below)

### Step 2.1 — Tape module

- `Done`: `include/proto/tape.hpp` implements `TapeWriter/TapeReader` with vector + file backends and a tagged record format.
  - Verified in: `src/demo/sim_harness.cpp` tape roundtrip self-test.

### Step 2.2 — Deterministic “consumption plans”

- `Done`: tape layout is explicitly documented in proto dealers/evaluators (e.g., `include/proto/reluars_dealer.hpp`, `include/proto/gelu_spline_dealer.hpp`) and consumed via `TapeReader`.

### Step 2.3 — Dealers write tapes

- `Done`: proto dealer headers include tape write paths:
  - `include/proto/reluars_dealer.hpp`
  - `include/proto/gelu_spline_dealer.hpp`

### Step 2.4 — Online evaluators can consume tapes

- `Done`: tape helpers exist for online eval, e.g.:
  - `include/proto/reluars_online_complete.hpp`
  - `include/proto/gelu_online_step_dcf.hpp`

### Step 2.5 — Harness uses true plaintext references

- `Done`: `src/demo/sim_harness.cpp` includes plaintext reference checks and tape replay modes.

---

## `milestone2.md` — Proto common+tape + deterministic consumption (and SUF/MASK follow-ons)

`milestone2.md` contains (a) drop-in suggestions for `proto/common.hpp` + `proto/tape.hpp` + dealer/online tape wiring, and (b) “next milestone” guidance for SUF + mask rewrite.

### Proto/common + tape deliverables

- `Done`: `include/proto/common.hpp` exists and uses canonical `core/*` helpers (endian-stable pack/unpack + ring ops).
  - Verify: `rg "include/core/serialization.hpp|pack_u64_le|unpack_u64_le" include/proto/common.hpp include/core`
- `Done`: `include/proto/tape.hpp` exists (tagged record format, vector + file backends) and is used by dealers/online code with deterministic ordering.
  - Verify: `rg "class TapeWriter|class TapeReader" include/proto/tape.hpp`
  - Evidence of real usage: `include/proto/reluars_dealer.hpp`, `include/proto/gelu_spline_dealer.hpp`, `src/demo/sim_harness.cpp`

### Milestone 3 deliverables (SUF reference semantics)

- `Done`: `include/suf/validate.hpp`
- `Done`: `include/suf/ref_eval.hpp`
- `Done`: `src/demo/test_suf_ref_eval.cpp` (exhaustive/random validation)

### Milestone 4 deliverables (mask rewrite engine)

- `Done`: `include/suf/mask_rewrite.hpp`
- `Done`: `include/suf/mask_rewrite_eval.hpp`
- `Done`: `src/demo/test_mask_rewrite.cpp` (property tests for rewrite correctness)
- `Done`: compiler integration in `src/compiler/suf_to_pfss.cpp`

---

## `milestone3.md` — Wrap bit must not be public + SUF hardening

- `Done`: “wrap flag not public” fix is implemented using **additive shared wrap bits** (`wrap_sign_share`) and MPC selection (no shared public wrap boolean).
  - Evidence:
    - `include/proto/reluars_dealer.hpp` / `include/proto/reluars_online_complete.hpp`
    - `include/proto/gelu_spline_dealer.hpp` / `include/proto/gelu_online_step_dcf.hpp`
    - `src/demo/test_postproc_hooks.cpp`

---

## `milestone4.md` — Milestone 5/6: SUF→PFSS compiler + baseline backends

### Milestone 5 “compiler correctness tests”

- `Done`: SUF→PFSS compilation exists (`include/compiler/suf_to_pfss.hpp`, `src/compiler/suf_to_pfss.cpp`) with two-program output.
- `Done`: compile correctness test exists as `src/demo/test_compile_pfss.cpp`.

### Milestone 6 backend interface + adapters

- `Done`: PFSS batch interface + clear backend:
  - `include/proto/pfss_backend_batch.hpp`, `include/proto/backend_clear.hpp`
- `Done`: myl7 adapter:
  - `include/proto/myl7_fss_backend.hpp`
- `Done`: SigmaFast backend (CPU packed path):
  - `include/proto/sigma_fast_backend_ext.hpp`

Key backend tests:

- `Done`: `src/demo/test_pred_semantics.cpp`
- `Done`: `src/demo/test_myl7_bit_order.cpp`
- `Done`: `src/demo/test_sigmafast.cpp`, `src/demo/test_sigmafast_packed_equiv.cpp`

---

## `milestone5.md` — Milestone 7/8: SigmaFast + Composite runtime

### Milestone 7 (SigmaFastBackend)

- `Done`: packed predicates (`PredOutMode::kPackedMask_Xor`) support is in `include/compiler/pfss_program_desc.hpp` + backend hooks.
- `Done`: packed pred/interval LUT paths + equivalence tests:
  - `src/demo/test_sigmafast_packed_equiv.cpp`
  - `src/demo/test_composite_equiv_proto_sigmafast.cpp`
- `Done`: benches:
  - `src/bench/bench_sigmafast_pred.cpp`
  - `src/bench/bench_sigmafast_coeff.cpp`
  - `src/bench/bench_sigmafast_gates.cpp`

### Milestone 8 (Composite runtime)

- `Done`: generic composite runtime + tape I/O:
  - `include/gates/composite_fss.hpp`
- `Done`: postproc hooks:
  - `include/gates/postproc_hooks.hpp`
- `Done`: equivalence tests (proto vs composite under multiple backends):
  - `src/demo/test_composite_equiv_proto.cpp`
  - `src/demo/test_composite_equiv_proto_myl7.cpp`
  - `src/demo/test_composite_equiv_proto_sigmafast.cpp`
  - `src/demo/test_composite_runtime.cpp`

---

## `milestone6.md` — Milestone 9: LLM gates + blocks

- `Done`: reusable piecewise poly scaffold + tables:
  - `include/gates/piecewise_poly.hpp`, `include/gates/tables/*`
- `Done`: scalar gates:
  - `include/gates/silu_spline_gate.hpp`
  - `include/gates/nexp_gate.hpp`
  - `include/gates/reciprocal_gate.hpp`
  - `include/gates/rsqrt_gate.hpp`
- `Done`: blocks:
  - `include/gates/softmax_block.hpp`
  - `include/gates/layernorm_block.hpp`
- `Done`: taskified versions exist (runtime scheduling + batching):
  - `include/nn/softmax_block_task.hpp`
  - `include/runtime/phase_tasks.hpp` (CubicPolyTask/RecipTask/RsqrtTask/LayerNormTask)
- `Done`: correctness tests:
  - `src/demo/test_llm_gates.cpp`
  - `src/demo/test_softmax_task.cpp`
  - `src/demo/test_layernorm_task.cpp`
- `Done`: benchmarks:
  - `src/bench/bench_llm_gates.cpp`
  - `src/bench/bench_softmax_norm.cpp`

---

## `milestone7.md` — Milestone 10: Matmul + Attention + Transformer layer

- `Done`: tensor views + linops:
  - `include/nn/tensor_view.hpp`, `include/nn/linops.hpp`, `src/nn/linops.cpp`
- `Done`: matmul public weights:
  - `include/nn/matmul_publicW.hpp`, `src/nn/matmul_publicW.cpp`
- `Done`: matmul beaver + truncation integration:
  - `include/nn/matmul_beaver.hpp`, `src/nn/matmul_beaver.cpp`
- `Done`: attention + KV cache + transformer layer:
  - `include/nn/attention_block.hpp`, `src/nn/attention_block.cpp`
  - `include/nn/kv_cache.hpp`
  - `include/nn/transformer_layer.hpp`, `src/nn/transformer_layer.cpp`
- `Done`: end-to-end executor tests:
  - `src/demo/test_matmul_and_attention_executor.cpp`
  - `src/demo/test_matmul_executor.cpp`

---

## `milestone8.md` — Milestone 11 (revised): scheduling + batching + faithful trunc/ARS + GPU overlap

### 11.1 Faithful truncation / ARS as Composite gates

- `Done`: gate kinds exist and are wired through composite:
  - `include/gates/trunc_faithful_gate.hpp`
  - `include/gates/ars_faithful_gate.hpp`
  - `include/compiler/pfss_program_desc.hpp` (GateKind entries)
  - `include/compiler/truncation_lowering.hpp` (lowering into composite bundles + hooks)
  - `src/demo/test_truncation.cpp`, `src/bench/bench_truncation.cpp`
- `Done`: `GateKind::GapARS` is a real SIGMA-style fast path (fewer bool ports, no full-width wrap compare predicates).
  - Evidence: `include/suf/trunc_suf_builders.hpp` (`build_gapars_suf`), `include/gates/postproc_hooks.hpp` (`GapArsPostProc`), `include/gates/composite_fss.hpp` (`composite_gen_trunc_gate` layout)
  - Verified in: `src/demo/test_gapars_fastpath.cpp`, `src/demo/test_cuda_trunc_gapars.cpp`

### 11.2 Range / eff_bits / GapCert passes (automatic selection)

- `Done`: range + mask-bound propagation and GapCert-driven `AutoTrunc` selection are implemented and exercised.
  - Range/GapCert: `include/compiler/range_analysis.hpp`, `src/compiler/layer_graph.cpp`
  - Integration: `include/nn/layer_context.hpp` (`finalize_layer`, `record_rescale`)
  - Verified in: `src/demo/test_gapars_selector.cpp`, `src/demo/test_mask_abs_propagation.cpp`, `src/demo/test_range_proofs.cpp`

### 11.3 PFSS program merging / layer-level batching

- `Done`: PFSS superbatch + planner exist and are used to batch across tasks/phases:
  - `include/runtime/pfss_superbatch.hpp`, `src/runtime/pfss_superbatch.cpp`
  - `include/runtime/phase_executor.hpp`
  - `include/runtime/pfss_phase_planner.hpp`
- `Done`: ragged/causal shapes supported via `row_offsets`/`row_lengths` (reduces unnecessary work):
  - `include/nn/softmax_block_task.hpp`
  - planner regressions: `src/demo/test_planner_causal_bytes.cpp`, `src/demo/test_planner_ragged_bytes.cpp`

### 11.4 Truncation hoisting

- `Done`: conservative rescale hoisting/merging implemented (`LayerGraph::hoist_rescales`) and used by `nn::finalize_layer`.
  - Verified in: `src/demo/test_hoist_rescales.cpp`

### 11.5 Layer-wide Beaver open fusion

- `Done`: `OpenCollector` batches opens and is integrated into tasks:
  - `include/runtime/open_collector.hpp`, `src/runtime/open_collector.cpp`
  - used by matmul + tasks in `include/runtime/phase_tasks.hpp`

### 11.6 CPU PFSS + GPU GEMM overlap

- `Done`: overlap hooks exist (separate streams + staged PFSS) and are exercised by the overlap benchmark.
  - Evidence: `include/runtime/pfss_gpu_staging.hpp`, `src/runtime/pfss_gpu_stager_cuda.cpp`, `src/bench/bench_gemm_overlap.cpp`

### 11.8 eff_bits-aware packing (GPU staging)

- `Done` (GPU staging / PFSS hatx packing): PFSS batch can pack hatx with `eff_bits` and unpack on device.
  - `src/runtime/pfss_superbatch.cpp` (host pack + device unpack)
  - `include/runtime/cuda_primitives.hpp` + `cuda/cuda_primitives.cu`
  - regression: `src/demo/test_planner_effbits_budget.cpp`
- `Done` (network packing): `OpenCollector` supports optional `eff_bits` packing for open traffic (env `SUF_OPEN_PACK_EFFBITS=1`).
  - Implementation: `src/runtime/open_collector.cpp`
  - Regression: `src/demo/test_open_collector_packing.cpp`

### 11.7 GPU linear core (cuBLASLt/CUTLASS + epilogue fusion)

- `Done`: CUDA matmul for public weights exists with bias fusion + caching and tiling knobs.
  - Evidence: `include/nn/matmul_gpu.hpp`, `src/nn/matmul_gpu.cu` (env `SUF_MATMUL_GPU_TILE`, cache flags)

### 11.9 “Only compute necessary elements” (ragged/causal attention work)

- `Done` (planner + softmax task shape support): causal/ragged shapes are represented as row-offset/row-length metadata and used by the planner to reduce PFSS work.
  - Evidence: `include/nn/softmax_block_task.hpp`, `include/runtime/pfss_phase_planner.hpp`
  - Regressions: `src/demo/test_planner_causal_bytes.cpp`, `src/demo/test_planner_ragged_bytes.cpp`

### 11.10 Bench: layer breakdown

- `Done`: `bench_layer_breakdown` exists and runs end-to-end:
  - `src/bench/bench_layer_breakdown.cpp`
  - build target: `bench_layer_breakdown`

---

## `milestone11_gpu.md` — GPU fast path status

- `Done`: GPU backend, staged PFSS, CUDA postproc kernels, and overlap hooks are present; remaining work is perf tuning.
  - Evidence listed in that doc aligns with code in:
    - `include/proto/backend_gpu.hpp`
    - `include/runtime/pfss_gpu_staging.hpp`, `src/runtime/pfss_gpu_stager_cuda.cpp`
    - `include/runtime/cuda_primitives.hpp`
    - CUDA tests: `src/demo/test_pfss_gpu.cpp`, `src/demo/test_cuda_packed_pfss.cpp`, `src/demo/test_cuda_trunc_gapars.cpp`, etc.

---

## `revise_m11.md` — Consolidated remaining work

These items are implemented in the current codebase:

1. **Cross-phase PFSS “super-plan” + finer barriers** (`Done`): `runtime::PfssLayerPlanner` barriers in `src/nn/transformer_layer.cpp`
2. **GapCert tightening / mask-bound use** (`Done`): `mask_abs` + `GapCert` propagation in `src/compiler/layer_graph.cpp`
3. **Hoist/rescale** (`Done`): `LayerGraph::hoist_rescales` + range re-propagation in `include/nn/layer_context.hpp`
4. **Packing/flush budgets + regressions** (`Done`): planner limits + `src/demo/test_pfss_superbatch_limits.cpp`, `src/demo/test_planner_effbits_budget.cpp`

See “How to Finish Incomplete Tasks” below for detailed completion steps.

---

## `super.md` — Device-resident PFSS outputs + GPU postproc kernels

This doc describes a GPU end-to-end device pipeline. Current status in code:

- `Done/Partial` (device views exist):
  - `runtime::PfssResultView` contains device pointers via `arith_device/bools_device` in `include/runtime/pfss_superbatch.hpp`.
  - `PfssSuperBatch` supports `device_outputs` mode and stages hatx to device.
- `Done` (CUDA micro-kernels exist and are tested):
  - `include/runtime/cuda_primitives.hpp` wrappers + CUDA tests:
    - `src/demo/test_cuda_trunc_postproc.cpp`
    - `src/demo/test_cuda_horner_cubic.cpp`
    - `src/demo/test_cuda_beaver_mul.cpp`
    - `src/demo/test_cuda_recip_task.cpp`
- `Done` (device-only end-to-end softmax):
  - `src/demo/test_softmax_gpu_smoke.cpp` validates a device-only softmax pipeline (PFSS outputs + trunc postproc kept on device) matches the CPU reference.

---

# How to Finish Incomplete Tasks (Detailed)

This section is intentionally detailed and repo-specific. It is ordered by “unblocks the most acceptance criteria first”.

## A) Implement real GapARS (currently placeholder)

### Goal

Make `GateKind::GapARS` materially cheaper than faithful ARS when a proof-grade `GapCert` holds, and ensure the compiler/runtime selects it automatically and safely.

### Current state

- GapARS is wired end-to-end, but semantics are currently identical to faithful ARS:
  - `include/gates/gapars_gate.hpp`
  - `include/gates/postproc_hooks.hpp` (`GapArsPostProc` matches faithful path)

### Concrete completion steps

1. **Define a concrete GapARS protocol variant for this repo**
   - Choose the exact “gap condition” and the minimal predicate set required.
   - Keep the interface identical (same `TruncationLoweringResult` surface), so tasks do not change.

2. **Extend `suf::build_gapars_suf` to emit the right primitive predicates**
   - File: `include/suf/trunc_suf_builders.hpp`
   - Today it emits the same predicate set as ARS (`carry` + `sign`).
   - Update it to emit the predicates you actually need for GapARS (e.g., omit sign-extension compare if certificate guarantees it, or replace with cheaper low-bit predicates).

3. **Teach the compiler to pick GapARS only with `RangeKind::Proof`**
   - File: `include/compiler/range_analysis.hpp` (already has `can_gapars`)
   - Ensure selection requires:
     - `gap.kind == Proof`
     - mask bound (`mask_abs`) is not “unknown” and is propagated (see section B).

4. **Implement a real `GapArsPostProc`**
   - File: `include/gates/postproc_hooks.hpp`
   - Make it do fewer operations than `FaithfulArsPostProc` and/or depend on a smaller PFSS output surface.
   - Add a micro-test that checks:
     - reconstructed output matches faithful ARS output for values satisfying the certificate
     - it refuses/does not get selected when the certificate does not hold.

5. **Update `bench_truncation` to report “protocol cost”**
   - File: `src/bench/bench_truncation.cpp`
   - Add counters for:
     - number of PFSS pred bits and coeff words
     - number of Beaver opens used by postproc (if any)
   - Use this to demonstrate GapARS is cheaper when enabled.

6. **Add/extend a “GapARS selection” regression**
   - Existing: `src/demo/test_gapars_selector.cpp`
   - Add a test that drives a full composite truncation job and asserts:
     - `compiled.gate_kind == GapARS` when proofs hold
     - otherwise `FaithfulARS` (or `FaithfulTR` for unsigned).

## B) Tighten GapCert proofs + propagate mask bounds more aggressively

### Goal

Make `AutoTrunc` choose GapARS in more places safely, and allow more hoists with proof guards.

### Concrete steps

1. **Make mask bounds first-class in graph metadata**
   - Files:
     - `include/compiler/pfss_program_desc.hpp` (`GapCert{mask_abs}`)
     - `include/nn/layer_context.hpp` / `src/nn/transformer_layer.cpp`
   - Ensure every tensor carries a non-default `mask_abs` whenever a mask is introduced (matmul outputs, rescale sites).

2. **Propagate `mask_abs` through ops in `range_propagation`**
   - File: `include/compiler/range_propagation.hpp`
   - When you combine masked values (add/sub/axpy), update mask bounds conservatively:
     - `mask_abs(add) = mask_abs(a)+mask_abs(b)` (cap at 2^64-1)
     - `mask_abs(mul_const) = |c|*mask_abs(a)`

3. **Upgrade key proof sites from Hint→Proof**
   - File: `src/compiler/layer_graph.cpp`
   - Identify the “known difficulties” in `revise_m11.md` (activations/LN affine) and:
     - compute worst-case bounds from table specs (SiLU/nExp clamps)
     - use deterministic integer arithmetic (ceil-div) for sums/means

4. **Add regressions that assert proof-grade bounds are produced**
   - Add a `src/demo/test_range_proofs.cpp` (or extend existing tests) that:
     - builds a small layer graph (LN→Attn→MLP) in code
     - runs the range pass
     - asserts key nodes have `RangeKind::Proof` and finite `mask_abs`

## C) Cross-phase “super-plan” and finer dependency barriers

### Goal

Reduce flush count and enable deeper overlap by keeping PFSS/Open batches alive across multiple sub-steps, but still avoid deadlocks.

### Current state

- `PfssLayerPlanner` exists and supports explicit barriers:
  - `include/runtime/pfss_phase_planner.hpp`
- Some code paths still drain coarsely (`drain_all`) at phase boundaries:
  - `src/nn/transformer_layer.cpp`

### Concrete steps

1. **Define explicit dependency points in attention + MLP**
   - Files:
     - `src/nn/attention_block.cpp`
     - `src/nn/mlp_block.cpp`
   - Replace coarse `drain_all` barriers with:
     - “drain only open” when you just need Beaver opens to proceed
     - “drain PFSS coeff only” when a gate needs coefficients but not trunc outputs, etc.

2. **Make stall-driven flush the default when `keep_batches=true`**
   - File: `include/runtime/phase_executor.hpp`
   - Ensure tasks request flushes only when they truly need results (`Need::{Open,PfssCoeff,PfssTrunc}`).
   - Avoid unconditional flushes at phase end unless a barrier policy demands it.

3. **Add a deadlock regression**
   - Extend `src/demo/test_pfss_async_stress.cpp` or add a new test:
     - run a layer with `keep_batches=true` and inner barriers disabled
     - assert it completes and `PhaseExecutor` does not throw the “deadlock” error.

4. **Add statistics to validate the improvement**
   - Use existing totals:
     - `PfssLayerPlanner::Totals` (flush count, active elems, cost_effbits, flush_ns)
   - Wire those into `bench_layer_breakdown` output to show:
     - fewer flushes after barrier tightening.

## D) Extend hoist/rescale safely across more chains

### Goal

Reduce the number of truncation gates without overflow/precision regressions, guarded by proof-grade bounds.

### Concrete steps

1. **Add “proof-required” hoists for activation chains**
   - File: `src/compiler/layer_graph.cpp` (`hoist_rescales`)
   - Add patterns:
     - Rescale after `BiasAdd` → hoist over bias if bias can be safely upscaled
     - Rescale around `SiLU`/`GELU` when the table spec clamps guarantee no overflow

2. **Enforce “no mixed-sign hoist” unless certified**
   - Require:
     - either `GapCert` proof or “unsigned-only” path
   - Reject hoist if sign uncertainty could make ARS vs TR semantics change.

3. **Regression tests**
   - Add/extend a test that:
     - builds a graph with back-to-back rescales and activations
     - checks hoisted graph produces identical outputs (bit-exact) when using the same truncation semantics,
       and within tolerance when using the optimized placement mode.

## E) GPU device-only pipeline (super.md)

### Goal

Keep PFSS outputs and postproc on device across softmax/LN/layer to reduce H2D/D2H and CPU bottlenecks.

### Concrete steps

1. **Standardize “device output ownership”**
   - Files:
     - `include/runtime/pfss_superbatch.hpp`
     - `src/runtime/pfss_superbatch.cpp`
   - Ensure:
     - `device_outputs` implies the backend returns or stages `arith_device/bools_device`
     - tasks can “take ownership” (or borrow) without double-free.

2. **Make task outputs optionally device-backed**
   - Files:
     - `include/runtime/phase_tasks.hpp` (`TruncTask`, `CubicPolyTask`, `RecipTask`, `MulRowBroadcastTask`, `LayerNormTask`)
   - Prefer a consistent pattern:
     - if `R.device_pipeline==true`, keep device output and skip host materialization
     - otherwise copy back and keep existing CPU semantics.

3. **End-to-end “device-only softmax”**
   - Start from `SoftmaxBlockTask` (already has `device_only` and GPU row-sum paths).
   - Ensure:
     - `nExp` coeff PFSS outputs stay device-side (avoid host copy)
     - row-broadcast mul stays device-side
     - final probability truncation stays device-side

4. **End-to-end “device-only LN”**
   - Use existing CUDA row kernels:
     - `launch_row_mean_kernel`, `launch_row_variance_kernel`
   - Keep mean/var and rsqrt updates on device.

5. **Add one integration test**
   - Extend `src/demo/test_softmax_gpu_smoke.cpp` to run:
     - CPU pipeline vs GPU device-only pipeline for a small shape
     - compare reconstructed outputs.

## F) Performance polish (milestone11_gpu.md)

The remaining GPU milestone item is intentionally open-ended. A practical way to close it:

1. Add a small “tuning harness”:
   - extend `bench_pfss_cpu_gpu.cpp` and `bench_gemm_overlap.cpp` with:
     - sweep block sizes (`SUF_PFSS_GPU_BLOCK`)
     - sweep GEMM tile (`SUF_MATMUL_GPU_TILE`)
2. Add a target acceptance output in `bench.md`:
   - record at least one representative config (B,T,D,H) and report:
     - PFSS time, open time, GEMM time, overlap %, total latency.
