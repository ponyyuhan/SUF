# Milestone 11 — GPU PFSS backend plan (Sigma-style overlap)

Goal: add a CUDA PFSS backend (Sigma-style: GPU PRG/DPF, packed pred/coeff eval, overlap with GEMMs) while keeping CPU as the default and tests green. This document now tracks what is done vs. not done, the missing pieces, and a finer-grained task list.

## Scope and guardrails
- CPU remains default; GPU is opt-in via `SUF_ENABLE_CUDA` + `SUF_PFSS_BACKEND=gpu` (env).
- Start with a GPU stub (ClearBackend) so CI stays green; incrementally drop in real kernels.
- Preserve existing PFSS batching/planner APIs; add backend selection and device staging behind them.

## Current progress
- ✅ Backend selection factory (`proto/backend_factory.hpp`) with `cpu/gpu/auto` + env `SUF_PFSS_BACKEND`; CPU default intact.
- ✅ LayerContext supports backend override/ownership and env selection helper; PFSS batches track device-byte budgets.
- ✅ GPU stager interface (`PfssGpuStager`) plumbed into `PfssSuperBatch`; pending device bytes counted.
- ✅ GPU backend skeleton (`cuda/pfss_backend_gpu.cu`): derives from `PfssIntervalLutExt`, flattens/upload pred/coeff keys to device (placeholder `cuda_pfss::upload_key`), caches handles; composite eval still uses CPU after staging.
- ✅ CUDA-gated regression `test_pfss_gpu` (built only if `HAVE_SUF_CUDA`) exercises DCF/interval LUT/composite truncation against CPU.
- ❌ Real device-side AES-CTR/DPF + hatx pack/payload accumulation not implemented; CUDA kernels are copy/zero stubs.
- ❌ Planner/runtime not yet choosing GPU backend automatically in main paths; no device budget enforcement beyond SuperBatch; no overlap/pipeline on GPU.
- ❌ No GPU-vs-CPU equivalence ctests once real kernels exist; no GPU packing kernels for eff_bits.

## Remaining tasks (detailed)

### 1) Device key/packer and kernels (blocking)
- Define device-side key structs in `pfss_cuda_api.hpp` (pred step-DCF, coeff step-DCF, interval LUT metadata).
- Implement host packers to flatten keys into these structs and upload with `cuda_pfss::upload_key` (one device blob per pred/coeff program).
- Implement AES-CTR PRG kernel (block counter-based) and DPF traversal for predicates (bits) and coeff (StepDCF delta accumulation / Interval LUT fetch).
- Implement hatx pack kernel (bits from `uint64_t x̂`) to SoA layout expected by composite eval.
- Implement payload accumulation kernels to produce `(haty_share, bool_share)` identical to CPU layout (r words arith per row, ell bools).
- Wire `GpuPfssBackend::eval_composite` to call these kernels end-to-end (no CPU fallback) and reuse cached device keys.

### 2) Planner/runtime wiring (GPU selection + budgets)
- Add env-driven backend selection in layer entry points (attention/MLP/transformer) via `LayerContext::select_backend_from_env()`.
- If GPU backend chosen: require a GPU stager, set `max_pending_device_bytes`, and route `pfss_batch` to the GPU backend.
- Ensure PFSS planner/PhaseExecutor honor device budgets and avoid host-side clear that would drop device buffers.
- Add guardrails: fail closed if GPU selected but stager/channel missing.

### 3) GPU packing/unpacking (eff_bits, causal/ragged)
- Implement CUDA pack/unpack for arbitrary `eff_bits` and causal/ragged softmax packing; integrate with planner byte budgets.
- Add bytes-based regressions for GPU path similar to CPU causal bytes test (gated on CUDA).

### 4) Equivalence tests (CUDA-gated)
- Add ctests (guarded by `HAVE_SUF_CUDA`) for:
  - Pred eval: small set of inputs vs. CPU.
  - StepDCF/Interval LUT coeff eval vs. CPU.
  - Composite truncation (GapARS/faithful) vs. CPU on tiny batches.
- Add a small attention/softmax smoke that runs GPU backend and compares to CPU for a tiny problem size.

### 5) Overlap/pipeline (after correctness)
- Add CUDA streams for PFSS eval; allow async runner to use GPU stream + dedicated channel if safe.
- Bench: layer-level breakdown reporting PFSS GPU time vs. GEMM, bytes, and overlap.

## Final goal
- GPU backend provides end-to-end device-side PFSS (pred/coeff/composite) with AES-CTR/DPF and hatx/payload kernels, selected via env.
- Planner/executor enforces device budgets, packs according to eff_bits, and supports overlap with GEMM.
- CUDA-gated tests/benches show GPU vs. CPU equivalence and basic performance counters.

## Challenges / watch-outs
- Correctly matching CPU composite layout (arith r words, bool ell bits) and GapARS semantics.
- Managing device memory/budgets without leaking or over-staging when super-batching.
- Thread-safety/stream-safety when integrating async PFSS runner and dedicated channels.

## Usage
- CPU (default): nothing to set.
- GPU opt-in (once real kernels land): build with `-DSUF_ENABLE_CUDA=ON`, run with `SUF_PFSS_BACKEND=gpu` (falls back to CPU if CUDA is absent or `allow_gpu_stub=false`).
