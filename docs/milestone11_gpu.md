# Milestone 11 — GPU PFSS backend plan (Sigma-style overlap)

Goal: add a CUDA PFSS backend (Sigma-style: GPU PRG/DPF, packed pred/coeff eval, overlap with GEMMs) while keeping CPU as the default and tests green. This document now tracks what is done vs. not done, the missing pieces, and a finer-grained task list.

## Scope and guardrails
- CPU remains default; GPU is opt-in via `SUF_ENABLE_CUDA` + `SUF_PFSS_BACKEND=gpu` (env).
- Start with a GPU stub (ClearBackend) so CI stays green; incrementally drop in real kernels.
- Preserve existing PFSS batching/planner APIs; add backend selection and device staging behind them.

## Current progress (updated)
- ✅ Backend selection factory (`proto/backend_factory.hpp`) with `cpu/gpu/auto` + env `SUF_PFSS_BACKEND`; CPU default intact.
- ✅ LayerContext supports backend override/ownership/env selection; PFSS batches track device-byte budgets and can stage hatx to GPU.
- ✅ GPU stager interface (`PfssGpuStager`) plumbed into `PfssSuperBatch`; pending device bytes counted.
- ✅ GPU backend: AES-CTR PRG + packed CDPF/vector-DPF kernels (pred masks, interval LUT payloads), device key blobs with round keys; staged eval interface (`PfssGpuStagedEval`) with copy/compute streams/events; compute stream exposed for overlap.
- ✅ Composite path uses packed predicates/cuts on GPU; CUDA-gated tests cover packed compare/LUT and composite truncation vs CPU (`test_pfss_gpu` with `RUN_GPU_COMPOSITE=1`).
- ✅ Overlap hooks: LayerContext exposes PFSS compute stream; matmul params accept an overlap stream; a tiled CUDA matmul (mod 2^64) exists; `bench_gemm_overlap` exercises PFSS packed compare + GEMM overlap (CUDA-only).
- ✅ eff_bits-aware CUDA pack/unpack (bitpacked H2D with device unpack) and ragged packing test (`test_cuda_pack_effbits`); backend auto-packs when `in_bits<64`.
- ✅ Planner/runtime wiring: env-driven GPU selection in attention/MLP; PFSS batches adopt GPU stager and device-byte budgets when GPU backend selected; CUDA stager (`CudaPfssStager`) available.
- ✅ Softmax GPU smoke added (`test_softmax_gpu_smoke`) alongside GapARS/Faithful trunc CUDA equivalence.
- ✅ GEMM kernel tuned (BK=32, vectorized loads) and overlap benchmark now runs PFSS + GEMM on separate streams/threads (`bench_gemm_overlap`).

## Remaining tasks (detailed)

### 1) Overlap/pipeline
- Plug PFSS compute stream into stream-aware GEMM/matmul (tile/WMMA per 32-bit halves if needed), run PFSS and GEMM on separate streams with dependency events, and extend `bench_gemm_overlap` with real timing on hardware.

### 2) Perf polish
- Benchmark PFSS kernels and end-to-end overlap (attention/MLP), tuning block sizes/SoA layout once GEMM overlap is wired.

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
