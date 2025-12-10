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
- ✅ Overlap hooks: LayerContext exposes PFSS compute stream; matmul uses a dedicated non-blocking stream (separate from PFSS) for overlap with adaptive 32/64 tiling (env `SUF_MATMUL_GPU_TILE=wide|narrow` to force); `bench_gemm_overlap` exercises PFSS packed compare + GEMM overlap (CUDA-only) and reports per-stream timings. PFSS kernels allow block-size tuning via `SUF_PFSS_GPU_BLOCK`.
- ✅ eff_bits-aware CUDA pack/unpack (bitpacked H2D with device unpack) and ragged packing test (`test_cuda_pack_effbits`); backend auto-packs when `in_bits<64`.
- ✅ Planner/runtime wiring: env-driven GPU selection in attention/MLP; PFSS batches adopt GPU stager and device-byte budgets when GPU backend selected; CUDA stager (`CudaPfssStager`) available.
- ✅ Softmax GPU smoke added (`test_softmax_gpu_smoke`) alongside GapARS/Faithful trunc CUDA equivalence.
- ✅ GEMM kernel tuned (BK=32, vectorized loads) and overlap benchmark now runs PFSS + GEMM on separate streams/threads (`bench_gemm_overlap`).
- ✅ GapARS selector respects per-tensor mask bounds (`mask_abs`) when deciding GapARS vs faithful ARS.

## Remaining tasks (detailed)

### 1) Perf polish
- Benchmark PFSS kernels and end-to-end overlap (attention/MLP), tuning block sizes/SoA layout and exploring WMMA/TC paths now that GEMM/PFSS streams are decoupled with timing in `bench_gemm_overlap`.

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



• What I changed

- Added an isolated GapARS truncation vs reference check and an nExp composite sanity check inside src/demo/test_softmax_gpu_smoke.cpp, using paired ProtoLocalChan instances so we can compare GPU PFSS bundles directly against the reference backend before running the softmax. This also logs
  progress ([dbg] …) to pinpoint where things diverge.
- Adjusted the truncation checker to use separate backends for each party (GPU0/GPU1 or ref/ref) to avoid shared-state interference.

Findings so far

- The softmax smoke now hangs in the new GapARS truncation check: it prints [dbg] Checking GapARS trunc bundle GPU vs ref... and then blocks. That means the GPU truncation composite path (GapARS bundle via composite_eval_batch_with_postproc with the CUDA backend) is not returning.
- Previously, the softmax test completed and showed value mismatches; with the new instrumentation, the blocking indicates the truncation path itself is problematic before we even reach the softmax pipeline.
- The nExp composite check hasn’t been reached yet (it’s after the trunc check), so we still need to confirm exp coefficients in this run.


Next steps I’d take

1. Debug the GapARS trunc composite with the CUDA backend: add logging around composite_eval_batch_with_postproc for GapARS (or temporarily switch to the reference backend there) to see if the hang is in PFSS eval vs the postproc hook. You can also run with a shorter timeout and
   CUDA_LAUNCH_BLOCKING=1 to see if a kernel launch is stuck.
2. If the hang is due to the PFSS GPU backend in multi-threaded use, try serializing the two parties or using a dedicated backend per party (already done) and check the CUDA streams/events in pfss_backend_gpu for deadlocks.
3. Once the trunc check returns, run the nExp composite check and the softmax mismatch again to localize any remaining numeric divergence (exp/sum/inv).

I didn’t rerun the full suite because the smoke test blocks at the new truncation check.

• The softmax smoke was hanging because the GapARS truncation check was running both parties sequentially on a single thread (deadlocked on the channel). I fixed eval_trunc in src/demo/test_softmax_gpu_smoke.cpp to spawn a thread for party1 and synchronize like the nExp composite path. The
test now progresses past the truncation check (with debug logs), so you can continue instrumenting the remaining nExp/softmax stages.

What changed:

- src/demo/test_softmax_gpu_smoke.cpp: eval_trunc now runs party0/party1 concurrently with proper exception handling; added dbg logs remain.

Next steps:

1. Re-run ./build/test_softmax_gpu_smoke (optionally with CUDA_LAUNCH_BLOCKING=1) to see if we hit the original numeric mismatch and use the stage-by-stage dumps already in the test.
2. If divergence persists, focus on the post-trunc/Horner path and GapARS trunc outputs versus reference.
