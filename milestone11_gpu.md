Milestone 11 (GPU fast path)
============================

Current progress
----------------
- Backend selection factory (`proto/backend_factory.hpp`) with `cpu/gpu/auto` + env `SUF_PFSS_BACKEND`; CPU stays default.
- LayerContext supports backend override/ownership/env selection; PFSS batches can stage to GPU and track device-byte budgets.
- GPU backend: AES-CTR PRG, packed CDPF for predicates, vector-DPF interval LUT for coeffs, device key blobs with round keys; staged eval interface (`PfssGpuStagedEval`) with copy/compute streams and events, exposing the compute stream for overlap.
- Packed predicate/coeff integration: composite evaluator accepts packed masks on GPU; packed compare/LUT unit tests vs CPU.
- Composite path: GPU packed composite truncation passes equivalence (`test_pfss_gpu` with `RUN_GPU_COMPOSITE=1`).
- Matmul overlap hooks: LayerContext exposes PFSS compute stream; matmul runs on a dedicated non-blocking stream (separate from PFSS) for overlap with adaptive tiling (32/64, env `SUF_MATMUL_GPU_TILE` to force). PFSS kernels allow block-size tuning via `SUF_PFSS_GPU_BLOCK`. `bench_gemm_overlap` exercises PFSS packed compare + GEMM overlap with per-stream timing (CUDA-only).
- eff_bits-aware CUDA pack/unpack (bitpacked H2D with device unpack) and ragged packing test (`test_cuda_pack_effbits`); backend auto-packs when `in_bits<64`.
- Planner/runtime wiring: env-driven GPU selection in attention/MLP; PFSS batches adopt GPU stager and device-byte budgets when GPU backend selected; CUDA stager (`CudaPfssStager`) available.
- Softmax GPU smoke added (`test_softmax_gpu_smoke`) alongside GapARS/Faithful trunc CUDA equivalence.
- GEMM kernel tuned (BK=32 with vectorized tile loads); overlap benchmark now launches PFSS + GEMM on separate streams/threads.
- GapARS selector now respects provided mask bounds when choosing GapARS vs faithful ARS (per-tensor mask_abs fed through LayerContext).

Remaining tasks
---------------
- **Perf polish**: benchmark PFSS kernels and end-to-end overlap (attention/MLP), tune block sizes/SoA layout/WMMA once the GEMM/PFSS overlap path is wired and timed.

How to run
----------
- Build with CUDA available (`nvcc` on PATH; CMake defines `SUF_HAVE_CUDA` automatically).
- Smoke tests:
  - `ctest -R test_cuda_packed_pfss -V`
  - `ctest -R test_pfss_gpu -V` (GPU composite skipped by default)
  - `RUN_GPU_COMPOSITE=1 ctest -R test_pfss_gpu -V` to exercise the packed composite path; add `GPU_COMPOSITE_DEBUG=1` for verbose stage logs.
