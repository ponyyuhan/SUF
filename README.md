# SUF / PFSS Prototype

This repository is a prototype implementation of:

- **SUF** (Structured‑Univariate Functions): a small IR for fixed‑point nonlinearities that exposes both arithmetic outputs and helper predicate bits.
- **PFSS** (Programmable FSS backend): a DPF/DCF-style backend that evaluates predicates and interval/LUT payloads on the **public masked input** \(\hat{x}=x+r_{\text{in}}\).
- A **compiler + runtime** stack that compiles SUF to PFSS programs, executes **Composite‑FSS gates**, and schedules/batches PFSS + Beaver opens across a transformer layer.

## Repository Layout

```
CMakeLists.txt            # build + options (myl7/fss, CUDA)
include/
  core/                   # canonical ring + serialization helpers
  mpc/                    # channel + MPC plumbing
  pfss/                   # PFSS program descriptions + cleartext adapters
  suf/                    # SUF IR + mask rewrite + ref_eval/validate
  compiler/               # SUF→PFSS compiler + range/gap + trunc lowering
  gates/                  # Composite‑FSS gates (Trunc/ARS/SiLU/nExp/Recip/Rsqrt/Softmax/LN…)
  runtime/                # PhaseExecutor, PfssSuperBatch, OpenCollector, planners, async/staged exec
  nn/                     # attention/MLP/transformer layer + LayerContext
  proto/                  # backend abstractions (clear/myl7/sigmafast/gpu), Beaver, tape
cuda/                     # CUDA PFSS + kernels + GPU backend glue
src/
  compiler/               # compiler pass implementations
  runtime/                # runtime implementations (OpenCollector, PFSS batching, GPU stager…)
  nn/                     # NN implementations (attention/mlp/transformer, GPU matmul…)
  demo/                   # demos + tests (unit/regression)
  bench/                  # benchmarks
docs/                     # milestone acceptance + audit docs
bench/                    # Sigma-vs-SUF harness + configs + collected hardware JSON
scripts/                  # helper scripts (hardware probe, sigma build helper)
```

## Build

Prereqs:
- CMake ≥ 3.15, a C++20-capable compiler
- OpenSSL (`libcrypto`) (required)
- OpenMP (recommended; required by some myl7/fss configs)
- CUDA toolkit + device (optional, for GPU path)

Recommended (Ninja):

```bash
cmake -S . -B build_ninja -GNinja
ninja -C build_ninja
ctest --test-dir build_ninja --output-on-failure
```

CUDA build:

```bash
cmake -S . -B build_cuda -GNinja -DSUF_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80
ninja -C build_cuda
ctest --test-dir build_cuda --output-on-failure
```

Notes:
- `SUF_USE_MYL7_FSS` / `SUF_FETCH_MYL7_FSS` control the optional myl7/fss adapter (default ON/ON).
- `CMAKE_CUDA_ARCHITECTURES` defaults to `80;86` if not specified.

## Tests

The `ctest` suite is wired in `CMakeLists.txt`. Useful filters:

```bash
ctest --test-dir build_ninja -R "test_(pfss_async|pfss_planner|pfss_superbatch_limits|open_collector_packing)" --output-on-failure
ctest --test-dir build_ninja -R "test_(gapars_selector|range_proofs|hoist_rescales)" --output-on-failure
ctest --test-dir build_cuda  -R "test_cuda_(prg|pred_mask|packed_pfss|trunc_postproc|trunc_gapars|beaver_mul|recip_task)" --output-on-failure
ctest --test-dir build_cuda  -R "test_(pfss_gpu|softmax_gpu_smoke)" --output-on-failure
```

## Benchmarks / Demos

Build targets live under `src/bench/` and `src/demo/`:

- `build_ninja/bench_layer_breakdown` (end-to-end layer breakdown)
- `build_ninja/bench_truncation` (TR/ARS/GapARS costs)
- `build_ninja/bench_pfss_cpu_gpu` (PFSS CPU/GPU staging + timing)
- `build_ninja/bench_softmax_norm` (softmax/LN block benches)
- `build_cuda/bench_gemm_overlap` (PFSS + GEMM overlap on separate CUDA streams)
- `build_ninja/sim_harness` (tape + end-to-end simulation harness)

## Key Runtime Concepts (Code Pointers)

- **SUF→PFSS compilation**: `src/compiler/suf_to_pfss.cpp` produces predicate/coeff program descriptions and a `compiler::CompiledSUFGate`.
- **Range / GapCert / AutoTrunc**: `include/compiler/range_analysis.hpp`, `src/compiler/layer_graph.cpp` drive `GateKind::AutoTrunc` selection.
- **Composite‑FSS**: `include/gates/composite_fss.hpp` defines key material + eval glue for gates.
- **Scheduling / batching**:
  - `include/runtime/phase_executor.hpp` runs tasks based on `Need::{Open,PfssCoeff,PfssTrunc}`.
  - `include/runtime/pfss_superbatch.hpp` batches PFSS jobs (and can stage to GPU).
  - `include/runtime/open_collector.hpp` batches Beaver opens; optional packing in `src/runtime/open_collector.cpp`.

Correctness note: trunc/ARS helper bits (`carry/sign/wrap`) are maintained as **additive shares**; `wrap` is provided directly by a PFSS predicate output `1[hatx < r_in]` (no public `r_in` compare).

## Environment Knobs

- PFSS backend selection: `SUF_PFSS_BACKEND=cpu|gpu|auto` (where applicable)
- GPU PFSS tuning: `SUF_PFSS_GPU_BLOCK`, `RUN_GPU_COMPOSITE=1`
- GPU matmul tuning: `SUF_MATMUL_GPU_TILE=wide|narrow`
- GPU caches: `SUF_NO_CACHE_KEYS=1`, `SUF_NO_CACHE_HATX=1`
- Benchmark toggles:
  - `SUF_PER_ELEMENT_MASKS=0|1` disables/enables per-element trunc/ARS masks (benchmark sets `0` by default for batching)
  - `SUF_BENCH_DEVICE_PIPELINE=1` keeps PFSS outputs on GPU when downstream can consume device pointers
- Open batching/packing:
  - `SUF_OPEN_PACK_EFFBITS=1` enables packed opens when shares fit in small bitwidth
  - `SUF_OPEN_PACK_MAX_BITS` caps packing width (default 48)

## Docs

- Milestone acceptance tracker: `docs/milestone_acceptance.md`
- Milestone task audit: `docs/milestone_task_audit.md`
- Design/notes: `milestone*.md`, `milestone11_gpu.md`, `revise_m11.md`, `super.md`
- Paper draft: `paper.md`

## Sigma-vs-SUF Harness

See `docs/benchmark_sigma_vs_suf.md` for end-to-end comparison instructions (time + key size).

- Runner: `bench/run_sigma_vs_suf.py`
- Configs: `bench/configs/sigma_vs_suf.json`, `bench/configs/sigma_vs_suf_bert_tiny.json`, `bench/configs/sigma_vs_suf_large.json`
- Hardware probe: `scripts/describe_hardware.py` (writes `bench/hardware_<hostname>.json`)
- Sigma build helper: `scripts/build_sigma.sh`

Smoke (bert-tiny):

```bash
python3 bench/run_sigma_vs_suf.py --config bench/configs/sigma_vs_suf_bert_tiny.json --timeout-sigma-s 1800
```
