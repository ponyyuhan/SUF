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
- libsodium (required when building the optional myl7/fss adapter)
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
- GPU postproc fast-paths (require CUDA + `PhaseExecutor` device pipeline in most cases):
  - `SUF_TRUNC_GPU=1` (GPU trunc/ARS postproc when PFSS device slices exist)
  - `SUF_HORNER_GPU=1` (GPU Horner polynomial evaluation for cubic bundles)
  - `SUF_SOFTMAX_GPU=1` (GPU row-sum / ragged row-sum inside softmax)
  - `SUF_LN_GPU=1` (GPU row-sum/variance helpers inside LayerNormTask)
  - `SUF_MUL_GPU=1` (GPU fast-path for BeaverMul64-based elementwise mul, where wired)
  - `SUF_MATMUL_BEAVER_GPU=1` (GPU Beaver matmul tasks, where wired)
- Benchmark toggles:
  - `SUF_PER_ELEMENT_MASKS=0|1` disables/enables per-element trunc/ARS masks (benchmark sets `0` by default for batching)
  - `SUF_BENCH_DEVICE_PIPELINE=1` keeps PFSS outputs on GPU when downstream can consume device pointers
  - `SUF_BENCH_CACHE_MATERIAL=1` caches expensive dealer-generated materials (GeLU/SiLU/nExp/recip + trunc bundles)
  - `SUF_FORCE_PFSS=1` forces the PFSS execution path even when a reference fast-path exists
  - `SUF_BENCH_PFSS_RING_POW2=20..30` sets the in-process PFSS byte-ring capacity as `2^k` bytes (bench harness)
  - `SUF_BENCH_NET_RING_POW2=20..28` sets the in-process net-channel ring capacity as `2^k` u64 words (bench harness)
- Open batching/packing:
  - `SUF_OPEN_PACK_EFFBITS=1` enables packed opens when shares fit in small bitwidth
  - `SUF_OPEN_PACK_MAX_BITS` caps packing width (default 56)
  - `SUF_OPEN_PACK_AUTO=1` skips packing when savings are small (default on in the benchmark)
  - `SUF_OPEN_PACK_MIN_SAVINGS_PCT` sets the minimum savings to keep packing (default 25)
  - `SUF_OPEN_PACK_DYNAMIC=1` uses per-flush max bitwidth to shrink packing width (off by default)
- BERT-Tiny activation tuning:
  - `SUF_GELU_CONST=1` enables a fast piecewise-constant GeLU approximation
  - `SUF_GELU_CONST_SEGMENTS` controls GeLU LUT granularity (default `256` in the benchmark)

## Performance Optimization Guide

This repo is intentionally “knob-heavy”: different models and phases stress different parts of the stack.
The goal is to keep the **protocol semantics** aligned with `paper.md` while minimizing **online time**.

### What “online” means (and what is / isn’t interactive)

- **SUF/PFSS evaluation** is *non-interactive* given the public masked input `hatx = x + r_in` (each party can locally evaluate its key on `hatx`).
- End-to-end transformer execution is still **interactive online** because it performs:
  - **opens** (e.g., to reveal `hatx` and Beaver `d,e` terms), and
  - **Beaver multiplication** for non-linear glue.

SIGMA is also an online two-party protocol; in the Sigma-vs-SUF harness we normalize Sigma’s single “Total Comm” bucket into `communication.net_bytes` so that byte objects match SUF’s reporting.

### Biggest levers for online time

- **Reduce interactive bandwidth**:
  - Enable packed opens: `SUF_OPEN_PACK_EFFBITS=1` (benchmark `--open-pack 1`).
  - If packing overhead dominates, keep it off and rely on `SUF_OPEN_PACK_AUTO=1` to skip marginal savings.
  - Avoid per-element trunc/ARS masks when batching matters: `SUF_PER_ELEMENT_MASKS=0`.
- **Avoid unnecessary flush/round overhead**:
  - Prefer `PhaseExecutor` lazy scheduling (default) so it batches opens/PFSS work until tasks stall.
  - Keep batches alive across phases (`PhaseExecutor::set_keep_batches(true)` is used in the transformer stack).
  - When profiling shows `open_pack_ns` or `open_scatter_ns` dominating, focus on host-side packaging before touching PFSS kernels.
  - If `pfss.num_flushes` ≈ `pfss.num_jobs`, the run is latency-bound by PFSS round trips; improving this typically requires batching more jobs per phase or swapping in a faster PFSS backend.
- **Exploit GPU overlap where safe**:
  - `attention_block.cpp` uses an **overlap GEMM stream** (`MatmulParams::overlap_stream`) so GPU PFSS kernels can overlap with matmul.
  - Enable weight/bias caching on GPU where pointers are stable: `MatmulParams::cache_weights/cache_bias`.
- **Make device-pipeline explicit** (when downstream can consume device pointers):
  - `SUF_BENCH_DEVICE_PIPELINE=1` turns on `PhaseExecutor` device pipeline and requests PFSS device slices.
  - Pair with `SUF_TRUNC_GPU=1` / `SUF_HORNER_GPU=1` / `SUF_SOFTMAX_GPU=1` to actually use the staged device slices.
- **Watch host-side packaging**:
  - `OpenCollector` flatten/scatter can dominate `online_profile.open_pack_ns` / `open_scatter_ns` on large models.
  - The current implementation parallelizes request flattening and reuses scratch buffers to reduce allocation overhead.
  - Bit-packing uses block-aligned chunks to enable OpenMP parallelization where `eff_bits` aligns to 64-bit words.
  - If open-pack time is still high, keep `SUF_OPEN_PACK_AUTO=1` and avoid forcing packing at 50–51-bit rings.

## Latest Benchmark Snapshot (SUF-only)

Collected with `build_ninja/bench_suf_transformer` (seq=128, batch=1, n_iters=1).
Full logs and Sigma baselines are in `benchmark_report.md`.

GPU:

| model | timing.online_time_s | communication.net_bytes | preprocessing.key_bytes |
| --- | ---: | ---: | ---: |
| bert-tiny | 0.508969 | 16,623,616 | 26,054,620 |
| bert-base | 5.93575 | 1,608,214,464 | 1,785,520,576 |
| bert-large | 16.2673 | 4,288,139,136 | 5,441,517,076 |
| gpt2 | 5.17519 | 1,402,218,432 | 1,728,894,396 |
| gpt-neo-1.3b | 44.1131 | 6,607,720,320 | 8,100,843,680 |

CPU:

| model | timing.online_time_s | communication.net_bytes | preprocessing.key_bytes |
| --- | ---: | ---: | ---: |
| bert-tiny | 0.245207 | 12,075,328 | 33,511,972 |
| bert-base | 45.113 | 434,702,208 | 1,059,609,376 |
| bert-large | 263.356 | 1,159,204,608 | 3,390,247,064 |
| gpt2 | 49.1632 | 378,521,472 | 1,002,096,824 |

### CPU monitoring

All end-to-end transformer benches log:
- `resources.cpu_user_s`, `resources.cpu_sys_s`
- `resources.cpu_util_avg` (process CPU time / wall time)
- `resources.cpu_util_samples_avg`, `resources.cpu_util_samples_max` (sampled CPU utilization)
- `resources.max_rss_kb`

Use these alongside `pfss.{num_jobs,num_flushes,open_flushes,opened_words}` to determine whether you’re bottlenecked by **protocol rounds**, **GPU kernels**, or **host-side packing/scatter**.

To get a timing breakdown of online work, set:
- `SUF_BENCH_PROFILE=1` (adds `online_profile.*` to the JSON logs)
- `SUF_BENCH_CPU_SAMPLER=1` and `SUF_BENCH_CPU_SAMPLE_MS=200` (optional CPU sampling)

### Notes on alternative DPF/DCF backends (libdpf / grotto)

This prototype currently includes a “sigmafast” backend and GPU stubs geared toward the paper’s measurement model.
Swapping the PFSS predicate backend to something like `libdpf`/grotto can be a good next step if profiling shows predicate evaluation dominates:

- Candidate: `external` DPF backends (e.g. grotto in `libdpf`) typically target high-throughput PRG + tree traversal.
- Integration constraints:
  - Keep the **same public masked input** model (`hatx`) and predicate semantics used in `paper.md`.
  - Preserve accounting objects (`communication.*`, `preprocessing.key_bytes_scope`) so comparisons remain consistent.

## Accuracy Bench (Table 4 style)

We include a scaffold to compare PyTorch (float32), Sigma (Table 4), and SUF fixed-point emulation:

```bash
python3 bench/accuracy_compare.py \
  --config bench/configs/accuracy_table4.json \
  --out-json bench/results/accuracy/accuracy_table4_suf.json \
  --out-md bench/results/accuracy/accuracy_table4_suf.md \
  --device cpu
```

Notes:
- The scaffold assumes **task-specific fine-tuned checkpoints** for GLUE tasks.
- `bench/configs/accuracy_table4.json` now includes fine-tuned BERT-tiny/BERT-large checkpoints; Llama2 entries remain skipped due to gated weights.
- SUF emulation applies fixed-point rounding at `frac_bits=12` and the same bitwidths as Sigma for comparability.

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

### Benchmark Defaults (BERT-Tiny)

`build_ninja/bench_suf_transformer` sets performance-focused defaults for end-to-end runs:

- `proto::set_ring_bits(spec.n_bits)` (BERT-Tiny uses `n_bits=37`)
- `SUF_OPEN_PACK_EFFBITS=1` by default on GPU (auto-pack can still skip small savings); CPU default OFF
- `SUF_OPEN_PACK_AUTO=1` (skip packing when savings are small)
- `SUF_PER_ELEMENT_MASKS=0` (avoid per-element trunc/ARS masks that prevent batching)
- GPU runs: `SUF_FORCE_PFSS=1` (stable PFSS accounting); CPU runs keep the deterministic reference fast-path unless you export `SUF_FORCE_PFSS=1`
- GPU runs: `SUF_BENCH_NET_RING_POW2=24` (larger net ring to reduce backpressure)
- BERT-Tiny: `SUF_GELU_CONST=1` and `SUF_GELU_CONST_SEGMENTS=256`

### Sigma (EzPC GPU-MPC) + SEAL Setup

Recommended build (auto-detects `CUDA_VERSION` and `GPU_ARCH`, and applies local patches including a SEAL mutex-include fix):

```bash
bash scripts/build_sigma.sh
```

Manual overrides:

```bash
CUDA_VERSION=11.7 GPU_ARCH=86 bash scripts/build_sigma.sh
```

Outputs:
- SIGMA binary: `external/sigma_ezpc/GPU-MPC/experiments/sigma/sigma`
- SEAL source vendored under: `external/sigma_ezpc/GPU-MPC/ext/sytorch/ext/sci/extern/SEAL` (built as part of Sytorch/SCI)

Host prerequisites (Ubuntu/Debian):

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build python3 python3-pip \\
  libssl-dev libgmp-dev libmpfr-dev libsodium-dev
```
