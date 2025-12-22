# Sigma vs SUF End-to-End Benchmark (Time + Key Size)

This repo contains an end-to-end benchmark harness that compares:

- **SIGMA** (GPU-MPC implementation from ePrint 2023/1269)
- **SUF/PFSS prototype** (this repo)

It reports **online latency**, **communication**, and **offline key material size** (both systems) in a common JSON/CSV format.

## What “end-to-end” means here

### SIGMA

SIGMA’s `experiments/sigma/sigma` binary runs:

1. **Dealer/keygen** (offline): generates all FSS keys into CPU memory and reports:
   - `Total time=... us`
   - `Key size=... B (...)`
2. **Evaluator/online**: runs secure inference with a peer and reports:
   - `Total time=... us` (online)
   - `Comm time=... us`
   - `Transfer time=... us`
   - `Total Comm=... B (...)`

SIGMA writes these into:
`external/sigma_ezpc/GPU-MPC/experiments/sigma/output/P{0,1}/models/{model}-{seq}/dealer.txt|evaluator.txt`.

### SUF

`build_ninja/bench_suf_transformer` runs a transformer forward pass under the SUF/PFSS runtime:

- Two parties are simulated **in-process** with a paired channel.
- We force the **PFSS execution path** even when using the clear/reference backend (for stable end-to-end accounting) via `SUF_FORCE_PFSS=1`.
- We enable `SUF_BENCH_CACHE_MATERIAL=1` so expensive dealer-generated materials (GeLU/SiLU/NExp/Recip + trunc bundles) are generated once and reused; this makes **keygen vs online** separation meaningful.
- We disable per-element trunc/ARS masks by default (`SUF_PER_ELEMENT_MASKS=0`) to keep truncation batched; set `SUF_PER_ELEMENT_MASKS=1` to re-enable per-element masking.
- We collect:
  - `timing.keygen_time_s` (approx: first-iter minus steady-state online mean; set `n_iters >= 2`)
  - `timing.online_time_s` (steady-state mean over iters 1..N-1 when `n_iters >= 2`)
  - `timing.wall_time_s` (sum over measured iterations)
  - `timing.online_time_s_mean` and `timing.online_time_s_max`
  - `communication.online_bytes`
  - `preprocessing.key_bytes` (see “How SUF key size is measured” below)

## Repo code structure (benchmark-related)

- `bench/run_sigma_vs_suf.py`
  - Orchestrates runs and writes `summary.jsonl` + `summary.csv`.
  - Runs SIGMA by spawning two local processes (P0/P1) on loopback.
  - Runs SUF via `build_ninja/bench_suf_transformer`.
- `bench/configs/sigma_vs_suf.json`
  - Main config (models, seq_lens, binaries, output dirs).
- `bench/configs/sigma_vs_suf_bert_tiny.json`
  - Small-memory smoke config (bert-tiny only).
- `scripts/build_sigma.sh`
  - Clones EzPC if missing, updates submodules, applies local patches, builds Sytorch + SIGMA.
- `scripts/patches/sigma_gpu_mpc.patch`
  - Local patch applied to EzPC checkout:
    - Fixes an NVCC 11.5 include-order issue in `GPU-MPC/experiments/sigma/sigma.cu`.
    - Adds a fallback for `cudaMallocAsync` on systems/drivers that don’t support it.
- `src/demo/bench_suf_transformer.cpp`
  - SUF-side end-to-end benchmark binary (runs `n_layers_run` layers).
  - Counts SUF “key bytes” by serializing composite gate tapes at keygen time.

## How SUF key size is measured

SUF’s runtime generates per-gate correlated randomness and PFSS keys as “composite keys”.
To estimate *offline key material size*, we:

1. Install a keygen hook (`gates::set_composite_keygen_hook`) that fires whenever composite keys are generated.
2. For each composite key-pair, serialize it into a “tape” (`gates::composite_write_tapes`).
3. Sum `tape0.bytes + tape1.bytes`, counting **once** (party 0, iteration 0).

This produces `preprocessing.key_bytes` in SUF logs, comparable to SIGMA’s “Key size”.

## Activation parity (BERT / GeLU)

SIGMA’s BERT models use **GeLU**. This repo now supports a GeLU MLP activation:

- `include/gates/tables/gelu_spline_table.hpp` + `include/suf/suf_gelu_builders.hpp`
- `include/gates/gelu_composite.hpp`
- `include/nn/mlp_block.hpp` (`Activation::{SiLU,GeLU}`)
- `src/nn/mlp_block.cpp` uses GeLU when the model spec requests it
- `src/nn/model_specs.cpp` marks BERT/GPT2 as `"gelu"`

## Build and run

### 1) Build SUF

```bash
cmake -S . -B build_ninja -GNinja
ninja -C build_ninja bench_suf_transformer
```

### 2) Build SIGMA (EzPC GPU-MPC)

```bash
bash scripts/build_sigma.sh
```

This produces:
`external/sigma_ezpc/GPU-MPC/experiments/sigma/sigma`

### 3) Run the comparison harness

Smoke (bert-tiny):

```bash
python3 bench/run_sigma_vs_suf.py --config bench/configs/sigma_vs_suf_bert_tiny.json --timeout-sigma-s 1800
```

General config:

```bash
python3 bench/run_sigma_vs_suf.py --config bench/configs/sigma_vs_suf.json --timeout-sigma-s 7200
```

Outputs:

- Per-run JSON logs in `results_dir`
- `summary.jsonl`
- `summary.csv`

## Current “apples-to-apples” config (GPU)

The repo maintains a small, stable comparison set (GPU-only, seq=128, B=1, includes GPT‑Neo 1.3B):

```bash
python3 bench/run_sigma_vs_suf.py --config bench/configs/sigma_vs_suf_current_gpu.json --timeout-sigma-s 1200
```

Outputs land in `bench/results/current_compare/` (notably `summary.csv` / `summary.jsonl`).

See `benchmark_report.md` for the latest tables and interpretation notes.
