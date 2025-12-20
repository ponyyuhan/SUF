# Sigma vs SUF benchmark report

Generated: `2025-12-20` (updated)

This report follows the benchmarking/protocol accounting design in `paper.md`.

## Data sources

- Sigma-vs-SUF harness summary: `bench/results/summary.csv`
- Raw SUF logs: `bench/results/suf_*_gpu_L128_B1.json`
- Raw Sigma logs: `bench/results/sigma_*_L128.json` (parsed from `external/sigma_ezpc/GPU-MPC/experiments/sigma/output/`)

## Settings

- Sequence length: `128`
- Batch size: `1`
- Harness config: `bench/configs/sigma_vs_suf_quick_gpu.json`
- SUF iterations: `3` (`--n-iters 3`) so `timing.keygen_time_s` is separable from steady-state online timing
- SUF open packing enabled: `--open-pack 1` (see `SUF_OPEN_PACK_*` in `README.md`)
- SUF per-element masks disabled: `--per-element-masks 0` (to preserve batching)

## Results (online)

| Model | Sigma online (s) | Sigma online (GB) | SUF online (s) | SUF online (GB) | Time ratio | Byte ratio |
|---|---:|---:|---:|---:|---:|---:|
| bert-tiny | 0.173 | 0.022 | 0.256 | 0.027 | 1.48x | 1.23x |
| bert-base | 2.867 | 1.062 | 5.838 | 1.304 | 2.04x | 1.23x |
| bert-large | 7.127 | 2.833 | 15.611 | 3.477 | 2.19x | 1.23x |
| gpt2 | 2.426 | 0.885 | 5.162 | 1.664 | 2.13x | 1.88x |

## Bottlenecks (SUF)

Enable `SUF_BENCH_PROFILE=1` to populate `online_profile.*` in the SUF JSON logs, then inspect:
- `open_flush_ns` vs `pfss_flush_eval_ns` vs `pfss_finalize_ns` to decide whether time is dominated by openings vs PFSS evaluation/finalize.
- `open_pack_ns` vs `open_comm_ns` vs `open_scatter_ns` to separate host packing/scatter overhead from “wire time”.
- `pfss.open_flushes`, `pfss.num_jobs` (from the SUF JSON) to see whether the runtime is round/flush-limited.

## Implementation changes in this snapshot (performance + accounting)

- `include/proto/beaver_mul64.hpp`: PFSS-channel Beaver communication now packs each opened `(e,f)` element to `ceil(n_bits/8)` bytes (byte-truncation) instead of always sending 8 bytes.
- `src/runtime/open_collector.cpp`: GPU runs default to device-side pack/unpack + scatter + “keep opened on device” whenever a CUDA stream is available (still overridable via `SUF_OPEN_PACK_DEVICE*` env vars).
- `include/gates/composite_fss.hpp`: avoids wasted predicate/cutpoint evaluation work for gates whose boolean output arity is zero, and adds a Horner-only fast path for those gates.
- `cuda/pfss_backend_gpu.cu`: removed hash-based “cache” checks for device key uploads (they scanned O(bytes) on the CPU); keys/x values are now copied directly via `cudaMemcpyAsync`.

## Notes / known gaps

- With this snapshot, SUF still trails Sigma on online time and online bytes across the measured models.
- The dominant remaining gap is the total number of Beaver multiplications inside Composite-FSS evaluation (seen as high `communication.pfss_bytes` and many PFSS-channel messages), especially for GPT-2.
