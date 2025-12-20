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
| bert-tiny | 0.173 | 0.022 | 0.117 | 0.016 | 0.68x | 0.75x |
| bert-base | 2.867 | 1.062 | 2.913 | 0.930 | 1.02x | 0.88x |
| bert-large | 7.127 | 2.833 | 9.014 | 2.479 | 1.26x | 0.88x |
| gpt2 | 2.426 | 0.885 | 2.600 | 1.095 | 1.07x | 1.24x |

## Bottlenecks (SUF)

Enable `SUF_BENCH_PROFILE=1` to populate `online_profile.*` in the SUF JSON logs, then inspect:
- `open_flush_ns` vs `pfss_flush_eval_ns` vs `pfss_finalize_ns` to decide whether time is dominated by openings vs PFSS evaluation/finalize.
- `open_pack_ns` vs `open_comm_ns` vs `open_scatter_ns` to separate host packing/scatter overhead from “wire time”.
- `pfss.open_flushes`, `pfss.num_jobs` (from the SUF JSON) to see whether the runtime is round/flush-limited.

## Implementation changes in this snapshot (performance + accounting)

- `include/proto/beaver_mul64.hpp`: PFSS-channel Beaver communication now packs each opened `(e,f)` element to `ceil(n_bits/8)` bytes (byte-truncation) instead of always sending 8 bytes.
- `src/runtime/open_collector.cpp`: GPU runs default to device-side pack/unpack + scatter + “keep opened on device” whenever a CUDA stream is available (still overridable via `SUF_OPEN_PACK_DEVICE*` env vars).
- `cuda/cuda_primitives.cu`, `include/runtime/cuda_primitives.hpp`, `include/runtime/phase_tasks.hpp`: MulTask’s GPU Beaver mul now uses device opens (when available) and an AoS triple kernel to cut host staging overhead.
- `include/gates/composite_fss.hpp`: avoids wasted predicate/cutpoint evaluation work for gates whose boolean output arity is zero, and adds a Horner-only fast path for those gates.
- `cuda/pfss_backend_gpu.cu`: removed hash-based “cache” checks for device key uploads (they scanned O(bytes) on the CPU); keys/x values are now copied directly via `cudaMemcpyAsync`.

## Notes / known gaps

- SUF now beats Sigma on `bert-tiny` (time+bytes) and matches Sigma’s bytes on `bert-base`/`bert-large` while remaining close on time.
- Remaining gap: `gpt2` still trails Sigma on time and has higher online bytes, indicating the dominant bottleneck is still the number of Beaver openings (not PFSS bytes, which remain 0 for SUF by design).
