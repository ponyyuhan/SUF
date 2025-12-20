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
| bert-tiny | 0.173 | 0.022 | 0.405 | 0.033 | 2.34x | 1.51x |
| bert-base | 2.867 | 1.062 | 9.432 | 1.529 | 3.29x | 1.44x |
| bert-large | 7.127 | 2.833 | 25.752 | 4.077 | 3.61x | 1.44x |
| gpt2 | 2.426 | 0.885 | 7.247 | 1.702 | 2.99x | 1.92x |

## Bottlenecks (SUF)

Enable `SUF_BENCH_PROFILE=1` to populate `online_profile.*` in the SUF JSON logs, then inspect:
- `open_flush_ns` vs `pfss_flush_eval_ns` vs `pfss_finalize_ns` to decide whether time is dominated by openings vs PFSS evaluation/finalize.
- `open_pack_ns` vs `open_comm_ns` vs `open_scatter_ns` to separate host packing/scatter overhead from “wire time”.
- `pfss.open_flushes`, `pfss.num_jobs` (from the SUF JSON) to see whether the runtime is round/flush-limited.

## Implementation changes in this snapshot (performance + accounting)

- `include/gates/composite_fss.hpp`: reduced Beaver round overhead inside Composite-FSS evaluation:
  - fused cutpoint selector network into 2 `mul_batch` calls per block,
  - fused selector-weighted boolean blending to 1 `mul_batch` per piece (instead of 1 per boolean output),
  - fused Horner’s rule multiplications across all arithmetic outputs (1 `mul_batch` per degree step).
- `include/proto/beaver_mul64.hpp`: thread-local scratch for batched `mul_batch` (avoids allocation churn); OpenMP gating for very large batches.
- `cuda/pfss_backend_gpu.cu`: packed GPU DCF keys store `alpha` as a u64 threshold (avoid per-bit comparisons in kernels).

## Notes / known gaps

- With this snapshot, SUF still trails Sigma on both online time and online bytes across all measured models.
- The dominant gap is in `communication.pfss_bytes` (Composite-FSS Beaver work) and overall `timing.online_time_s` for large models.
