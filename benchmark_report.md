# Sigma vs SUF benchmark report

Generated: `2025-12-21` (updated)

This report follows the benchmarking/protocol accounting design in `paper.md`.

## Data sources

- Sigma-vs-SUF harness summary: `bench/results/current_compare/summary.csv`
- Raw SUF logs: `bench/results/current_compare/suf_*_gpu_L128_B1.json`
- Ad-hoc SUF “bench now” logs (latest workspace run): `bench/results/bench_now_*_gpu.json`
- Raw Sigma logs: `bench/results/current_compare/sigma_*_L128.json` (parsed from `external/sigma_ezpc/GPU-MPC/experiments/sigma/output/`)

## Settings

- Sequence length: `128`
- Batch size: `1`
- Harness config: `bench/configs/sigma_vs_suf_current_gpu.json`
- SUF iterations: `3` (`--n-iters 3`) so `timing.keygen_time_s` is separable from steady-state online timing
- SUF open packing enabled: `--open-pack 1` (see `SUF_OPEN_PACK_*` in `README.md`)
- SUF per-element masks disabled: `--per-element-masks 0` (to preserve batching)

## Results (online)

| Model | Sigma online (s) | Sigma online (GB) | SUF online (s) | SUF online (GB) | Time ratio | Byte ratio |
|---|---:|---:|---:|---:|---:|---:|
| bert-tiny | 0.166 | 0.022 | 0.075 | 0.016 | 0.45x | 0.75x |
| bert-base | 2.557 | 1.062 | 2.710 | 0.930 | 1.06x | 0.87x |
| bert-large | 6.932 | 2.833 | 8.449 | 2.479 | 1.22x | 0.87x |
| gpt2 | 2.353 | 0.885 | 3.249 | 1.081 | 1.38x | 1.22x |

## Results (preprocessing key size)

| Model | Sigma key (GB) | SUF key (GB) | Ratio |
|---|---:|---:|---:|
| bert-tiny | 0.350 | 0.026 | 0.07x |
| bert-base | 18.076 | 1.786 | 0.10x |
| bert-large | 48.800 | 4.763 | 0.10x |
| gpt2 | 15.346 | 1.786 | 0.12x |

## Bottlenecks (SUF)

Enable `SUF_BENCH_PROFILE=1` to populate `online_profile.*` in the SUF JSON logs, then inspect:
- `open_flush_ns` vs `pfss_flush_eval_ns` vs `pfss_finalize_ns` to decide whether time is dominated by openings vs PFSS evaluation/finalize.
- `open_pack_ns` vs `open_comm_ns` vs `open_scatter_ns` to separate host packing/scatter overhead from “wire time”.
- `pfss.open_flushes`, `pfss.num_jobs`, `pfss.num_flushes` (from the SUF JSON, and now also in `bench/results/summary.csv`) to see whether the runtime is round/flush-limited.

## Implementation changes in this snapshot (performance + accounting)

- `include/proto/beaver_mul64.hpp`: PFSS-channel Beaver communication now packs each opened `(e,f)` element to `ceil(n_bits/8)` bytes (byte-truncation) instead of always sending 8 bytes.
- `src/runtime/open_collector.cpp`: GPU runs default to device-side pack/unpack + scatter + “keep opened on device” whenever a CUDA stream is available (still overridable via `SUF_OPEN_PACK_DEVICE*` env vars).
- `cuda/cuda_primitives.cu`, `include/runtime/cuda_primitives.hpp`, `include/runtime/phase_tasks.hpp`: MulTask’s GPU Beaver mul now uses device opens (when available) and an AoS triple kernel to cut host staging overhead.
- `include/gates/composite_fss.hpp`: avoids wasted predicate/cutpoint evaluation work for gates whose boolean output arity is zero, and adds a Horner-only fast path for those gates.
- `include/gates/composite_fss.hpp`: parallelizes interval-LUT output masking (`+ r_out_share`) with OpenMP for large batches to reduce host-side overhead and party skew (which inflates `open_comm_ns` in the single-process harness).
- `cuda/pfss_backend_gpu.cu`: removed hash-based “cache” checks for device key uploads (they scanned O(bytes) on the CPU); keys/x values are now copied directly via `cudaMemcpyAsync`.
- `cuda/pfss_kernels.cu`, `cuda/pfss_backend_gpu.cu`: added shared-memory cached broadcast kernels for small LUT/cutpoint tables to reduce repeated global loads of key material in the hot GPU PFSS path.
- `src/demo/bench_suf_transformer.cpp`: reduced the default OpenMP thread count in the 2-party single-process harness for GPU runs to avoid starving the in-process “net ring” spin loops (reduces `open_comm` and improves end-to-end online time).
- `src/demo/bench_suf_transformer.cpp`: tuned benchmark default `SUF_OPEN_PACK_DEVICE_MIN_WORDS` to reduce GPU packing contention with PFSS kernels.
- `src/runtime/pfss_superbatch.cpp`: device-output buffer owners now use a stager-independent `cudaFree` deleter so they can outlive the benchmark-scoped stager without crashing.

## Notes / known gaps

- SUF beats Sigma on `bert-tiny` (online time + bytes) and remains byte-competitive on `bert-base`/`bert-large`.
- Remaining gap: `bert-base`, `bert-large`, and `gpt2` still trail Sigma on online time; `gpt2` also has higher online bytes, indicating the dominant bottleneck remains the number of Beaver/mask openings (not PFSS channel bytes, which remain 0 for SUF by design).
