# Sigma vs SUF benchmark report

Generated: `2025-12-22` (updated)

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
- Models: `bert-tiny`, `bert-base`, `bert-large`, `gpt2`, `gpt-neo-1.3b`
- SUF iterations: `3` (`--n-iters 3`) so `timing.keygen_time_s` is separable from steady-state online timing
- SUF open packing enabled: `--open-pack 1` (see `SUF_OPEN_PACK_*` in `README.md`)
- SUF per-element masks disabled (default): `SUF_PER_ELEMENT_MASKS=0` (to preserve batching)

## Results (online)

| Model | Sigma online (s) | Sigma online (GB) | SUF online (s) | SUF online (GB) | Time ratio | Byte ratio |
|---|---:|---:|---:|---:|---:|---:|
| bert-tiny | 0.169924 | 0.022 | 0.059369 | 0.016 | 0.35x | 0.75x |
| bert-base | 2.659462 | 1.062 | 1.479530 | 0.580 | 0.56x | 0.55x |
| bert-large | 6.774047 | 2.833 | 3.348170 | 1.547 | 0.49x | 0.55x |
| gpt2 | 2.451263 | 0.885 | 1.405910 | 0.609 | 0.57x | 0.69x |
| gpt-neo-1.3b | 11.236027 | 4.326 | 4.273380 | 2.658 | 0.38x | 0.61x |

## Results (preprocessing)

| Model | Sigma keygen (s) | Sigma key (GB) | SUF keygen (s) | SUF key (GB) | Time ratio | Byte ratio |
|---|---:|---:|---:|---:|---:|---:|
| bert-tiny | 1.573166 | 0.350 | 0.325510 | 0.026 | 0.21x | 0.07x |
| bert-base | 20.372591 | 18.076 | 0.542003 | 1.135 | 0.03x | 0.06x |
| bert-large | 41.705359 | 48.800 | 0.387240 | 2.989 | 0.01x | 0.06x |
| gpt2 | 20.523452 | 15.346 | 0.492133 | 1.135 | 0.02x | 0.07x |
| gpt-neo-1.3b | 40.242867 | 81.806 | 3.504040 | 3.693 | 0.09x | 0.05x |

## Bottlenecks (SUF)

Enable `SUF_BENCH_PROFILE=1` to populate `online_profile.*` in the SUF JSON logs, then inspect:
- `open_flush_ns` vs `pfss_flush_eval_ns` vs `pfss_finalize_ns` to decide whether time is dominated by openings vs PFSS evaluation/finalize.
- `open_pack_ns` vs `open_comm_ns` vs `open_scatter_ns` to separate host packing/scatter overhead from “wire time”.
- `pfss.open_flushes`, `pfss.num_jobs`, `pfss.num_flushes` (from the SUF JSON, and now also in `bench/results/summary.csv`) to see whether the runtime is round/flush-limited.
- `resources.cpu_util_avg` and `preprocessing.open_pack_device_min_words` to detect CPU oversubscription or suboptimal open-pack thresholds (these can dominate end-to-end time in the single-process 2-party harness).

## Key optimizations enabled in this snapshot

- `include/nn/layer_context.hpp`: caches expensive scans of public weights (`row_l1_max` and `range_from_public_weights`) to avoid repeated O(|W|) host work (critical for GPT‑Neo 1.3B). Toggle via `SUF_CACHE_WEIGHT_BOUNDS=0|1` (default: enabled).
- `include/gates/composite_fss.hpp`, `cuda/cuda_primitives.cu`: device‑pipeline interval‑LUT outputs are masked (`+ r_out_share`) on GPU (avoids large synchronous H2D copies).
- `src/runtime/open_collector.cpp`: GPU runs use device-side pack/unpack + scatter and can keep opened values on device (`SUF_OPEN_PACK_DEVICE*=1` defaults when CUDA is available).
- `include/runtime/phase_tasks.hpp`, `src/nn/attention_block.cpp`: Beaver mul/matmul tasks reuse CUDA scratch buffers (and can cache H2D uploads of fixed key material) to reduce allocator + memcpy overhead in steady-state online runs.
- `src/nn/mlp_block.cpp`: reuses large intermediate buffers across layers/iters (thread-local) and truncates in-place to avoid allocator + extra copies dominating large-model online time.

## Notes / known gaps

- SUF beats Sigma on all current measured models (`bert-tiny`, `bert-base`, `bert-large`, `gpt2`, `gpt-neo-1.3b`) in both online time and online bytes under the benchmark settings described above.
- The largest remaining sensitivity is *host-side contention* in the single-process harness; always record `resources.cpu_util_avg` alongside timing/bytes when comparing variants.
