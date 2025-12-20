# Sigma vs SUF benchmark report

Generated: `2025-12-19`

This report follows the benchmarking/protocol accounting design in `paper.md`.

## Data sources

- SUF (this repo) logs: `bench/results/current_compare/2025-12-19_current_gpu`
- Sigma logs: `bench/results/current_compare/2025-12-19_current_gpu/sigma_*_L128.json` (copied from a previously captured Sigma run; Sigma was not rerun here due to prior build/run deadlocks)

## Settings

- Sequence length: `128`
- Batch size: `1`
- SUF iterations: `2` (`--n-iters 2`) so `timing.keygen_time_s` is separable from steady-state online timing
- SUF open packing enabled: `--open-pack 1` (see `SUF_OPEN_PACK_*` in `README.md`)
- SUF per-element masks disabled: `--per-element-masks 0` (to preserve batching)

## Results (online)

| Model | Sigma online (s) | Sigma online (GB) | SUF online (s) | SUF online (GB) | Time ratio | Byte ratio |
|---|---:|---:|---:|---:|---:|---:|
| bert-tiny | 0.194 | 0.022 | 0.158 | 0.017 | 0.82x | 0.77x |
| bert-base | 2.728 | 1.062 | 5.642 | 0.946 | 2.07x | 0.89x |
| bert-large | 6.862 | 2.833 | 14.213 | 2.558 | 2.07x | 0.90x |
| gpt2 | 2.389 | 0.885 | 4.539 | 1.100 | 1.90x | 1.24x |
| gpt-neo-1.3b | 11.324 | 4.326 | 33.350 | 5.312 | 2.95x | 1.23x |

Raw CSV/JSONL: `bench/results/current_compare/2025-12-19_current_gpu/summary.csv`, `bench/results/current_compare/2025-12-19_current_gpu/summary.jsonl`.

## Bottlenecks (SUF)

Enable `SUF_BENCH_PROFILE=1` to populate `online_profile.*` in the SUF JSON logs, then inspect:
- `open_flush_ns` vs `pfss_flush_eval_ns` vs `pfss_finalize_ns` to decide whether time is dominated by openings vs PFSS evaluation/finalize.
- `open_pack_ns` vs `open_comm_ns` vs `open_scatter_ns` to separate host packing/scatter overhead from “wire time”.
- `pfss.open_flushes`, `pfss.num_jobs` (from the SUF JSON) to see whether the runtime is round/flush-limited.

## Implementation changes in this snapshot (performance + accounting)

- `include/runtime/open_collector.hpp` and `src/runtime/open_collector.cpp`: added `OpenCollector::reserve()` and a contiguous pending buffer to remove flush-time gather copies.
- `include/proto/beaver_mul64.hpp`: combined `(e,f)` exchange into one message in `mul()`/`mul_batch()` to reduce PFSS-channel call overhead.
- `src/nn/attention_block.cpp` and `include/runtime/phase_tasks.hpp`: migrated hot open producers to `OpenCollector::reserve()` to avoid per-task temporary allocations/copies.
- `cuda/cuda_primitives.cu` + `include/runtime/cuda_primitives.hpp`: added an optional CUDA kernel to compute opened values on-device during device-packing; guarded by `SUF_OPEN_PACK_DEVICE_SCATTER=1` (off by default because it can increase contention when both parties share one GPU).

## Notes / known gaps

- SUF still trails Sigma on online time for larger models, even when SUF online bytes are smaller (BERT base/large). This indicates a nontrivial efficiency gap beyond just wire volume (e.g., per-round overheads, PFSS evaluation cost, and/or GPU/host staging).
- CPU end-to-end runs for the larger models (notably `gpt-neo-1.3b`) are currently too slow to include in the “current” snapshot; only GPU results are summarized here.

