# Sigma vs SUF benchmark report

Generated: `2025-12-20`

This report follows the benchmarking/protocol accounting design in `paper.md`.

## Data sources

- SUF (this repo) logs: `bench/results/current_compare/2025-12-20_prefillfix_gpu`
- Sigma logs: `bench/results/current_compare/2025-12-20_prefillfix_gpu/sigma_*_L128.json` (copied from a previously captured Sigma run; Sigma was not rerun here due to prior build/run deadlocks)

## Settings

- Sequence length: `128`
- Batch size: `1`
- SUF iterations: `2` (`--n-iters 2`) so `timing.keygen_time_s` is separable from steady-state online timing
- SUF open packing enabled: `--open-pack 1` (see `SUF_OPEN_PACK_*` in `README.md`)
- SUF per-element masks disabled: `--per-element-masks 0` (to preserve batching)
- SUF causal prefill enabled: `SUF_CAUSAL_PREFILL=1` (full-matrix causal prefill)

## Results (online)

| Model | Sigma online (s) | Sigma online (GB) | SUF online (s) | SUF online (GB) | Time ratio | Byte ratio |
|---|---:|---:|---:|---:|---:|---:|
| bert-tiny | 0.194 | 0.022 | 0.390 | 0.033 | 2.01x | 1.51x |
| bert-base | 2.728 | 1.062 | 9.258 | 1.529 | 3.39x | 1.44x |
| bert-large | 6.862 | 2.833 | 26.094 | 4.113 | 3.80x | 1.45x |
| gpt2 | 2.389 | 0.885 | 6.994 | 1.395 | 2.93x | 1.58x |
| gpt-neo-1.3b | 11.324 | 4.326 | 38.638 | 6.100 | 3.41x | 1.41x |

Raw CSV/JSONL: `bench/results/current_compare/2025-12-20_prefillfix_gpu/summary.csv`, `bench/results/current_compare/2025-12-20_prefillfix_gpu/summary.jsonl`.

Additional run (did not improve time in this environment): `bench/results/current_compare/2025-12-20_prefillfix_gpu_kernels` (enabling `SUF_MATMUL_BEAVER_GPU=1`, `SUF_MUL_GPU=1`).

## Bottlenecks (SUF)

Enable `SUF_BENCH_PROFILE=1` to populate `online_profile.*` in the SUF JSON logs, then inspect:
- `open_flush_ns` vs `pfss_flush_eval_ns` vs `pfss_finalize_ns` to decide whether time is dominated by openings vs PFSS evaluation/finalize.
- `open_pack_ns` vs `open_comm_ns` vs `open_scatter_ns` to separate host packing/scatter overhead from “wire time”.
- `pfss.open_flushes`, `pfss.num_jobs` (from the SUF JSON) to see whether the runtime is round/flush-limited.

## Implementation changes in this snapshot (performance + accounting)

- `src/nn/attention_block.cpp`: fixed causal prefill softmax correctness by avoiding in-place use of `nn::RowMaxDiffTask` (the output buffer must not alias the input unless the task is explicitly alias-safe).
- `include/nn/row_maxdiff_task.hpp`: made `nn::RowMaxDiffTask` robust to in-place use by snapshotting the original active inputs before writing outputs.

## Notes / known gaps

- With this snapshot, SUF still trails Sigma on both online time and online bytes across all measured models.
