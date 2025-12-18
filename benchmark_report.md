# BERT-Tiny (“BERT-Tintin”) benchmark: Sigma vs SUF

- Repo commit: `958ace32fb6f5f200b68e41e169ddfab3c28c01c`
- Generated: `2025-12-18T06:38:11Z`

## Commands run

- Tests: `ctest --test-dir build_ninja --output-on-failure` (34/34 passed)
- Benchmark: `python3 bench/run_sigma_vs_suf.py --config bench/configs/sigma_vs_suf_bert_tiny.json --timeout-sigma-s 3600`

## Baseline results (before SUF dealer-style LN material caching)

Source: `bench/results/bert_tiny/baseline_summary.csv`

| system | backend | model | seq_len | batch_size | preprocessing.key_bytes | timing.keygen_time_s | timing.online_time_s | timing.wall_time_s | communication.online_bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sigma | gpu | bert-tiny | 128 |  | 350064640 | 3.51469 | 0.252971 | 20.35753345489502 | 21675034 |
| suf | cpu | bert-tiny | 128 | 1 | 38248 | 0.031156 | 0.191792 | 0.606535 | 8046528 |
| suf | gpu | bert-tiny | 128 | 1 | 57824 | 0.30374 | 0.09386 | 0.585323 | 16590912 |

## Updated results (after SUF dealer-style LN material caching)

Change: `src/nn/transformer_layer.cpp` now caches LayerNorm/rsqrt dealer material when `SUF_BENCH_CACHE_MATERIAL=1`, so keygen is performed once (shared across both in-process parties) instead of being redundantly regenerated and double-counted.

Source: `bench/results/bert_tiny/summary.csv`

| system | backend | model | seq_len | batch_size | preprocessing.key_bytes | timing.keygen_time_s | timing.online_time_s | timing.wall_time_s | communication.online_bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sigma | gpu | bert-tiny | 128 |  | 350064640 | 2.810994 | 0.242699 | 20.531543016433716 | 21675034 |
| suf | cpu | bert-tiny | 128 | 1 | 15280 | 0.0331171 | 0.19048 | 0.604559 | 8046528 |
| suf | gpu | bert-tiny | 128 | 1 | 30512 | 0.299857 | 0.0916698 | 0.574869 | 16590912 |

## Delta (updated - baseline)

| system | backend | metric | baseline | new | delta | delta% |
|---|---|---:|---:|---:|---:|---:|
| sigma | gpu | preprocessing.key_bytes | 350064640 | 350064640 | 0 | 0.00% |
| sigma | gpu | timing.keygen_time_s | 3.51469 | 2.81099 | -0.703696 | -20.02% |
| sigma | gpu | timing.online_time_s | 0.252971 | 0.242699 | -0.010272 | -4.06% |
| sigma | gpu | timing.wall_time_s | 20.3575 | 20.5315 | 0.17401 | 0.85% |
| sigma | gpu | communication.online_bytes | 21675034 | 21675034 | 0 | 0.00% |
| suf | cpu | preprocessing.key_bytes | 38248 | 15280 | -22968 | -60.05% |
| suf | cpu | timing.keygen_time_s | 0.031156 | 0.0331171 | 0.0019611 | 6.29% |
| suf | cpu | timing.online_time_s | 0.191792 | 0.19048 | -0.001312 | -0.68% |
| suf | cpu | timing.wall_time_s | 0.606535 | 0.604559 | -0.001976 | -0.33% |
| suf | cpu | communication.online_bytes | 8046528 | 8046528 | 0 | 0.00% |
| suf | gpu | preprocessing.key_bytes | 57824 | 30512 | -27312 | -47.23% |
| suf | gpu | timing.keygen_time_s | 0.30374 | 0.299857 | -0.003883 | -1.28% |
| suf | gpu | timing.online_time_s | 0.09386 | 0.0916698 | -0.0021902 | -2.33% |
| suf | gpu | timing.wall_time_s | 0.585323 | 0.574869 | -0.010454 | -1.79% |
| suf | gpu | communication.online_bytes | 16590912 | 16590912 | 0 | 0.00% |

