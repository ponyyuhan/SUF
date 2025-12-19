# Sigma vs SUF benchmarks (updated)

Generated: `2025-12-19`

## Commands run

- Tests: `ctest --test-dir build_ninja --output-on-failure`
- SUF GPU benches (seq=128, batch=1):
  - `build_ninja/bench_suf_transformer --model bert-tiny --backend gpu --seq-len 128 --batch-size 1 --n-iters 1 --log-json bench/results/opt_runs/bert-tiny_gpu_buf.json`
  - `build_ninja/bench_suf_transformer --model bert-base --backend gpu --seq-len 128 --batch-size 1 --n-iters 1 --log-json bench/results/opt_runs/bert-base_gpu_buf.json`
  - `build_ninja/bench_suf_transformer --model bert-large --backend gpu --seq-len 128 --batch-size 1 --n-iters 1 --log-json bench/results/opt_runs/bert-large_gpu_buf.json`
  - `build_ninja/bench_suf_transformer --model gpt2 --backend gpu --seq-len 128 --batch-size 1 --n-iters 1 --log-json bench/results/opt_runs/gpt2_gpu_buf.json`
  - `build_ninja/bench_suf_transformer --model gpt-neo-1.3b --backend gpu --seq-len 128 --batch-size 1 --n-iters 1 --log-json bench/results/opt_runs/gpt-neo-1.3b_gpu_buf.json`
- SUF CPU benches (seq=128, batch=1):
  - `build_ninja/bench_suf_transformer --model bert-tiny --backend cpu --seq-len 128 --batch-size 1 --n-iters 1 --log-json bench/results/opt_runs/bert-tiny_cpu_buf.json`
  - `build_ninja/bench_suf_transformer --model bert-base --backend cpu --seq-len 128 --batch-size 1 --n-iters 1 --log-json bench/results/opt_runs/bert-base_cpu_buf.json`
  - `build_ninja/bench_suf_transformer --model bert-large --backend cpu --seq-len 128 --batch-size 1 --n-iters 1 --log-json bench/results/opt_runs/bert-large_cpu_buf.json`
  - `build_ninja/bench_suf_transformer --model gpt2 --backend cpu --seq-len 128 --batch-size 1 --n-iters 1 --log-json bench/results/opt_runs/gpt2_cpu_buf.json`
- Profiling highlight:
  - `SUF_BENCH_PROFILE=1 build_ninja/bench_suf_transformer --model gpt-neo-1.3b --backend gpu --seq-len 128 --batch-size 1 --n-iters 1 --log-json bench/results/opt_runs/gpt-neo-1.3b_gpu_profile_buf.json`
- Note: SIGMA benchmarks were not rerun due to prior build deadlocks; see “Sigma baseline” below.

## Stats consistency notes (paper.md)

- **Offline key bytes (`preprocessing.key_bytes`)**: SUF now treats this as **dealer-total bytes across both parties** and tags the object as `preprocessing.key_bytes_scope = "dealer_total"`.
- **Online bytes (`communication.*`)**:
  - SUF reports `communication.net_bytes` (wire bytes on the net-channel) and `communication.pfss_bytes` (wire bytes on the PFSS-channel), with `communication.online_bytes = net_bytes + pfss_bytes`.
  - PFSS evaluation is **non-interactive** at the protocol level (paper.md §3), but the benchmark harness can still show nonzero `communication.pfss_bytes` (e.g., backend bookkeeping / simulated PFSS channel usage). Online comm is still dominated by openings (`communication.open_*`), i.e., **net bytes**.
- **SIGMA schema normalization**: `bench/run_sigma_vs_suf.py` now maps SIGMA’s `Total Comm` to `communication.net_bytes` so Sigma/SUF comparisons use consistent byte objects.
- **More open breakdown**: SUF logs now include `pfss.opened_words_{beaver,mask,other}` (and corresponding `communication.open_bytes_*`) to align the “what are we counting?” question with Sigma’s single `Total Comm` bucket.

## Results (latest SUF-only)

### GPU (seq=128, batch=1)

Source: `bench/results/opt_runs/*_gpu_buf.json`

| model | preprocessing.key_bytes | timing.online_time_s | communication.net_bytes |
| --- | ---: | ---: | ---: |
| bert-tiny | 26,054,620 | 0.508969 | 16,623,616 |
| bert-base | 1,785,520,576 | 5.93575 | 1,608,214,464 |
| bert-large | 5,441,517,076 | 16.2673 | 4,288,139,136 |
| gpt2 | 1,728,894,396 | 5.17519 | 1,402,218,432 |
| gpt-neo-1.3b | 8,100,843,680 | 44.1131 | 6,607,720,320 |

### CPU (seq=128, batch=1)

Source: `bench/results/opt_runs/*_cpu_buf.json`

| model | preprocessing.key_bytes | timing.online_time_s | communication.net_bytes |
| --- | ---: | ---: | ---: |
| bert-tiny | 33,511,972 | 0.245207 | 12,075,328 |
| bert-base | 1,059,609,376 | 45.113 | 434,702,208 |
| bert-large | 3,390,247,064 | 263.356 | 1,159,204,608 |
| gpt2 | 1,002,096,824 | 49.1632 | 378,521,472 |

### Online profile highlight (gpt-neo-1.3b, GPU)

Source: `bench/results/opt_runs/gpt-neo-1.3b_gpu_profile_buf.json`

| metric | value (ns) |
| --- | ---: |
| open_flush_ns | 13,943,162,902 |
| open_pack_ns | 678,436,741 |
| open_comm_ns | 9,419,577,990 |
| open_scatter_ns | 3,576,591,938 |

Note: with scratch-buffer reuse, `open_pack_ns` dropped below 1s for gpt-neo-1.3b; `open_comm_ns` and PFSS flush eval remain dominant.
Packing at 50–51 bits still increased total time in local tests because CPU pack/unpack adds multiple seconds per run; net-byte savings did not offset the cost.

## Sigma baseline (from 2025-12-18 run)

| model | preprocessing.key_bytes | timing.online_time_s | communication.net_bytes |
| --- | ---: | ---: | ---: |
| bert-tiny | 350,064,640 | 0.186802 | 21,675,034 |
| bert-base | 18,075,947,008 | 3.386349 | 1,062,390,674 |
| bert-large | 48,799,535,104 | 8.253441 | 2,832,800,546 |
| gpt2 | 15,346,094,080 | 2.947291 | 885,146,258 |
| gpt-neo-1.3b | 81,805,541,376 | 13.723769 | 4,325,592,866 |

## LLaMA2-7B status (not rerun)

- The attempted SIGMA run for LLaMA2-7B (`bench/configs/sigma_vs_suf_llama2_7b.json`) did not produce `dealer.txt/evaluator.txt` and one party exited with `-9` (likely OOM/kill). No SUF run was summarized for that config.
