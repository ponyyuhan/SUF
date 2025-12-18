# Sigma vs SUF benchmarks (updated)

Generated: `2025-12-18`

## Commands run

- Tests: `ctest --test-dir build_ninja --output-on-failure`
- Benchmarks:
  - `python3 bench/run_sigma_vs_suf.py --config bench/configs/sigma_vs_suf_bert_tiny.json --timeout-sigma-s 7200`
  - `python3 bench/run_sigma_vs_suf.py --config bench/configs/sigma_vs_suf_gpt2.json --timeout-sigma-s 7200`
  - `python3 bench/run_sigma_vs_suf.py --config bench/configs/sigma_vs_suf_bert_base.json --timeout-sigma-s 7200`
  - `python3 bench/run_sigma_vs_suf.py --config bench/configs/sigma_vs_suf_large_opt.json --skip-sigma`
  - Attempted (failed SIGMA run): `python3 bench/run_sigma_vs_suf.py --config bench/configs/sigma_vs_suf_llama2_7b.json --timeout-sigma-s 20000`

## Stats consistency notes (paper.md)

- **Offline key bytes (`preprocessing.key_bytes`)**: SUF now treats this as **dealer-total bytes across both parties** and tags the object as `preprocessing.key_bytes_scope = "dealer_total"`.
- **Online bytes (`communication.*`)**:
  - SUF reports `communication.net_bytes` (wire bytes on the net-channel) and `communication.pfss_bytes` (wire bytes on the PFSS-channel), with `communication.online_bytes = net_bytes + pfss_bytes`.
  - In this prototype, PFSS evaluation is **non-interactive** (paper.md §3), so `communication.pfss_bytes` is typically `0` and online comm is dominated by openings (`communication.open_*`), i.e., **net bytes**.
- **SIGMA schema normalization**: `bench/run_sigma_vs_suf.py` now maps SIGMA’s `Total Comm` to `communication.net_bytes` so Sigma/SUF comparisons use consistent byte objects.
- **More open breakdown**: SUF logs now include `pfss.opened_words_{beaver,mask,other}` (and corresponding `communication.open_bytes_*`) to align the “what are we counting?” question with Sigma’s single `Total Comm` bucket.

## Results (selected)

### BERT-Tiny (seq=128)

Source: `bench/results/bert_tiny/summary.csv`

| system | backend | preprocessing.key_bytes | timing.online_time_s | communication.net_bytes |
| --- | --- | ---: | ---: | ---: |
| sigma | gpu | 350,064,640 | 0.186802 | 21,675,034 |
| suf | gpu | 26,063,948 | 0.158082 | 16,623,616 |
| suf | cpu | 33,511,972 | 0.220452 | 8,087,424 |

### GPT-2 (seq=128)

Source: `bench/results/gpt2/summary.csv`

| system | backend | preprocessing.key_bytes | timing.online_time_s | communication.net_bytes | pfss.open_flushes | pfss.opened_words |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| sigma | gpu | 15,346,094,080 | 2.947291 | 885,146,258 | 0 | 0 |
| suf | gpu | 1,135,322,556 | 2.830700 | 627,795,456 | 1,152 | 97,400,832 |

### Larger models (optimized SUF flags, seq=128)

Source: `bench/results/large_models/summary.csv`

| system | model | preprocessing.key_bytes | timing.online_time_s | communication.online_bytes |
| --- | --- | ---: | ---: | ---: |
| sigma | bert-base | 18,075,947,008 | 3.386349 | 1,062,390,674 |
| suf | bert-base | 1,135,312,856 | 3.989390 | 792,826,368 |
| sigma | bert-large | 48,799,535,104 | 8.253441 | 2,832,800,546 |
| suf | bert-large | 3,696,962,152 | 11.163500 | 2,190,297,600 |
| sigma | gpt-neo-1.3b | 81,805,541,376 | 13.723769 | 4,325,592,866 |
| suf | gpt-neo-1.3b | 4,296,609,384 | 25.427800 | 2,745,113,088 |

## LLaMA2-7B status

- The attempted SIGMA run for LLaMA2-7B (`bench/configs/sigma_vs_suf_llama2_7b.json`) did not produce `dealer.txt/evaluator.txt` and one party exited with `-9` (likely OOM/kill). No SUF run was summarized for that config.
