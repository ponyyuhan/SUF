Here’s a concrete plan you can hand to Codex as a “do this end-to-end” spec. I’ll structure it like a project brief with clearly numbered tasks.

I’ll assume:

* Sigma repo: `https://github.com/mpc-msri/EzPC/tree/master/GPU-MPC`
* Sigma’s SyTorch/sigma experiments under `GPU-MPC/experiments/sigma` as per the paper.
* Your repo: the SUF/PFSS prototype you described (with `sim_harness`, `bench_pfss_cpu_gpu`, `bench_softmax_norm`, `bench_gemm_overlap`, GPU backend, etc.).

The goal is: **apples-to-apples comparison on real transformers** (BERT, GPT-2, GPT-Neo, optionally Llama2) on *your* CPU and GPU(s), for both CPU and GPU variants of Sigma and your system.

---

## 0. High-level goals & metrics (for Codex to keep in mind)

Tell Codex explicitly:

> The benchmark harness must produce, for each model/system/hardware combination:
>
> * Online latency (seconds) per forward pass.
> * Communication volume (GB) during online phase.
> * Preprocessing time & key size (optional but nice to have).
> * # of cryptographic calls (e.g., PRG/DPC/DPF evals) if easily available.
> * Basic hardware stats: max GPU memory used, CPU threads used.

We want:

* **Models**: same as Sigma paper where feasible: BERT-base, BERT-large, GPT-2 (124M), GPT-Neo-1.3B, and optionally Llama2-7B/13B if your GPU allows.
* **Sequence length**: start with 128 tokens to match Sigma’s primary experiments.
* **Security/precision parameters**: fixed-point precision `f = 12`, and bitwidths chosen so that Sigma matches PyTorch accuracy (≈ 37 bits for BERT-tiny, 48–51 bits for others).

---

## 1. Environment + hardware documentation

**Task 1A – Hardware JSON**

Have Codex create a small script `scripts/describe_hardware.py` in *your* repo that prints a JSON blob like:

```json
{
  "hostname": "...",
  "cpu_model": "...",
  "cpu_cores": 32,
  "ram_gb": 256,
  "gpus": [
    {
      "name": "NVIDIA A40",
      "memory_gb": 48,
      "cuda_driver": "XXX",
      "cuda_runtime": "YYY"
    }
  ],
  "network": {
    "measured_bandwidth_gbps": 9.4,
    "ping_ms": 0.05
  }
}
```

* Use `lscpu`, `nvidia-smi --query-gpu=name,memory.total --format=csv` and a simple `iperf3` call between the two machines if you have two hosts.
* Store this JSON under `bench/hardware_<hostname>.json`.

**Task 1B – Network mode**

Decide *in the config*:

* Mode `LAN`: two physical machines, one party per machine (closest to Sigma’s setup).
* Mode `Loopback`: two local processes on the same host (if you only have one machine). For fairness vs Sigma’s numbers, just record that your network is different.

---

## 2. Getting Sigma (code + models) ready

### 2.1. Clone & build Sigma

**Task 2A – Clone into a fixed directory**

Tell Codex:

1. Create `external/` under your SUF repo root if it doesn’t exist.

2. Inside `external/`, run:

   ```bash
   git clone https://github.com/mpc-msri/EzPC.git sigma_ezpc
   ```

3. We will treat `external/sigma_ezpc/GPU-MPC` as the Sigma root.

**Task 2B – Build Sigma CPU + GPU**

In `external/sigma_ezpc/GPU-MPC`:

* Create `scripts/build_sigma.sh` that:

  ```bash
  #!/usr/bin/env bash
  set -euo pipefail

  ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

  mkdir -p "$ROOT/build_cpu" "$ROOT/build_gpu"

  # CPU build (Omega: 4 threads default, as in Sigma’s paper)
  cmake -S "$ROOT" -B "$ROOT/build_cpu" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSIGMA_BACKEND=CPU
  cmake --build "$ROOT/build_cpu" -j$(nproc)

  # GPU build
  cmake -S "$ROOT" -B "$ROOT/build_gpu" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSIGMA_BACKEND=GPU
  cmake --build "$ROOT/build_gpu" -j$(nproc)
  ```

* Adjust flags according to their README / CMake options once Codex inspects the repo; the key is: **two build trees**, one CPU, one GPU.

### 2.2. Sigma models + datasets

From the Sigma paper: they use BERT-tiny/base/large on GLUE (SST-2, QNLI, MRPC), GPT-2, GPT-Neo-1.3B and Llama2-7B/13B on Lambada, sequence length 128.

For *performance* comparison you don’t actually need GLUE / Lambada data; random token inputs with the same sequence length suffice. But it’s good to have the correct pretrained weights.

**Task 2C – HF download script for Sigma**

In `external/sigma_ezpc/GPU-MPC/experiments/sigma` (or similar experiments folder), create `scripts/download_models.py`:

* Use `huggingface_hub` / `transformers` to download:

    * `bert-base-uncased`
    * `bert-large-uncased`
    * `gpt2`
    * `EleutherAI/gpt-neo-1.3B`
    * Optionally: `meta-llama/Llama-2-7b-hf`, `meta-llama/Llama-2-13b-hf` (if your GPU memory allows and you have accepted their license).

* Save models under a dedicated directory, e.g. `external/models_hf/`.

If Sigma already has scripts for this (very likely), instruct Codex to:

> Search for existing model download scripts in `experiments/sigma` (look for `download_models`, `models.py`, etc.) and extend them to support exactly the set above, with an option to skip Llama if GPU RAM < X GB.

**Task 2D – SyTorch configs**

Sigma uses a C++ “SyTorch” frontend that describes models (see sample Transformer block in Appendix I).

For bench fairness, we want:

* SyTorch configs for: BERT-base, BERT-large, GPT-2, GPT-Neo-1.3B.
* Quantization set to `f = 12`, bitwidths as in Table 4 (e.g., 50–51 bits for most models).

Have Codex:

* Inspect how existing Sigma experiments represent models (likely some C++ config or templated model classes).

* Add a tiny C++ or JSON “model registry” mapping:

  ```text
  name → (arch, HF_id, n_embd, n_layer, n_head, n_ctx, n_bits, frac_bits)
  ```

* Expose it via a small C++ CLI (`sigma_run_model`) that takes `--model gpt2 --backend cpu|gpu --seq-len 128 --batch-size 1`.

---

## 3. Preparing your SUF/PFSS code for the same models

### 3.1. Model spec bridge

You already have transformer blocks in `nn/transformer_layer.cpp`, `nn/attention_block.cpp`, `nn/mlp_block.cpp`, and a `LayerContext` that records rescale/trunc/clamp plans and shares a `PfssSuperBatch`. Your models are described in C++ already; what’s missing is a *common spec* shared with Sigma.

**Task 3A – Shared model spec file**

In *your* repo, create `include/nn/model_specs.hpp` plus `src/nn/model_specs.cpp` with something like:

```cpp
struct ModelSpec {
    std::string name;
    std::string hf_id;
    std::size_t n_layers;
    std::size_t n_heads;
    std::size_t d_model;
    std::size_t d_ff;
    std::size_t max_seq_len;
    unsigned fixed_n_bits;
    unsigned fixed_frac_bits;
};

const ModelSpec &get_model_spec(const std::string &name);
std::vector<ModelSpec> list_model_specs();
```

* Fill it with the same models and bitwidths you used for Sigma, e.g.:

    * `"bert-base"`: `n_layers=12, n_heads=12, d_model=768, fixed_n_bits=50, fixed_frac_bits=12`.
    * `"gpt2"`: `n_layers=12, n_heads=12, d_model=768, fixed_n_bits=50, fixed_frac_bits=12`.
    * `"gpt-neo-1.3b"`: etc (Codex can pull from HF configs).

**Task 3B – SUF transformer harness**

Create `src/demo/bench_suf_transformer.cpp`:

* Parse CLI arguments:

  ```bash
  ./bench_suf_transformer \
      --model gpt2 \
      --backend cpu|gpu \
      --seq-len 128 \
      --batch-size 1 \
      --n-iters 10 \
      --log-json out.json
  ```

* Steps inside `main()`:

    1. Lookup `ModelSpec spec = get_model_spec(model_name)`.
    2. Construct your transformer graph/layer stack using `nn/transformer_layer.cpp` with `spec`’s hyperparameters.
    3. Initialize fixed-point scaling (n, f) from `spec` and your `range_analysis` / `LayerContext`.
    4. Create random input shares consistent with your two-party protocol (same seed on both parties for reproducibility).
    5. Run a warm-up pass (to amortize compilation/JIT effects).
    6. Run `n_iters` forward passes, measuring total **online** time (from protocol start to end).
    7. Collect:

        * Total PFSS jobs, bytes, and flushes from `PfssSuperBatch`.
        * Total opened words/bytes from `OpenCollector`.
        * Preprocessing time and key size (if you run a separate dealer phase; you can reuse your existing SUF/PFSS keygen tests).
    8. Emit a JSON line with all metrics (see §5).

---

## 4. Unified benchmark driver comparing Sigma vs SUF

Now we want a single driver (in your repo) that:

* Spawns Sigma and SUF runs.
* Uses identical model specs, precision, and sequence length.
* Consolidates metrics into one CSV/JSON table.

**Task 4A – Bench config file**

Create `bench/configs/sigma_vs_suf.json`:

```json
{
  "models": ["bert-base", "bert-large", "gpt2", "gpt-neo-1.3b"],
  "seq_lens": [128],
  "batch_sizes": [1],
  "backends": ["cpu", "gpu"],
  "n_iters": 10,
  "sigma_root": "external/sigma_ezpc/GPU-MPC",
  "results_dir": "bench/results"
}
```

**Task 4B – Python orchestrator**

Add `bench/run_sigma_vs_suf.py` in *your* repo:

* For each `(model, seq_len, batch, backend)`:

    1. **Sigma run**:

        * Call Sigma CLI (you’ll define it in §2.2), e.g.:

          ```bash
          sigma_root/build_<backend>/bin/sigma_run_model \
            --model <name> --seq-len <L> --batch-size <B> \
            --backend <cpu|gpu> \
            --n-iters <N> \
            --log-json <results_dir>/sigma_<model>_<backend>.json
          ```

    2. **SUF run**:

        * Call your CLI:

          ```bash
          ./build/bench_suf_transformer \
            --model <name> --seq-len <L> --batch-size <B> \
            --backend <cpu|gpu> \
            --n-iters <N> \
            --log-json <results_dir>/suf_<model>_<backend>.json
          ```

    3. Parse both JSON logs and append a combined row to:

        * `bench/results/summary.csv`
        * `bench/results/summary.jsonl`

* The orchestrator should also embed the hardware JSON (from §1) into each run’s metadata.

---

## 5. Instrumentation: JSON schema & metrics

To make Codex’s life easy, define a JSON schema both Sigma and SUF harnesses emit.

**Task 5A – JSON schema**

A single object per run:

```json
{
  "system": "sigma" | "suf",
  "backend": "cpu" | "gpu",
  "model": "gpt2",
  "seq_len": 128,
  "batch_size": 1,
  "n_layers": 12,
  "n_heads": 12,
  "d_model": 768,
  "n_bits": 50,
  "frac_bits": 12,
  "hardware": { ... from hardware JSON ... },
  "timing": {
    "online_time_s_mean": 1.51,
    "online_time_s_std": 0.05,
    "preproc_time_s": 10.2,
    "total_time_s": 11.7
  },
  "communication": {
    "online_bytes": 8.2e8,
    "preproc_bytes": 3.1e9
  },
  "pfss": {
    "num_jobs": 12345,
    "total_hatx_words": 987654,
    "num_flushes": 37
  },
  "nonlinear_breakdown": {
    "gelu_calls": 1234,
    "silu_calls": 0,
    "softmax_calls": 768,
    "layernorm_calls": 192
  },
  "notes": "any extra debug info"
}
```

* For Sigma, you can approximate these counts by instrumenting their protocol wrappers (e.g., every time they call a GeLU protocol, increment a counter).
* For SUF, you already know this from your gate compiler / runtime.

**Task 5B – Where to hook communication**

For Both Sigma and SUF:

* Identify their **network abstraction** (likely some `Channel` object in GPU-MPC / your `proto` layer).

* Insert global counters:

  ```cpp
  static std::atomic<std::uint64_t> g_bytes_sent{0};
  static std::atomic<std::uint64_t> g_bytes_recv{0};
  ```

* Every send/recv increments these counters by the size.

* At end of the online phase, dump `g_bytes_sent + g_bytes_recv` as `communication.online_bytes`.

For preprocessing, either:

* Run a separate command for dealer/keygen and instrument that, or
* Split by phase in your protocol (if you already track which messages belong to preproc).

---

## 6. Ensuring fairness vs Sigma’s settings

Instruct Codex to enforce:

1. **Same numerical parameters**: Match bitwidths and precision from Sigma’s Table 4 (BERT/GPT/Llama).

2. **Same sequence length**: 128 tokens; optionally vary 64/256/512 to reproduce sequence-length scaling (Sigma’s Appendix K).

3. **Same model hyperparameters**: load from HF configs, don’t hand-wave them.

4. **Same number of CPU threads**:

    * For CPU backends, fix OMP threads to e.g. 4, as Sigma uses 4 threads in their eval.
    * For GPU, ensure CPU threads don’t explode (e.g., set `OMP_NUM_THREADS=4`).

5. **GPU memory constraints**:

    * Before running Llama2-7B/13B, check free GPU memory. Sigma used A6000s with 46GB; if your GPU has significantly less RAM, mark those models as “not supported” in the bench config rather than OOM-ing repeatedly.

---

## 7. Ragged / effective-bitwidth experiments (optional but nice)

Your system has explicit **effective bitwidth (`eff_bits`) and ragged packing** hooks; Sigma has effective bitwidth optimizations for GeLU/Softmax as well.

You can define **secondary experiments**:

* For each model, run:

    * Sigma default (with its effective bitwidth optimizations).
    * Your system with/without eff_bits and with/without ragged packing enabled.

Extend JSON schema with:

```json
"config_flags": {
  "sigma_eff_bits": true,
  "suf_eff_bits": true,
  "suf_ragged_pack": true
}
```

And add CLI flags:

* `--disable-eff-bits`, `--disable-ragged-pack` to your SUF harness.
* Equivalent flags for Sigma if they expose them; otherwise document that Sigma’s optimizations are always on.

---

## 8. Micro-benches (optional, but useful for the paper)

In addition to end-to-end transformers, define a **micro-bench suite**:

* Compare:

    * GeLU / SiLU
    * Softmax
    * LayerNorm / RMSNorm

for a single layer with realistic sizes (e.g., `d_model=768`, `seq_len=128`) on Sigma vs SUF, CPU vs GPU, as Sigma does in Table 2–3.

Have Codex:

* Add `bench/bench_nonlinear_sigma_vs_suf.py`.
* Wrap existing micro-bench binaries:

    * Sigma likely has tests/benches for each primitive.
    * Your repo already has `bench_softmax_norm`, `test_layernorm_task`, etc.

Emit a similar JSON schema but with `model="micro_gelu"` etc.

---

## 9. Concrete “prompt style” instructions for Codex

To give Codex directly, you can wrap the above into something like:

> You are editing a C++/CMake + Python project that contains a SUF/PFSS prototype and has `external/sigma_ezpc/GPU-MPC` as a subdirectory. Implement the following tasks, in order:
>
> 1. **Create a shared model specification** in `include/nn/model_specs.hpp` and `src/nn/model_specs.cpp` with entries for `bert-base`, `bert-large`, `gpt2`, `gpt-neo-1.3b`, and optionally `llama2-7b`, `llama2-13b`. For each, store HF ID, transformer hyperparameters, and fixed-point (n_bits, frac_bits) matching Sigma’s Table 4 (f=12, bitwidths 48–51).
> 2. **Add a SUF transformer benchmark binary** `bench_suf_transformer` in `src/demo/bench_suf_transformer.cpp`, wired into CMake, that:
     >
     >    * Parses CLI args (`--model`, `--backend`, `--seq-len`, `--batch-size`, `--n-iters`, `--log-json`).
>    * Builds the corresponding transformer using your existing `nn/transformer_layer.cpp` and `LayerContext`, with correct bitwidths.
>    * Runs a warm-up + N timed online iterations under PhaseExecutor, collecting PFSS/Open stats and communication bytes (instrument `proto` network layer if needed).
>    * Outputs a single JSON object to the specified file following the schema in §5.
> 3. **In the Sigma repo** (`external/sigma_ezpc/GPU-MPC`), add:
     >
     >    * `scripts/build_sigma.sh` to build CPU and GPU variants.
>    * A CLI binary `sigma_run_model` (C++), similar to SyTorch examples, that accepts the same CLI as `bench_suf_transformer` and runs Sigma’s protocols for that model/backend, producing the same JSON schema (sys=“sigma”).
>    * A Python script `experiments/sigma/scripts/download_models.py` that downloads the HF models (`bert-base-uncased`, `bert-large-uncased`, `gpt2`, `EleutherAI/gpt-neo-1.3B`, optional Llama2) into a known directory.
> 4. **Add a global network byte counter** in both codebases (Sigma + SUF) by wrapping send/recv in the respective channel abstraction and recording total bytes sent/received for online and preprocessing phases separately. Expose the totals to the benchmark CLIs so they can include them under `communication.online_bytes` etc.
> 5. **Create a unified Python orchestrator** `bench/run_sigma_vs_suf.py` in the SUF repo that:
     >
     >    * Reads `bench/configs/sigma_vs_suf.json`.
>    * Iterates over all models, backends, and sequence lengths.
>    * Calls Sigma’s `sigma_run_model` and your `bench_suf_transformer` with consistent args.
>    * Parses their JSON outputs and writes a combined CSV/JSONL with one row per run.
> 6. **Create a hardware description script** `scripts/describe_hardware.py` that collects CPU/GPU/network info as in §1 and writes it to `bench/hardware_<hostname>.json`. Merge this info into the benchmark JSON outputs.
> 7. (Optional) **Extend with micro-benchmarks** `bench/bench_nonlinear_sigma_vs_suf.py` and additional CLIs to reproduce/compare the per-nonlinearity numbers similar to Sigma’s Table 2–3.

---

If you give Codex this structured spec (maybe broken into a couple of prompts), it should have a clear path to:

* Wire up Sigma and your SUF/PFSS prototype.
* Run the same big models (BERT, GPT-2, GPT-Neo, maybe Llama2) on your CPU/GPU.
* Produce directly comparable latency + communication + complexity numbers suitable for the “Sigma vs Composite-FSS” section of your paper.
