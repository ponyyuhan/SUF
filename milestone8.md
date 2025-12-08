## Milestone 11 (Revised, compiler-driven) — **Global scheduling + maximal batching + secure faithful truncation/ARS as SUF/Composite**

Goal: turn the current “correct end-to-end layer” into a **SIGMA-like runtime** with (1) *cross-gate / cross-layer PFSS batching*, (2) *layer-wide Beaver open fusion*, (3) *CPU PFSS + GPU GEMM overlapped pipeline*, and (4) **secure faithful truncation/ARS** that is *compiled, scheduled, and hoisted* (not hand-inserted).

----

# 11.0 Why this milestone exists (and what changes vs now)

Right now your `matmul_publicW` and `matmul_beaver` end with **local arithmetic shifts** (`>> frac_bits`). As SIGMA notes, *local truncation is insecure* and also introduces correctness drift in long pipelines. So Milestone 11 replaces “local shift truncation” with **faithful truncation / ARS** protocols, and then *uses your SUF+Composite compiler/scheduler* to make them **cheaper than “drop-in SIGMA”** by:

* pushing truncation decisions into **compile-time range/gap proofs**
* compiling truncation into the **same PFSS program family** (predicates on masked inputs)
* **merging PFSS evals** (trunc + activation predicates co-scheduled)
* **hoisting** truncations (fewer trunc gates total)
* making the runtime **one-layer opens** + **packed communication** + **CPU/GPU overlap**

----

# 11.1 Secure Faithful Truncation / ARS as **first-class Composite gates**

### 11.1.1 Add new gate kinds + descriptors

**Add GateKind entries** (compiler + runtime):

* `GateKind::TR_Faithful` (truncate-reduce)
* `GateKind::ARS_Faithful`
* `GateKind::GapARS` (fast path when a “gap certificate” holds)
* (optional) `GateKind::SignExt` if you keep ARS = TR + SignExt as in SIGMA’s “no known gap” path

**Files**

* `include/gates/trunc_faithful_gate.hpp`
* `include/gates/ars_faithful_gate.hpp`
* `include/gates/gapars_gate.hpp`
* `include/compiler/pfss_program_desc.hpp` (new entries + params)
* `src/compiler/suf_to_pfss.cpp` (emit these gates from IR pass, see 11.2/11.3)

### 11.1.2 Implement protocols in *your masking model*

Use your existing invariant `(x̂ = x + r_in)` as the “opened masked value” that truncation protocols need anyway.

**Faithful TR (truncate-reduce)** (1 round in SIGMA; your PFSS already supports the needed compare):

* Inputs: shares of `x`, **public** `x̂`, and mask shares `r_in`.
* Target: `TR_{n,f}(x)` represented as an `n`-bit ring value (or `n-f` logical width stored in `n` bits).
* Core correction bit can be expressed as a PFSS predicate on `x̂ mod 2^f` vs `(r_in mod 2^f)` (exact form depends on your chosen derivation, but it’s a *masked-threshold compare* with target embedded in the key).
* Output style matches your system: **masked outputs** (`ŷ = y + r_out`) plus mask shares.

**Faithful ARS**
Two modes, selected by compiler (11.2):

* **GapARS** (cheap, 1 round): when a gap certificate holds (see SIGMA-style condition below), ARS reduces to a cheaper shift/correction structure (DPF on `f` bits only).
* **General ARS**: compose `TR_{n,f}` + `SignExt_{n-f → n}` with one additional PFSS compare for sign extension (or implement the exact SIGMA structure if you want strict comparability).

> Practical note: you don’t need to “copy SIGMA” line-by-line; you just need *the same semantics*, then your SUF compiler and batching machinery is the differentiator.

### 11.1.3 Integrate into Composite runtime with tape I/O

All trunc/ARS gates must support:

* `dealer_keygen(...)` producing PFSS keys + mask shares + any extra shared constants
* `eval_batch(...)` + `eval_single(...)`
* tape read/write (deterministic consumption)

**Files**

* `include/gates/composite_fss.hpp` (register new gate evaluators)
* `include/gates/postproc_hooks.hpp` (add “TruncHook/ARS-Hook” if you implement trunc as SUF+postproc; see 11.2)
* `src/demo/test_truncation.cpp` (new)
* `src/bench/bench_truncation.cpp` (new or fold into `bench_llm_gates`)

----

# 11.2 Compile-time **range / effective-bitwidth / gap certificates** (automatic GapARS)

This is where your approach becomes distinctly *not “just SIGMA engineering”*.

### 11.2.1 Add a range analysis pass over the NN graph

Compute per-tensor facts:

* signed range `[min,max]` in real domain (or conservative integer bounds)
* required headroom bits
* “effective bitwidth” (`eff_bits`) and “effective frac bits” (can differ from storage frac)
* whether values are provably within a **GapARS-safe region**

**GapARS certificate (SIGMA-style)**
SIGMA’s GapARS applies when cleartext lies in
`[0, 2^{n-2}) ∪ [2^n - 2^{n-2}, 2^n)` (intuitively: far from the “middle” where MSB-to-wrap ambiguity exists).
Your pass should produce:

* `GapCert{holds: bool, margin_bits: k}` where `k≥2` implies stricter than needed
* propagate through ops conservatively

**Files**

* `include/compiler/range_analysis.hpp`
* `src/compiler/range_analysis.cpp`
* extend `include/nn/tensor_view.hpp` (attach `{n, frac_bits, eff_bits, range}` metadata)
* extend `include/compiler/compiled_suf_gate.hpp` (store range facts on ports)

### 11.2.2 Emit the cheapest truncation gate automatically

From range facts:

* if `GapCert` holds → emit `GateKind::GapARS`
* else emit `GateKind::ARS_Faithful` or `TR_Faithful` + `SignExt`
* if the op only needs logical downscale (unsigned domain) → `TR_Faithful`

This should happen in a **graph rewrite pass** (before SUF→PFSS lowering), so truncation becomes part of your compiled program.

**Files**

* `src/compiler/insert_truncation_pass.cpp` (new)
* `include/compiler/passes.hpp` (register pass order)

----

# 11.3 Compile truncation into **SUF/Composite**, then merge PFSS programs

You asked specifically: *“compile truncation into SUF and merge PFSS across gate/layer; then truncation hoisting.”* Here is the concrete way to do it without breaking your SUF definition:

### 11.3.1 “Trunc as SUF + postproc” (fits your current architecture)

Truncation isn’t a polynomial, but it **is**:

* a small set of **predicate bits** on masked input (low-bit compares / MSB patterns), plus
* **linear post-processing** on shares (shift + add/sub small corrections)

So represent truncation as:

* **SUF predicate-only output** (or SUF with trivial arithmetic channel)
* then apply a **postproc hook** to compute the final arithmetic result

Concretely:

* SUF outputs helper bits: `carry`, `sign`, `wrap`, etc.
* Composite hook computes TR/ARS result on shares + adds output mask

**Benefits**

* the truncation predicates now live in the **same PFSS predicate batch** as other SUF predicates in that layer, enabling real merges.

**Files**

* `include/suf/trunc_suf_builders.hpp` (new “builders” that produce SUF IR for trunc/ars)
* `include/gates/postproc_hooks.hpp` (add `HookKind::FaithfulTR`, `HookKind::GapARS`, `HookKind::SignExt`)
* `src/compiler/suf_to_pfss.cpp` (lower trunc SUFs like any other SUF)

### 11.3.2 PFSS program merging (predicates + coefficient LUTs)

Implement a **PFSS merge planner** that, per layer, constructs:

* one **super-batch** for predicate evaluations (packed SoA)
* one (or few) super-batches for coefficient/LUT programs grouped by (domain bits, LUT shape, payload shape)

Key idea: your runtime should *never* “evaluate PFSS per gate object” once batched execution starts. It should evaluate PFSS per **layer-plan**.

**Files**

* `include/runtime/pfss_batch_plan.hpp`
* `src/runtime/pfss_batch_plan.cpp`
* `include/gates/composite_fss.hpp` updated to accept a `PFSSBatchPlan`

----

# 11.4 Truncation hoisting (fewer trunc calls, same or better accuracy)

### 11.4.1 What we hoist

Targets:

* after GEMMs (Q/K/V/Out/FFN projections) you currently rescale immediately
* in practice, you can often postpone rescale across:
    * bias add
    * residual add
    * sometimes even LN pre-statistics (if you track scale consistently)

### 11.4.2 Hoisting pass (compiler)

Introduce a pass that:

1. constructs a **scale graph**: every edge has `(frac_bits, eff_bits, range)`
2. finds maximal regions where values can remain in a higher-precision scale without overflow
3. replaces multiple truncations by a **single trunc** at the end of the region
   (or replaces TR+ARS by one ARS at region boundary)
4. updates downstream gate expectations (your SUF tables already parameterize `frac_bits`)

**Correctness policy**

* *Default mode*: “Sigma-compatible placement” (same trunc points, easier perf comparisons)
* *Optimized mode*: “Hoisted truncation placement” (expect slightly different numeric error; validated against plaintext tolerance)

**Files**

* `src/compiler/truncation_hoist_pass.cpp`
* new metadata: `TensorScale{frac_bits, scale_id}`

----

# 11.5 Layer-wide Beaver opens fusion (minimize round trips)

Implement a **LayerOpenCollector**:

* Collect every Beaver open `(E = X-A, F = W-B, …)` across the entire layer into:
    * one packed buffer per direction (or per stream)
* Perform **one network round-trip per layer** (or two if you split forward/backward dependencies)

**Files**

* `include/runtime/open_collector.hpp`
* `src/runtime/open_collector.cpp`
* modify `src/nn/matmul_beaver.cpp` to *not* open immediately; instead enqueue into collector

----

# 11.6 CPU PFSS + GPU GEMM overlapped pipeline (streams + double buffering)

### 11.6.1 Execution model

Per layer:

1. GPU launches GEMMs (cuBLASLt/CUTLASS), writes results to GPU buffers (still secret shares).
2. CPU concurrently runs PFSS eval prep / key expansion / AES-CTR for the *next* PFSS batch.
3. Network comm for Beaver opens overlaps with GPU compute.
4. GPU does pack/unpack + reduction kernels where safe.

### 11.6.2 Concrete implementation

* `ThreadPool` / work-stealing scheduler with 3 lanes:
    * Lane A: GPU GEMM stream(s)
    * Lane B: CPU PFSS (AES-NI, OpenMP)
    * Lane C: Comm (async send/recv)
* double-buffer:
    * `open_send_buf[2]`, `open_recv_buf[2]`
    * `pfss_in_buf[2]`, `pfss_out_buf[2]`

**Files**

* `include/runtime/pipeline.hpp`
* `src/runtime/pipeline.cpp`
* `include/runtime/cuda_stream_pool.hpp` + `src/runtime/cuda_stream_pool.cu` (if CUDA build)
* integrate into `src/nn/transformer_layer.cpp` so layer eval uses the pipeline API

----

# 11.7 GPU linear core: CUTLASS/cuBLASLt + epilogue fusion

Implement:

* `MatmulPublicW_GPU` using cuBLASLt (preferred for fused epilogues) and optional CUTLASS for custom layouts
* `MatmulBeaver_GPU` for local terms; Beaver opens still on CPU/comm

Key: support epilogues/fusions:

* bias add
* residual add
* optional “scale-only” operations (but **not** truncation; trunc is now faithful gate)

**Files**

* `include/nn/cuda/matmul_cublaslt.hpp`
* `src/nn/cuda/matmul_cublaslt.cu`
* `include/nn/cuda/epilogue_kernels.cuh`

----

# 11.8 SIGMA-style communication packing (GPU-side packing/unpacking)

Two packing layers:

1. **Network packing**
    * pack non-standard bitwidth ring elements (`eff_bits`) into dense byte streams
    * avoid sending full 64-bit words when only 37 bits are meaningful
2. **GPU transfer packing**
    * keep packed buffers on GPU
    * GPU kernels unpack to 32/64-bit lanes where needed

You already have `bytes` and pack/unpack utils; this milestone makes them **first-class in runtime** with per-tensor packing metadata from range analysis.

**Files**

* `include/runtime/comm_packing.hpp`
* `src/runtime/comm_packing.cpp`
* `src/runtime/cuda/pack_kernels.cu` (GPU pack/unpack)

----

# 11.9 “Only compute necessary elements” optimizations (attention/softmax)

Implement:

* causal mask optimization: when computing attention scores for prefill (`T×T`), skip upper triangle:
    * max reduction: only over valid prefix
    * nExp calls: only for valid positions
    * sum/recip: only valid positions
* in step mode (`t` incremental): attention is `1×t` anyway; keep it tight
* propagate this into scheduling so PFSS batches only contain the *used* elements.

**Files**

* `src/nn/attention_block.cpp` (generate index lists for valid entries)
* `include/runtime/indexed_batch.hpp` (PFSS eval over sparse indices)
* `bench_attention` updated to benchmark mask-skip effectiveness

----

# 11.10 Tests, benches, and acceptance criteria (“Definition of done”)

### 11.10.1 New correctness tests

1. **Faithful truncation unit tests**
    * `test_faithful_trunc.cpp`: TR/ARS/GapARS vs plaintext reference across random samples
    * include edge cases near boundaries and negative values
2. **Compiler proofs smoke tests**
    * ensure gap certificates are emitted when expected
    * ensure hoisting doesn’t violate range invariants
3. **End-to-end layer test**
    * transformer layer output matches plaintext within tolerance *after replacing local shifts*

### 11.10.2 New benchmarks

1. `bench_truncation --mode=tr|ars|gapars --N=...`
2. `bench_layer_breakdown --layer=... --B,T,D,H ...`
    * report:
        * PFSS time breakdown (pred vs coeff)
        * Beaver open time + bytes
        * GPU GEMM time
        * packing/unpacking time
        * total layer latency

### 11.10.3 Milestone 11 “Done” criteria

* **No local-shift truncation** remains in matmul paths (public or Beaver); all scaling is done via faithful TR/ARS gates (or hoisted variants).
* Compiler emits **GapARS** automatically when gap certificate holds.
* Runtime shows:
    * PFSS evals are executed as **layer-level super-batches**
    * Beaver opens are **fused per layer**
    * CPU PFSS + GPU GEMM **overlap** (visible in breakdown)
    * communication is **packed** according to `eff_bits`
* Demonstrate at least one representative config where **PFSS+comm are not dominating** the layer latency.

----

If you want, I can also give a concrete “patch-style checklist” for editing your existing `matmul_publicW.cpp` / `matmul_beaver.cpp` to remove the local shifts and route all rescale through the new truncation gates (including how to thread `(x̂, r_in)` through the layer planner).
