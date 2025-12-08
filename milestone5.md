# Milestone 7 — **SigmaFastBackend (CPU, SIGMA-like throughput)**

## 7.0 Goal / Scope

**Goal:** Replace the “baseline” backend cost profile (one-DCF-per-query or step-DCF loops) with a backend that has:

* **Packed comparisons**: evaluate many `1[hatx < theta_j]` (and `hatx_low < theta`) using *one traversal* per group, outputting bitmasks / bit‑shares efficiently.
* **Vector‑payload interval LUT**: fetch coefficient vectors in ~one PFSS call per element (or close), not `O(num_cuts)` calls.

**Scope:** CPU only (SIMD + threads). GPU comes after (a later milestone).

You already have:

* stable PFSS interface + SoA pack/unpack
* validated compiler outputs `PredProgramDesc` / `CoeffProgramDesc`
* working Clear + myl7, and a SigmaFast stub

Milestone 7 turns the stub into a **real engine**.

---

## 7.1 Design choice: two phases (practical, low risk)

### Phase A (fast integration, immediate speedups)

Implement `SigmaFastBackend` that **keeps your PFSS interface** but lowers:

* `PredProgramDesc` → **grouped DCF keys** (still “one DCF per threshold”), evaluated in a *vectorized batch kernel* with SoA layout.
* `CoeffProgramDesc(kStepDcf)` → evaluate step cuts in **vectorized form** (multiple cuts per loop, fused decoding + accumulation).

This already yields large wall‑clock improvements because:

* no per-key parsing per element
* one tight loop over AES calls (or myl7 calls) using SoA, good cache reuse
* amortized overhead per query/cut

### Phase B (SIGMA-like core, fewer traversals)

Replace Phase A’s “many DCFs” with true **packed multi-threshold compare** and **interval LUT**:

* multi-output DCF / CDPF for comparisons that returns a payload bitmask (e.g., 64 thresholds per call)
* vector‑payload DPF/DCF for interval LUT (one program eval returns `out_words` u64 shares)

Phase B is the “paper-level” Sigma-like improvement (fewer traversals and better constants).

**Milestone 7 completion means Phase B is implemented**, but Phase A is a great stepping stone for correctness + benchmarking.

---

## 7.2 Packed comparisons (Pred program)

Your compiler emits `RawPredQuery` of:

* `LT_U64(theta)`
* `LTLOW(f, theta)` for several f (notably `f=12` and maybe `f=63/64` if derived)

### 7.2.1 Keygen lowering (Phase B)

Introduce a SigmaFast internal representation that groups queries:

* bucket by `(kind, f)`
* inside each bucket, sort thresholds (stable mapping back to query indices)

Then generate **packed compare keys** per bucket:

* pick a packing width `W = 64` thresholds per packed key (aligns great with `u64` masks)
* each packed key returns `ceil(W/64)=1` output word (a bitmask share)
* if `Tbucket > 64`, produce `G = ceil(Tbucket/64)` packed keys

**New internal object:**

```cpp
struct PackedCmpBucketKey {
  RawPredKind kind;
  uint8_t f;
  uint32_t num_groups;  // ceil(T/64)
  // Each group has one packed-DCF key blob for party b
  std::vector<Bytes> group_keys_b;
  // thresholds[group][0..<=64) for debug/validation only (not online)
};
```

### 7.2.2 Online eval lowering (Phase B)

For each bucket:

* evaluate each group key once per element
* output bitmask shares to `out_masks[N][num_groups]`
* then expand to `u64PerBit` if needed **only if the gate layer requires it**

    * Prefer to keep masks packed in SigmaFast, and teach your boolean DAG evaluator to read packed bits.

**Deliverable:** Add an optional `PredOutMode::kPackedMask`:

* output is `u64 masks[ceil(Q/64)]` per element (additive shares of bitmasks)
* This is what SIGMA-like pipelines want.

---

## 7.3 Vector-payload interval LUT (Coeff program)

Your compiler already supports:

* `CoeffMode::kIntervalLut`
* `CoeffMode::kStepDcf`

Milestone 7 should implement **kIntervalLut** in SigmaFast.

### 7.3.1 Recommended representation

Implement an **interval-LUT FSS** that returns the payload vector directly.

Input: `hatx ∈ [0,2^64)`
Intervals: disjoint non-wrapping `[lo, hi)` (your compiler already splits wrap)

Output: `payload_words[out_words]` as additive shares in `Z_2^64`.

### 7.3.2 Practical implementation approach

To keep implementation feasible, implement interval LUT as:

1. Compute secret-shared **interval index** using packed compares on the boundaries (prefix structure), then
2. Select payload vector using a small selection network

This is “Option 1” from your earlier outline, but engineered to be fast for `m <= 8..16`, which matches transformer splines.

**How it works (efficiently):**

* Let boundaries `b0 < b1 < ... < b_{m-1}` in masked domain.
* Compute bits `c_i = 1[hatx < b_i]` for all i (this is exactly what packed compare gives you).
* Convert these into a one-hot interval selector `s_k` (secret-shared).
* Compute:
  [
  payload = \sum_k s_k \cdot payload_k
  ]
  This requires multiplications of secret bits with u64 words. But:
* `m` is small (typical 8), and `out_words` is moderate (e.g., `r*(d+1)` with `r<=2`, `d<=3` ⇒ 8–16 words).
* With your **batched Beaver**, this is a small fixed overhead.

**When to switch to “true 1-call LUT FSS”:**

* if you want bigger m (softmax exp tables / reciprocal mantissa tables) and larger vectors, you can later implement a real vector-payload DPF. But for Milestone 7, the “packed compare + selection network” is implementable and already Sigma-like.

**Keygen output for kIntervalLut mode in SigmaFast:**

* packed compare keys for the boundaries
* payload vectors as additive shares stored in key (per interval)
* plus beaver triples needed for `s_k * payload_words` multiplications (dealer precomputes)

---

## 7.4 CPU performance engineering (what to build)

### 7.4.1 Crypto/PRG core

Create `include/crypto/`:

* `aes128_ni.hpp` (AES-NI blocks, fixed-key mode)
* `prg.hpp` (expand state -> two children per tree level)
* `block.hpp` (128-bit block abstraction)

### 7.4.2 SoA layout requirements

Your current SoA packers are good; lock these rules for SigmaFast:

* keys stored as contiguous arrays per level, per group, per party
* evaluation processes batches of inputs in blocks (e.g., 1024 or 4096) to fit L2 cache

### 7.4.3 Threading model

Add a backend‑internal parallel loop for `eval_many`:

* partition inputs into chunks per thread
* each thread has its own scratch buffer for PRG states
* avoid false sharing on output (per-thread output slices)

---

## 7.5 Milestone 7 test + benchmark suite (must exist)

### Correctness tests (must)

* `test_sigmafast_pred_equiv`: SigmaFast pred outputs == ClearBackend outputs for random programs/inputs
* `test_sigmafast_coeff_equiv`: SigmaFast coeff outputs == ClearBackend (both intervalLut mode and stepDcf mode)
* `test_gate_equiv_sigmafast`: run compiled GeLU/ReluARS with SigmaFast and compare reconstructed outputs to ref_eval/plaintext (you already have references)

### Benchmarks (must)

Add `src/bench/`:

* `bench_pred.cpp`:

    * measure ns/element vs number of thresholds (16, 32, 64, 128)
    * measure also `LTLOW(f=12)` buckets
* `bench_coeff.cpp`:

    * m=8, out_words=8..32, compare selection-network LUT vs stepDcf baseline
* `bench_gates.cpp`:

    * ReluARS, GeLU at N=1e6 elements

**Definition of done (Milestone 7):**

* SigmaFast passes all correctness tests
* Bench shows **clear gains** vs Myl7 baseline:

    * packed pred: ≥3–10× faster for `T≈32..64`
    * coeff LUT: significantly faster than stepDCF at typical spline sizes
    * gate throughput scales with threads and beats baseline end-to-end

---

# Milestone 8 — Generic Composite‑FSS Gate Runtime (auto-generated gates from SUF)

You already have:

* SUF reference semantics + mask rewrite
* SUF→PFSS compiler producing (PredDesc, CoeffDesc, BoolDag, coeff payload layout)
* working hand-coded ReluARS/GeLU evaluators

Milestone 8 makes the runtime *systematic*:

* a generic `CompGen` and `CompEvalBatch` that can run any compiled SUF gate
* ReluARS/GeLU become “spec + postproc”, not handwritten PFSS wiring

---

## 8.1 Core runtime API (batch-first)

Create:

### `include/gates/composite_fss.hpp`

Key abstractions:

```cpp
struct GateKeyParty {
  // Masks:
  std::vector<u64> r_in_share;    // size N or streamed
  std::vector<u64> r_out_share;   // size N*out_r (or per-instance)

  // Backend keys + meta for pred/coeff programs (SoA-capable)
  pfss::Bytes pred_keys_flat;
  pfss::PredKeyMeta pred_meta;

  pfss::Bytes coeff_keys_flat;
  pfss::CoeffKeyMeta coeff_meta;

  // Beaver resources (either as vectors or tape cursors)
  std::vector<BeaverTriple64Share> triples64;

  // Optional: additional per-gate constants
  // (ReluARS delta table, trunc parameters, etc.)
};

struct GateBatchInput {
  const u64* hatx_public; // N public values
  std::size_t N;
};

struct GateBatchOutput {
  // masked arithmetic outputs as shares in Z_2^64
  std::vector<u64> haty_share; // size N*out_r
  // boolean helper outputs as additive 0/1 u64 shares
  std::vector<u64> bout_share; // size N*out_ell
};
```

And two functions:

* `CompGenBatch(...) -> (GateKeyParty k0, GateKeyParty k1)`
* `CompEvalBatch(party, backend, channel, keyParty, input) -> output`

This becomes the universal gate engine.

---

## 8.2 CompGenBatch (dealer/offline) integrates compiler + backend + beaver layout

Implement:

### `gates/compgen_batch.cpp`

Steps per gate type:

1. Choose masks:

    * sample `r_in` per instance (or per tensor) and share it into parties
    * sample `r_out[j]` per output channel and share it similarly
2. Compile SUF once per gate type **or** compile per-instance if masks differ.
   Practical approach:

    * compile descriptors per instance because `r_in` changes thresholds/cuts
    * BUT cache the unmasked SUF structure, only rewrite constants per instance
3. Backend proggen:

    * `prog_gen_pred(desc_i)` → append party keys to `pred_keys_flat`
    * `prog_gen_coeff(desc_i)` → append party keys to `coeff_keys_flat`
4. Allocate triples:

    * arithmetic Horner: `N * out_r * degree` muls (or reuse x powers to reduce)
    * BoolDag evaluation: count ANDs and required mults depending on your boolean representation
5. Write tapes:

    * GateKeyParty should be serializable (and streamable)
    * integrate with your tape module (already stable)

**Definition of done for CompGenBatch:**

* Given a SUF, it can produce party keys and tapes for any N
* running CompEvalBatch reconstructs to ref_eval outputs

---

## 8.3 CompEvalBatch (online) is a fixed pipeline

Implement:

### `gates/compeval_batch.cpp`

Pipeline (per party):

1. **Masked→shares (vectorized, no comm):**

    * `x_share[i] = (party==0) ? (hatx[i] - r_in_share[i]) : (0 - r_in_share[i])`

2. **PFSS pred eval (batched):**

    * backend `eval_pred_many(pred_keys_flat, ..., hatx[N]) -> pred_out_flat`
    * decode to either:

        * `u64 per bit` shares (baseline)
        * or `u64 bitmasks` (SigmaFast), with adapters for BoolDag

3. **BoolDag evaluation (batched):**

    * evaluate all Boolean outputs `B(x)` using pred bits and wrap shares
    * output `bout_share[N * ell]`

4. **PFSS coeff eval (batched):**

    * backend `eval_coeff_many` -> `coeff_words_share[N * out_words]`

5. **Polynomial evaluation (Horner) (batched Beaver):**

    * For each output channel j:

        * y = Horner(coeffs[j], x_share) using `BeaverMul64Batch`
    * This should be implemented as:

        * one `mul_batch` per Horner stage across all elements (great locality)

6. **Add output masks:**

    * `haty_share = y + r_out_share`

That’s the generic SUF gate path.

---

## 8.4 Handling “post-processing” gates (ReluARS) cleanly

Some gates are SUF + extra arithmetic logic (truncation + LUT correction). You want Milestone 8 to support this without reintroducing bespoke PFSS logic.

Introduce an optional hook:

```cpp
struct PostProc {
  virtual void run_batch(
      int party,
      IChannel& ch,
      BeaverMul64Batch& mul,
      // inputs:
      const u64* x_share,
      const u64* hatx_public,
      const u64* suf_arity_share, // e.g., x_plus share, or other SUF arith channels
      const u64* suf_bool_share,  // helper bits w/t/d shares etc.
      std::size_t N,
      // in/out:
      u64* haty_share_out) const = 0;
};
```

* **GeLU**: postproc is trivial (`y = x_plus + delta`), can be folded into generic linear combine (no hook needed).
* **ReluARS**: postproc implements:

    * q = ARS(x_plus, f=12) using t-bit, r_hi shares, and your existing truncation circuit
    * delta correction LUT8 indexed by (w,t,d) shares
    * add mask

The key point: PFSS work is still generic; only the final arithmetic combination is special.

**Definition of done (Milestone 8):**

* ReluARS and GeLU both run through Composite runtime, matching your existing hand-written outputs bit-for-bit under identical approximations.

---

## 8.5 Equivalence testing strategy (must be explicit)

Add a test binary:

### `src/demo/test_composite_runtime_equiv.cpp`

For each gate (ReluARS, GeLU):

* generate N random inputs
* run old (handwritten proto) path
* run new Composite runtime path (compiled SUF + backend)
* verify reconstructed outputs match exactly

Run under:

* ClearBackend
* Myl7FssBackend
* SigmaFastBackend (once Milestone 7 is done)

---

# Summary: Milestone 7–8 “Definition of Done”

### Milestone 7 (SigmaFastBackend)

* real packed pred implementation (bitmask output supported)
* coeff interval LUT fast path (at least packed-compare + selection network)
* microbench + gate bench shows strong speedups vs baseline
* all correctness tests pass vs ClearBackend / ref_eval

### Milestone 8 (Composite runtime)

* generic CompGen/CompEvalBatch for compiled SUF gates
* ReluARS/GeLU implemented as SUF+postproc (no bespoke PFSS wiring)
* strict equivalence tests vs old per-gate evaluators under all backends

If you want, I can also propose a **data-structure exact layout** for `SigmaFastBackend` packed keys (per-level arrays, per-group correction words, SoA strides) so you can implement Phase B without redesigning halfway through.
