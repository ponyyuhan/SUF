## Milestone 0 — Freeze the contract (so every later module is consistent)

### 0.1 Parameters (hard-coded first, config later)

* Ring: `R = Z_2^64` (`uint64_t`)
* Fixed-point: choose `f` (start with `f=12` like many FSS systems; allow later config)
* Threat model: semi-honest, preprocessing with dealer (as in your writeup)
* Model privacy (decide early because it changes linear layers):

    * **Option A (SIGMA-like common)**: weights are **public** to both servers; only activations are secret-shared.
    * **Option B**: weights are secret-shared; requires **matrix Beaver triples** for every GEMM (offline blows up).

**Definition of done**: a single `params.h` that is included everywhere, and unit tests compile with it.

---

## Milestone 1 — Core runtime primitives (clean, typed, testable)

This replaces “one-off code in a harness” with a reusable core.

### 1.1 Core types

Create:

* `core/ring64.h`

    * `using u64 = uint64_t;`
    * `add/sub/mul` mod `2^64` (wraparound)
* `core/bytes.h`

    * little-endian pack/unpack functions for `u64`
* `core/shares.h`

    * `struct ArithShare64 { u64 s; };` (additive in `Z_2^64`)
    * `struct XorShare64 { u64 s; };` (bitwise XOR share for packed predicate bits)

Why XOR shares? Predicate outputs are bits; XOR-sharing is “native” and avoids 64× bloat.

### 1.2 Channels

* `net/ichannel.h`
* `net/mem_channel.h` (your in-process duplex)
* `net/tcp_channel.h` (later; not blocking now)

### 1.3 Beaver multiplication (batched)

* `mpc/beaver64.h`

    * `mul_batch(x[], y[]) -> z[]` with **one open (e,f) round** per batch, not per multiply.
* `mpc/beaver_bit.h` (optional but recommended)

    * XOR-share AND using bit triples.

**Definition of done**

* Unit tests:

    * `test_beaver64_roundtrip`: random `x,y` => reconstruct `x*y`
    * `test_beaver64_batch_equivalence`: batched equals scalar
* Your existing sim harness should migrate to these types.

---

## Milestone 2 — Preprocessing “tape” format (streamable offline artifacts)

LLM inference needs huge preprocessing. If you don’t stream, you’ll die on memory.

### 2.1 Tape records

Implement:

* `offline/tape_writer.h`, `offline/tape_reader.h`

Record format (simple, robust):

* `[u32 type][u32 bytes][payload...]`
  Types:
* `MASK64` (one `u64` share)
* `FSS_KEY` (blob)
* `TRIPLE64` (`a,b,c` shares)
* (optional) `TRIPLEBIT`

### 2.2 Deterministic consumption order

Every online op must consume from tape in a **deterministic sequence**:

* gate instance i consumes: pred-key, coeff-key, triples, r_in share, r_out share, etc.

**Definition of done**

* `test_tape_roundtrip` writes random records and reads identical bytes.
* A “tape replay” test: run online twice from same tape => identical outputs.

---

## Milestone 3 — SUF IR (the real “compiler IR”, not handwritten gates)

### 3.1 SUF descriptor objects

Files:

* `suf/ast.h`
* `suf/ir.h`
* `suf/ref_eval.h`

Data model:

* `struct Poly { vector<u64> coeff; /* size d+1 */ };`
* `struct PolyVec { vector<Poly> outs; /* size r */ };`
* `struct BoolExpr` AST with primitives:

    * `LT(beta)` meaning `1[x < beta]`
    * `LTLOW(f, gamma)` meaning `1[x mod 2^f < gamma]`
    * `MSB(c)` meaning `MSB(x+c)`
    * plus `NOT/AND/OR/XOR/CONST`
* `struct SUF { int n, r, ell, d; vector<u64> alpha; vector<PolyVec> P; vector<vector<BoolExpr>> B; }`

### 3.2 Reference evaluator (ground truth)

`(y_vec, b_vec) = suf_eval_ref(SUF, x)` in cleartext.

**Definition of done**

* Unit tests for closure sanity (cartesian product etc. optional).
* Property test on random SUFs (small n for exhaustive). Even `n=8` exhaustive is super valuable.

---

## Milestone 4 — Mask-aware predicate rewrite engine (§3.3 as code)

This is critical for correctness; your gates will be wrong without it.

### 4.1 Rewrite API

`suf/mask_rewrite.h`:

* `BoolExpr rewrite_under_mask(const BoolExpr& e, u64 r_in, int n);`
* output uses only primitives on `hatx`:

    * `LT_hat(theta)`
    * `LTLOW_hat(f, theta)`
    * `MSB_hat(c')` (where `c' = c - r_in` in `Z_2^64`, but stored implicitly as hidden constants inside PFSS keys)

### 4.2 Correctness testing

Property test:

* sample `x, r`
* let `hatx = x + r`
* check: `eval(e, x) == eval(rewrite(e,r), hatx)` (where rewrite-eval uses rewritten primitives)

**Definition of done**

* 100k random tests pass for each primitive class:

    * comparisons (wrap/no-wrap)
    * low-bit comparisons
    * MSB shifts

---

## Milestone 5 — SUF → PFSS compiler skeleton (two-program output)

This milestone turns SUF into the objects your backend can keygen/eval.

### 5.1 Program descriptors (backend-agnostic)

Create:

* `compiler/pfss_programs.h`

Define:

* `struct PredProgramDesc { int n; vector<PredicatePrimitive> prims; /* thresholds incl. low-bit */ };`
* `struct CoeffProgramDesc { int n; vector<IntervalPayload> intervals; int out_words; };`

    * `IntervalPayload { u64 lo, hi; vector<u64> payload_words; }` describing `[lo,hi)` in **masked domain**

### 5.2 Compiler entry point

`compiler/suf_to_pfss.h`:

* `CompiledGate compile_suf_to_pfss(const SUF& F, u64 r_in, vector<u64> r_out);`
  Where `CompiledGate` contains:
* masked SUF metadata,
* `PredProgramDesc`,
* `CoeffProgramDesc`.

**Definition of done**

* For a random SUF:

    * compile
    * simulate PFSS by directly evaluating desc on cleartext hatx
    * confirm outputs match masked SUF definition.

(Still no real DPF/DCF yet—this is a compiler correctness milestone.)

---

## Milestone 6 — PFSS backend “baseline” + myl7 adapter (correctness first)

You want two backends:

* `Myl7FssBackend` (open-source adapter)
* `SigmaFastBackend` (high-efficiency)

Start with a baseline you can trust.

### 6.1 Backend interface (typed but compatible with “bytes-out”)

`pfss/pfss_backend.h`:

* `gen_lt(n, alpha, out_bytes)` / `eval_lt(n, key, x)`
* `gen_interval_lut(n, intervals, out_bytes)` / `eval_interval_lut(...)`
* plus batched eval: `eval_many_u64(...)`

Result is still “bytes-out”, but gate code will reinterpret:

* predicate output bytes → packed XOR bitmasks (`u64`)
* coeff output bytes → `u64[]` additive shares

### 6.2 Myl7 adapter skeleton

`pfss/myl7_backend.h/.cc`:

* define exact mapping:

    * input encoding: MSB-first bits (or u64 input; pick one and stick to it)
    * output group law: XOR for bitmasks, addition mod `2^64` for coeffs
    * key blob layout and SoA pack/unpack

**Definition of done**

* Backend test suite runs with:

    * `ClearPfssBackend` (your fake backend)
    * `Myl7FssBackend` (even if slow)
* Same gate tests pass under both.

---

## Milestone 7 — SigmaFastBackend (what makes it SIGMA-like)

This is the *performance* milestone. Do CPU first; GPU is a separate milestone.

### 7.1 CPU packed comparisons

Goal: produce many `1[hatx < theta_j]` bits at high throughput.

Concrete approach (implementation-feasible, SIGMA-like):

* implement a **comparison DCF/CDPF** core for 64-bit domains
* batch evaluations across many instances and many thresholds using:

    * SoA key layout
    * AES-NI (or ChaCha) PRG expansions
    * wide-vector pipelining (AVX2/AVX-512)

API:

* `gen_packed_lt(n=64, thresholds[T], pack_bits=true) -> keypair`
* `eval_packed_lt_many(xs[N]) -> out_bitmask_words[N][ceil(T/64)]`

### 7.2 CPU vector-payload interval LUT

Two realistic options:

**Option 1 (fast to implement, good performance when m small ≤ 8–16):**

* represent interval LUT via a packed-compare + **2–4 round selection network** on secret-shared interval id
* dealer stores **all coefficients as shares**, selection multiplies indicators with coeff shares
* acceptable when `m*(d+1)*r` small (typical splines).

**Option 2 (best asymptotics / closest to “1 PFSS call for coeffs”):**

* implement a real **interval-LUT FSS** (vector payload) that outputs the correct payload directly.
* more complex; but this is the cleanest match to your §3.4.3.

**Definition of done**

* Microbenchmarks:

    * packed comparisons: throughput in “evals/sec” for `T≈16..64`
    * coeff LUT: throughput for `m<=8, d<=3` vectors
* Gate-level end-to-end benchmarks:

    * ReluARS and GeLU batched over `N = 1e6` elements.

---

## Milestone 8 — Generic Composite-FSS gate runtime (auto-generated gates)

Right now you have hand-written gate logic. Now you make it *systematic*.

### 8.1 `CompGen` and `CompEvalBatch`

Files:

* `gates/composite_gate.h/.cc`

`CompGen`:

* consumes SUF + masks
* uses compiler outputs to keygen backend programs
* allocates the exact Beaver triples needed

`CompEvalBatch`:

* batch PFSS eval for pred and coeff
* compute helper bits / boolean formulas (XOR shares)
* evaluate polynomial via Horner (arith shares + batched Beaver)
* mask outputs

**Definition of done**

* ReluARS and GeLU are produced *from SUF specs*, and match old hand-written versions bit-for-bit (under same approximations).

---

## Milestone 9 — Add the missing LLM gates (minimum set)

To run transformer inference you need:

* `SiLU` (spline)
* `exp` approx (nExp / exp2 / clipped exp)
* `reciprocal` (softmax normalization)
* `rsqrt` (LayerNorm)
* Vector blocks:

    * `SoftmaxBlock` (max, exp, sum, reciprocal, multiply)
    * `LayerNorm` (mean, variance, rsqrt, affine)

**Definition of done**

* Each gate has:

    * cleartext ref
    * 2PC correctness test with random inputs
    * batched benchmark

---

## Milestone 10 — Linear algebra + attention (the true LLM core)

### 10.1 GEMM engine

If weights public: local GEMM on shares (fastest).
If weights private: **matrix Beaver triples** (mandatory).

Implement:

* `linear/matmul_publicW.cc` and/or `linear/matmul_beaver.cc`
* GPU path: cuBLAS for local products + overlap with comm

### 10.2 Attention + KV cache

Implement:

* QKV projections
* attention scores
* softmax block
* output projection
* KV cache in shares/masked format

**Definition of done**

* One transformer layer end-to-end runs correctly vs plaintext within approximation error.

---

## Milestone 11 — Full runtime: scheduling, batching, CPU/GPU strategy

This is where “works” becomes “SIGMA-like”.

### 11.1 Scheduling rules

* Batch PFSS evals per layer (maximal batching)
* Batch BeaverMul opens per layer (minimize round trips)
* Fuse where beneficial (ReluARS style)

### 11.2 CPU vs GPU split (practical)

A highly practical SIGMA-like split is:

* GPU: GEMMs (cuBLAS), reductions if safe
* CPU: PFSS/CDPF (AES-NI), Beaver opens, control
* Overlap comm with GPU compute via streams/threads

**Definition of done**

* Measure layer latency breakdown and show that PFSS and comm are not dominating.

---

# What we do *next* (the “Step 1” you should implement immediately)

If you want the most rigorous path, implement **Milestone 1 + Milestone 2 first**, because everything else depends on them:

1. `core/*`, `net/*`, `mpc/beaver64.*` (batched)
2. `offline/tape_*` (streaming)
3. migrate your current harness into `tests/test_reluars_gelu_sim.cc`

Once that foundation is stable, Milestones 3–5 (SUF IR + rewrite + compiler) become straightforward and testable.

