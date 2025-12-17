Great — with Milestones **1–4** completed (and the wrap leak fixed correctly via **additive shares + MPC SEL**), you’re ready for the “real system” steps:

* **Milestone 5:** Make SUF compile into *backend-consumable* PFSS programs (two-program output: **Pred** + **Coeff**), and prove compiler correctness against your SUF reference semantics.
* **Milestone 6:** Implement a *baseline* PFSS backend that can actually keygen/eval those programs using (a) Clear backend and (b) a real adapter (myl7/fss), plus keep a SigmaFast stub under the same interface.

Below is a detailed, concrete, repo-aligned plan.

---

# Milestone 5 — SUF → PFSS Compiler (two-program output, compiler correctness first)

## 5.0 Goal and non-goals

### Goal

Turn a validated `SUF` + masks `(r_in, r_out[])` into **exactly two PFSS program descriptions**:

1. `PredProgramDesc`: “compute all primitive bits needed by this SUF **on masked input** (\hat x)”
2. `CoeffProgramDesc`: “given (\hat x), return additive shares of the active interval’s polynomial coefficients” (vector payload)

Then ensure that, with a cleartext simulator backend, evaluating these two programs + local post-processing reconstructs **exactly** the masked SUF output:
[
\widehat F(\hat x) = (P(x)+r_{out},; B(x)), \quad x=\hat x-r_{in}.
]

### Non-goals (for Milestone 5)

* No real cryptography performance yet.
* No packed multi-threshold compare yet (that’s Milestone 6+/SigmaFast).
* No model-wide graph integration yet (just gate-level correctness).

---

## 5.1 Canonical program descriptors (backend-agnostic)

Create a new header (location consistent with your tree):

### `include/compiler/pfss_program_desc.hpp`

Define *minimal* descriptors that preserve structure and future optimization hooks.

```cpp
namespace compiler {

// A “raw compare bit” the backend can directly evaluate on public hatx.
enum class RawPredKind : uint8_t {
  kLtU64,     // b = 1[hatx < theta] in Z_2^64 (unsigned)
  kLtLow,     // b = 1[(hatx mod 2^f) < theta] (f<=64)
};

// One raw predicate query => one output bit (in your current arithmetic-bit domain).
struct RawPredQuery {
  RawPredKind kind;
  uint8_t f;        // only used when kind==kLtLow
  uint64_t theta;   // threshold in that domain
};

// Output packing choice (keep simple now, future-proof later).
enum class PredOutMode : uint8_t {
  kU64PerBit,   // output words = num_bits, each word is 0/1 share
  // kPackedBitsXor, // optional future mode (SigmaFast): packed XOR shares
};

struct PredProgramDesc {
  int n = 64;
  PredOutMode out_mode = PredOutMode::kU64PerBit;
  std::vector<RawPredQuery> queries;  // deduplicated, stable order
};

// Coeff LUT can be expressed in two backends:
// (A) IntervalLUT (fast path), (B) Step-DCF (baseline path)
enum class CoeffMode : uint8_t { kIntervalLut, kStepDcf };

struct IntervalPayload {
  uint64_t lo;   // inclusive
  uint64_t hi;   // exclusive, with lo<hi in unsigned, no wrap (already split)
  std::vector<uint64_t> payload_words; // size = out_words
};

struct CoeffProgramDesc {
  int n = 64;
  CoeffMode mode;
  int out_words = 0;  // e.g. r*(d+1)
  // mode == kIntervalLut:
  std::vector<IntervalPayload> intervals;

  // mode == kStepDcf:
  // base_payload + sum_{i} DCF_i(hatx) where DCF_i outputs delta[i] if hatx >= cut[i], else 0
  std::vector<uint64_t> base_payload_words;         // size out_words
  std::vector<uint64_t> cutpoints_ge;              // sorted ascending
  std::vector<std::vector<uint64_t>> deltas_words; // deltas_words[i].size()==out_words
};

} // namespace compiler
```

**Rationale**

* `PredProgramDesc` retains the ability to later be lowered to:

    * many single DCFs (baseline)
    * a packed multi-threshold compare (SigmaFast)
* `CoeffProgramDesc` supports both:

    * robust baseline (`StepDcf`, which maps to what you’re already doing)
    * fast interval LUT (what SigmaFast should implement)

---

## 5.2 Compiler output object

Create:

### `include/compiler/compiled_suf_gate.hpp`

```cpp
namespace compiler {

struct CompiledBoolDag; // your compiled boolean DAG format (or reuse existing BoolExpr with indices)

struct CompiledSUFGate {
  uint64_t r_in;                    // dealer-only
  std::vector<uint64_t> r_out;      // dealer-only, size r

  PredProgramDesc pred;
  CoeffProgramDesc coeff;

  // Mapping from SUF boolean outputs B_i(x) to expressions over pred bits (+ wrap shares if used)
  // Keep this in a normal form that is easy to evaluate on shares.
  CompiledBoolDag bool_dag;

  // Metadata: polynomial degree d, output arity r, etc.
  int degree = 0;
  int r = 0;
  int ell = 0;
};

} // namespace compiler
```

---

## 5.3 Integration point: SUF → PFSS compilation entry point

Create:

### `include/compiler/suf_to_pfss.hpp`

```cpp
namespace compiler {

CompiledSUFGate compile_suf_to_pfss_two_programs(
    const suf::SUF& F,         // your SUF IR object
    uint64_t r_in,
    const std::vector<uint64_t>& r_out,
    CoeffMode coeff_mode /*default StepDcf for baseline*/);

} // namespace compiler
```

---

## 5.4 Exact compilation algorithm (rigorous)

### Step 0: validate

* `suf::validate(F)` must pass (you already have).

### Step 1: collect SUF primitives

Walk:

* all BoolExpr nodes in `B_i(x)` for all intervals
* all interval boundary tests needed for SUF semantics (if any)
  Collect unique primitives in a normalized form:
* `LT(beta)`
* `LTLOW(f, gamma)`
* `MSB_ADD(c)` (meaning MSB(x+c))

### Step 2: apply mask rewrite recipes (§3.3 → your `mask_rewrite.hpp`)

For each primitive `p(x)` produce a **masked boolean expression** over raw predicates on `hatx`.
This uses your already-tested recipes:

* `rewrite_lt(r_in, beta)` → requires `hatx<theta0`, `hatx<theta1` plus wrap-share (already fixed as secret)
* `rewrite_ltlow(r_in, f, gamma)` → requires `hatlow<theta0`, `hatlow<theta1` plus wrap-share
* `rewrite_msb_add(r_in, c)` → also reduces to a rotated interval membership of length `2^63`, so same pattern

**Important compiler invariant:** the result of rewriting must be **a pure BoolExpr over raw predicates**, plus optional *secret-shared constant bits* (wrap shares). Those wrap shares must be emitted as “dealer constants” to be included in party keys later.

### Step 3: deduplicate raw queries and assign stable indices

You now have a set of raw queries of the form:

* `LT_U64(theta)` and `LTLOW(f, theta)`

Deduplicate:

* key by `(kind,f,theta)`
  Assign each query an index `qid ∈ [0..Q-1]` in a stable order:
* first all `LT_U64` sorted by theta (or in encounter order)
* then per `f`, all `LTLOW` sorted by theta

These indices define the output layout of `PredProgramDesc`.

### Step 4: re-express rewritten predicates and B(x) in terms of these indices

Replace each raw predicate with `PredBit(qid)` nodes.
Compile the rewritten expressions into your `CompiledBoolDag` format.

**Strongly recommended normal form**
Represent boolean formulas as an **arithmetic-bit DAG over Z_2^64** (0/1 shares), since that matches your existing `bit_ring_ops.hpp` / SEL logic:

* NOT: `1 - b`
* XOR: `a + b - 2ab` is expensive; instead keep XOR as `a ^ b` only if you’re in XOR-sharing.
  Given you currently use *additive-bit* domain, use these ops:
* AND: multiplication (Beaver64 or BeaverBit + conversion)
* OR: `a + b - a*b` (one multiplication)
* XOR: `a + b - 2ab` (one multiplication + extra add/sub)

To keep Milestone 5 testable without MPC, you can store the **logical structure** and evaluate it in plaintext in `ref_eval`, then later you’ll choose the MPC realization cost model.

### Step 5: compile coefficient program

Your SUF provides per-interval polynomial coefficients:

* `a_{i,j,k}` for interval `i`, output `j`, power `k`.

Define payload words for interval `i`:

* size = `out_words = r*(degree+1)`
* recommended layout for Horner: store **descending powers** per output:

    * `[(a_{j,d}, a_{j,d-1}, ..., a_{j,0}) for j=0..r-1]`

Now express the LUT selection **in masked domain (\hat x)**.

#### 5.5.1 Canonical “split-wrap” interval construction (no leakage)

For each original interval `[α_i, α_{i+1})` in **unmasked** domain:

* compute `lo = α_i + r_in`
* compute `hi = α_{i+1} + r_in`
  All mod 2^64.

If `lo < hi` (unsigned): one masked interval `[lo, hi)`
If `lo >= hi`: split into

* `[lo, 2^64)` and `[0, hi)`

Collect all such pieces into a list, each carrying payload of original interval `i`.
Finally, sort by `lo` and verify they are disjoint / cover exactly once as expected for a partition (for safety).

This yields a non-wrapping interval list suitable for `IntervalLut` mode.

#### 5.5.2 Step-DCF mode construction (baseline)

If using `CoeffMode::kStepDcf`:

* Sort masked interval boundaries into ascending cutpoints:

    * `cut[0..M-2]` = boundaries between intervals in masked domain (after split)
* Choose `base_payload = payload of first interval in sorted order`.
* For each subsequent interval boundary `cut[t]`, compute delta:

    * `delta[t] = payload(interval_{t+1}) - payload(interval_t)` wordwise mod 2^64

Then define:
[
payload(\hat x)=base + \sum_{t} \mathbf{1}[\hat x \ge cut[t]] \cdot delta[t]
]
But **avoid MPC multiplications** by programming each step as a vector-payload DCF that outputs either `delta[t]` or `0`. That matches your existing step-DCF pipeline.

**Compiler emits:** `base_payload_words`, `cutpoints_ge`, `deltas_words`.

---

## 5.5 Definition of Done (Milestone 5) — compiler correctness tests

Add a new test binary:

### `src/demo/test_suf_to_pfss_compile.cpp`

Test plan:

1. Generate random SUFs (small `n=8` and real `n=64`)
2. Sample random masks `r_in`, `r_out[]`
3. Compile: `compile_suf_to_pfss_two_programs(F, r_in, r_out, StepDcf)`
4. For random `x`:

    * compute `hatx = x + r_in`
    * **simulate pred program** in plaintext:

        * evaluate each raw query on `hatx` → bits
    * **simulate bool_dag** in plaintext:

        * compute boolean outputs
    * **simulate coeff program**:

        * evaluate step/interval selection in plaintext to obtain coefficient payload
    * evaluate polynomials at `x` → y
    * check equals masked SUF: `ref_eval_masked(F, r_in, r_out, hatx)`

This proves:

* mask rewrite integration is correct (§3.3)
* coefficient rotation/splitting is correct
* end-to-end compiler semantics match SUF reference

---

# Milestone 6 — PFSS backend baseline + Myl7 adapter + SigmaFast stub

## 6.0 Goal

Provide a backend layer that can:

* `ProgGen(PredProgramDesc)` → `(key0,key1)`
* `Eval(PredKey_b, hatx)` → shares of pred outputs
* `ProgGen(CoeffProgramDesc)` → `(key0,key1)`
* `Eval(CoeffKey_b, hatx)` → additive `u64` coefficient shares

And make both:

* your **Clear backend**
* your **Myl7 adapter**
  pass the **same test suite**:
* backend primitive tests
* compiled gate tests (ReluARS/GeLU via compilation or via current proto evaluators)
* harness end-to-end

---

## 6.1 Canonical PFSS interface (typed but still “bytes-out”)

Create (or extend) a single header for the non-proto layer:

### `include/pfss/pfss_backend.hpp`

Key idea: keep `FssKey` as `std::vector<uint8_t>` (bytes-in/bytes-out), but include **meta** so decoders know how many words/bits.

```cpp
namespace pfss {

using Bytes = std::vector<std::uint8_t>;

struct KeyPair { Bytes k0, k1; };

struct PredKeyMeta {
  int n = 64;
  compiler::PredOutMode out_mode;
  std::size_t out_u64_words; // if kU64PerBit => out_u64_words = num_bits
};

struct CoeffKeyMeta {
  int n = 64;
  compiler::CoeffMode mode;
  int out_words;
  std::size_t num_cuts; // step mode
};

// Backend API:
struct PfssBackend {
  virtual ~PfssBackend() = default;

  virtual KeyPair prog_gen_pred(const compiler::PredProgramDesc& desc, PredKeyMeta* meta_out) const = 0;
  virtual KeyPair prog_gen_coeff(const compiler::CoeffProgramDesc& desc, CoeffKeyMeta* meta_out) const = 0;

  // Eval returns raw bytes; caller decodes into u64 shares using canonical little-endian.
  virtual Bytes eval_pred(const Bytes& key_b, const PredKeyMeta& meta, std::uint64_t hatx) const = 0;
  virtual Bytes eval_coeff(const Bytes& key_b, const CoeffKeyMeta& meta, std::uint64_t hatx) const = 0;

  // Batched (CPU SIMD / CUDA-friendly): same meta for all instances.
  virtual void eval_pred_many(const Bytes& keys_flat, std::size_t key_stride,
                             const PredKeyMeta& meta,
                             const std::uint64_t* hatx, std::size_t N,
                             std::uint8_t* out_flat, std::size_t out_stride) const = 0;

  virtual void eval_coeff_many(const Bytes& keys_flat, std::size_t key_stride,
                               const CoeffKeyMeta& meta,
                               const std::uint64_t* hatx, std::size_t N,
                               std::uint8_t* out_flat, std::size_t out_stride) const = 0;
};

} // namespace pfss
```

**Why this matches your current system**

* It’s still “bytes-out”.
* It supports SoA packing (`keys_flat`, `out_flat`) directly for SIMD/GPU scheduling.
* It allows different internal implementations (Clear, myl7, SigmaFast).

---

## 6.2 Baseline backend lowering strategy (correctness-first)

Implement:

### `pfss/ClearPfssBackend`

* For `prog_gen_pred`: serialize queries into key (plaintext ok), return `(k0=key,k1=empty)` or `(k0=key,k1=zero)` as you already do.
* For `prog_gen_coeff`:

    * if `StepDcf`: same: key contains base + cutpoints + deltas, party0 gets actual payload and party1 zeros (test-only).
    * if `IntervalLut`: key contains explicit intervals+payloads.

### `pfss/Myl7FssBackend`

This is the real milestone work.

#### 6.2.1 Pred program lowering (baseline)

Given `PredProgramDesc` with Q raw queries:

* Generate **one DCF key per query** (baseline; no packing yet).
* Concatenate them into a single `Bytes` blob with a self-describing format:

**Recommended key blob encoding**

```
u32 Q
repeat Q:
  u8 kind (RawPredKind)
  u8 f
  u16 reserved
  u32 key_len
  key_bytes[key_len]
```

**Eval**

* Parse blob
* For each query:

    * call into myl7 to eval that DCF at `hatx` (or `hatx_low` for LtLow)
    * obtain a bit share (0/1 as u64) and append to output
* Output format for `kU64PerBit`: `Q * 8` bytes, little-endian `u64` per bit

This matches your current arithmetic-bit domain and avoids conversions.

#### 6.2.2 Coeff program lowering (baseline Step-DCF)

Given `CoeffProgramDesc` in `kStepDcf` form:

* Dealer samples random base share vector for each party:

    * Choose `base0` random, set `base1 = base - base0`
* For each cutpoint `cut[t]` and delta vector `delta[t]`:

    * Use myl7 vector-payload DCF to program a function that outputs:

        * `delta[t]` if `hatx >= cut[t]` else `0`
    * Again generate keys `(k0,k1)` whose outputs sum to delta[t] vector.

**Key blob format**

```
u32 out_words
u32 num_cuts
u64_vec base_share[out_words]
repeat num_cuts:
  u64 cutpoint
  u32 key_len
  key_bytes[key_len]    // DCF key for vector payload
```

**Eval**

* Parse
* Start with `acc = base_share`
* For each cut:

    * eval DCF at hatx → vector share `v_t` (out_words u64 shares)
    * acc += v_t
* Output `acc` as `out_words*8` bytes little-endian

This maps directly onto your existing GeLU step backend and generalizes to any SUF coefficient LUT.

#### 6.2.3 Interval LUT mode (optional in Milestone 6)

If myl7 has an interval-LUT primitive, you can implement `kIntervalLut`. If not:

* it’s fine to return “unsupported” for now and keep compilation using `kStepDcf` in Milestone 5.
* SigmaFast will implement `kIntervalLut` later.

---

## 6.3 SigmaFastBackend stub (same interface, upgrade later)

Implement:

### `pfss/SigmaFastBackend` (stub)

Must compile and pass tests while internally delegating to baseline:

* `prog_gen_pred`: for now, call baseline `prog_gen_pred`
* `eval_pred_many`: for now, call baseline `eval_pred_many`

But keep the type hooks needed later:

* store thresholds in a grouped form internally:

    * all `LT_U64` thresholds (vector)
    * per `f`, all `LTLOW` thresholds
      so you can replace the internals with a **packed multi-threshold compare engine**.

**Success criterion for the stub**

* It produces identical outputs to Clear/myl7 for the same `PredProgramDesc`.

---

## 6.4 Definition of Done (Milestone 6) — test matrix

Add backend-level tests:

1. `test_pfss_pred_program.cpp`

* random lists of queries
* random `hatx`
* reconstruct pred outputs from both parties and compare to plaintext query evaluation

2. `test_pfss_coeff_step.cpp`

* random cutpoints + random payload words
* reconstruct coefficient vector and compare to plaintext step function selection

3. `test_compiled_gate_under_backends.cpp`

* compile a SUF (start with GeLU spline SUF already in repo)
* generate keys via backend
* evaluate via backend (both parties)
* compare reconstructed outputs to SUF ref

Run all tests under:

* `ClearPfssBackend`
* `Myl7FssBackend`
* `SigmaFastBackend` (stub delegate)

---

# Practical sequencing (what to implement first)

To keep momentum and avoid refactors:

1. **Milestone 5**: implement descriptors + compiler + compile correctness tests using *pure plaintext simulation* (no real PFSS keys yet).
2. **Milestone 6**: wire compiler output to PFSS interface:

    * start with Clear backend (easy)
    * then myl7 adapter using “one DCF per query/cut”
    * finally SigmaFast stub delegating to baseline

This will give you an end-to-end path:
**SUF → compiler → PFSS keys → online eval**
with correctness validated at each boundary.
