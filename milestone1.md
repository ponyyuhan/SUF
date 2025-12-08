
* **Milestone 1** in your repo = make “proto runtime primitives” (ring/bytes/channel/beaver/bit ops) **canonical** and **batched** (single open round for a whole vector).
* **Milestone 2** in your repo = make preprocessing artifacts **streamable and deterministic** via a **tape** (per-party), and refactor dealer/online code to **consume the same ordering** always.

Below is a step-by-step, concrete plan that fits your current structure and minimizes churn.

---

## Ground rules (based on your choices)

* **Option A**: weights are public → linear layers are local on shares (no matrix Beaver triples). This reduces offline load massively.
* **f = 12** fixed-point → ReluARS truncation logic and “low-bit” predicates are keyed to `f=12`.

---

# Milestone 1 (in *your* repo): “Canonical runtime primitives + batching correctness”

### Goal

Your `proto/` already contains the right ideas. Milestone 1 is to make them **the stable core** (even if they stay under `proto/` for now), with:

1. a single set of `u64` ring ops & serialization helpers,
2. a single channel abstraction used everywhere,
3. **batched BeaverMul** (the most important piece for SIGMA-like throughput),
4. unit tests that prove correctness independent of SUF/FSS.

You already have many of these. The work is to **lock interfaces** and **remove per-file drift**.

---

## Step 1.1 — Choose one “canonical runtime namespace”

You currently have both:

* `include/core/*` (ring & serialization)
* `include/proto/common.hpp`, `bit_ring_ops.hpp`, `channel.hpp`, `beaver_mul64.hpp`

**Do this:**

* Keep `include/core/*` as the canonical basic building blocks (ring + bytes).
* Keep `include/proto/*` as canonical MPC runtime (channel + Beaver + bit ops), but **make proto depend on core** (not vice versa).

**Concrete actions**

* Ensure `proto/common.hpp` includes `core/ring64` and `core/bytes` rather than re-defining pack/unpack.
* Ensure all `u64` ops used in proto resolve to one implementation (wraparound).

**Definition of done**

* Grep check: only one implementation of `pack_u64_le/unpack_u64_le` and ring ops is used by all modules.

---

## Step 1.2 — Make Beaver multiplication batched the default

You already have `proto/beaver_mul64.hpp`. SIGMA-style performance requires:

* one communication round per batch: open all `e[i], f[i]` together.

**Implement / refactor to this API (exact contract):**

```cpp
struct BeaverTriple64Share { u64 a,b,c; };

class BeaverMul64Batch {
public:
  BeaverMul64Batch(int party, IChannel& ch, span<const BeaverTriple64Share> triples);

  // out[i] = x[i] * y[i] as additive shares in Z_2^64
  void mul(span<const u64> x, span<const u64> y, span<u64> out);

  size_t triples_used() const;
};
```

**Important details**

* Must consume triples sequentially (deterministic)
* Must send `e_s[]` and `f_s[]` as contiguous buffers
* Must reconstruct `e,f` and apply:

    * `z = c_i + e*b_i + f*a_i + (party==0? e*f : 0)`

**Definition of done**

* Add a unit-like test (can live under `src/demo/` initially):

    * generate random triples (insecure OK for test),
    * random secrets split to both parties,
    * run `mul()` once with `N=1024`,
    * reconstruct and compare to `x*y`.

You already validated similar logic in the harness; now it becomes a permanently tested primitive.

---

## Step 1.3 — Standardize “bit operations on shares”

Right now your predicate bits are often treated as **arithmetic u64 shares** of {0,1}. That’s acceptable for correctness, but inefficient.

For Milestone 1, just lock a contract:

* **Gate logic may represent Boolean values as arithmetic shares** in `Z_2^64` (0/1).
* `bit_ring_ops.hpp` must provide:

    * `not(bit)`, `and(bit,bit)`, `or(bit,bit)`, `xor(bit,bit)`
    * `sel(bit, x0, x1)` (using BeaverMul)
    * `lut8(w,t,d, table[8])`

**Definition of done**

* Deterministic selection primitives run with the batched Beaver (or scalar Beaver wrapper).
* Add a small test: verify LUT8 selection matches plaintext on random shares.

---

## Step 1.4 — Make batching utilities “canonical”

You already have `pack_utils.hpp`. For Milestone 1, enforce interface stability:

* `pack_keys_flat(keys[N]) -> bytes[N * key_bytes]`
* optional: `pack_by_cut(cut_keys[C][N]) -> bytes[C][N * key_bytes]`

**Definition of done**

* A test that:

    * packs keys,
    * unpacks them,
    * evaluates DCF individually vs via `eval_many_u64` produces identical outputs.

(Works with `backend_clear.hpp` now; later with myl7.)

---

# Milestone 2 (in *your* repo): “Tape (offline artifacts) + deterministic consumption”

### Goal

Right now dealer outputs an in-memory struct per gate. That’s fine for prototyping but not scalable.

Milestone 2 turns preprocessing into:

* **two per-party tapes**: `tape_P0`, `tape_P1`
* deterministic record order
* online evaluators can run using only:

    * public masked input `hatx`
    * a channel
    * a tape cursor

This is the single largest step from “prototype slice” → “system”.

---

## Step 2.1 — Add a tape module under your existing structure

Place it where it belongs in your repo:

* `include/proto/tape.hpp` (or `include/mpc/tape.hpp` if you prefer)
* `proto` is fine since it’s runtime/preproc glue.

### Record format (simple, robust, matches your needs)

Each record:

* `tag: u32`
* `len: u32`
* `payload: len bytes` (little-endian for numeric arrays)

Tags you need *now*:

* `U64` (len=8)
* `U64VEC` (len=8*k)
* `BYTES` (arbitrary; for FSS keys)
* `TRIPLE64` (len=24)

You do **not** need a million tag types; keep it minimal.

**Definition of done**

* Tape supports both:

    * `VectorTape` (in-memory) for tests/harness
    * `FileTape` (persisted) for real runs
* There is a “roundtrip test”: write random records → read back equals.

---

## Step 2.2 — Introduce `TapeCursor` and “consumption plans”

The hardest bug class in preprocessing systems is: *producer and consumer disagree on order*.

So: every gate must have an explicit “plan” documenting exactly what it consumes.

### Implement a tiny cursor API

```cpp
class TapeReader {
public:
  u64 read_u64();
  vector<u64> read_u64_vec(size_t words);
  vector<uint8_t> read_bytes();
  BeaverTriple64Share read_triple64();
};
```

### Write down exact consumption order (ReluARS)

For **each instance** (per element), consume in this order (per party tape):

1. `r_in_share : U64`
2. `r_hi_share : U64`   (since f=12)
3. `r_out_share: U64`
4. `k_hat_lt_r : BYTES`
5. `k_hat_lt_r2: BYTES`
6. `k_zlow_lt_rlo : BYTES`  (domain f bits)
7. `k_zlow_lt_rlo1: BYTES`
8. `beaver_triples: TRIPLE64 * Nmul` (whatever your evaluator uses)

If you include the correction table as public constants, don’t store it.

### And for GeLU (interval LUT version)

Per instance:

1. `r_in_share : U64`
2. `r_out_share: U64`
3. `k_hat_lt_r : BYTES`
4. `k_hat_lt_r2: BYTES`
5. Coeff program key:

    * either `interval_lut_key: BYTES` (fast path)
    * or `step_cut_keys[j]: BYTES` for each cut, plus base coeff shares as `U64VEC`
6. `beaver_triples: TRIPLE64 * (d + extras)`

**Definition of done**

* A “plan file” comment block exists in each dealer/evaluator header.
* Online code reads exactly those fields in that order.

---

## Step 2.3 — Refactor dealers to *write tapes* instead of returning structs

You do not need to delete the existing in-memory key structs yet. Do a dual-path:

* `ReluARSDealer::gen_instance(...) -> ReluARSKeyParty` (existing)
* `ReluARSDealer::write_instance(TapeWriter& t0, TapeWriter& t1, ...)` (new)

Same for GeLU.

**Definition of done**

* You can generate N instances into tape and run online purely by reading tape.

---

## Step 2.4 — Refactor online evaluators to support tape consumption

Again do dual-path to minimize disruption:

* `reluars_online_complete.hpp` currently takes a `KeyParty&`.
* Add a wrapper:

    * `ReluARSEvaluatorFromTape` that does:

        1. read key material for one instance from tape
        2. call the existing evaluator logic (or inline the same logic)

Same for GeLU evaluators.

**Definition of done**

* `demo_proto_endtoend.cpp` gains a new mode:

    * dealer writes tape
    * evaluators read tape
    * outputs match the old in-memory mode bit-for-bit

---

## Step 2.5 — Upgrade `sim_harness.cpp` correctness checks (must do)

Right now you said:

> harness uses “reconstruct value self-check” rather than true plaintext reference

That’s not enough once you start swapping backends.

**Do this now:**

* Add true plaintext references:

    * ReluARS reference (matching your truncation + delta)
    * GeLU spline reference (evaluate same spline coeffs and interval logic)

Then your harness checks:

* reconstructed `y` equals plaintext `F(x)` (mod 2^64) within expected quantization.

**Definition of done**

* 10k random tests pass with:

    * `backend_clear.hpp`
    * tape-based consumption
    * both parties in-process

---

# Concrete “next changes” list mapped to your tree

### Add

* `include/proto/tape.hpp`
* `src/demo/demo_proto_tape_endtoend.cpp` (or extend existing `demo_proto_endtoend.cpp`)
* tests: `src/demo/sim_harness_tape.cpp` (or integrate flags in your harness)

### Modify

* `include/proto/beaver_mul64.hpp` → add/ensure `mul_batch` support (or add `beaver_mul64_batch.hpp` and switch callers)
* `include/proto/reluars_dealer.hpp` → add tape writer path
* `include/proto/reluars_online_complete.hpp` → add tape reader wrapper
* `include/proto/gelu_spline_dealer.hpp` and `gelu_online_interval_lut.hpp` similarly
* `sim_harness.cpp` → add plaintext references & tape mode

---

# Why this approach is the best fit for your current state

* It **preserves** your working slice and your proto design.
* It turns your proto layer into a **real runtime substrate** (Milestone 1).
* It introduces the **single missing systems piece** (Tape) without forcing you to commit to a final backend yet (Milestone 2).
* It materially reduces future pain when integrating myl7 and then SigmaFastBackend: once tape + batched beaver are stable, backend swapping is localized to `pfss/`.

---

If you want, I can output (next message) a **drop-in `include/proto/tape.hpp`** that matches your existing `proto/common.hpp` style and a patch-style outline showing exactly how `reluars_dealer.hpp` writes tape and how the evaluator consumes it.
