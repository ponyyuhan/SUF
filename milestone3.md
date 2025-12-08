## ✅ Checkpoint: the **wrap_flag / wrap_sign** must NOT be public

You currently store:

* `bool wrap_sign` in `ReluARSPartyKey(Online)`
* tape begins with `[wrap_flag] ...`

If that `wrap_flag` (or `wrap_sign`) is **the same value for both parties** (i.e., public), it leaks information about the mask (r). Since (\hat x = x + r) is public, **any leaked bit about (r)** leaks a bit about (x) (because (x = \hat x - r)). This breaks privacy.

**Fix (minimal, keeps your logic):**

* Replace `bool wrap_sign` with **a secret-shared bit** (stored as `u64` additive share `0/1`), e.g.

    * `u64 wrap_sign_share` in each party key
    * tape stores `U64 wrap_sign_share` (not a bool)
* In online evaluation, use `wrap_sign_share` only via MPC selection:

    * `sel(w, A, B) = B + w*(A - B)` (one Beaver mul)

This way, neither party learns wrap, but your formula can still branch internally.

I strongly recommend you do this now because Milestone 3/4 will generalize masking rewrites—and you don’t want “public wrap” creeping into the generic engine.

---

# Milestone 3 — SUF IR stabilization (reference semantics + validation + tests)

You said SUF IR exists but isn’t hardened. Milestone 3 makes SUF **executable and testable** independent of PFSS.

### Deliverables (in your repo)

Add:

1. `include/suf/validate.hpp`
2. `include/suf/ref_eval.hpp`
3. `src/demo/test_suf_ref_eval.cpp` (or integrate into your harness test suite)

### 3.1 `include/suf/validate.hpp` (what must be validated)

Your SUF invariants should be enforced at construction or at least in debug:

* Interval boundaries:

    * strictly increasing in unsigned order
    * cover the whole domain in your representation (either last implicit end = (2^{64}), or explicit sentinel)
* Polynomial degrees:

    * each poly has degree ≤ `d` (or stored exactly `d+1`)
* Bool expressions:

    * no unknown node kinds
    * `LTLOW(f, gamma)` has `0 < f ≤ 64`, `gamma < 2^f` (for `f=64`, gamma < 2^64 is always true if u64)
* Output arity matches metadata

### 3.2 `include/suf/ref_eval.hpp` (precise semantics)

You need a *single source of truth* for evaluating:

* predicates
* BoolExpr
* polynomials (Horner mod (2^{64}))
* interval selection

**Predicate semantics (matching your paper):**

* `LT(beta)` means `uint64_t(x) < uint64_t(beta)` (unsigned)
* `LTLOW(f,gamma)` means `(x mod 2^f) < gamma`

    * for `f==64`: `x mod 2^64 == x`
    * for `f<64`: `x_low = x & ((1ULL<<f)-1)`
* `MSB(x)` means bit 63 of x (two’s complement sign): `(x >> 63) & 1`
* `MSB(x+c)` means `MSB(x + c)` mod (2^{64})

**Poly evaluation:**

```cpp
y = a_d;
for k = d-1..0: y = y*x + a_k;   // all in uint64 wraparound
```

**Interval selection:**
Assuming your IR stores `alpha[0..m]` with `alpha[i] < alpha[i+1]` and last is sentinel,
select i s.t. `alpha[i] <= x < alpha[i+1]`.

If your IR uses “implicit last end = 2^64”, implement `end = (i+1<m ? alpha[i+1] : 2^64)` using `unsigned __int128` in validator/ref-eval (only for comparisons), while still storing `u64` boundaries.

### 3.3 Tests (what actually gives you confidence)

Add one exhaustive test at small width and one randomized test at 64-bit.

* **Exhaustive mode**: instantiate SUF with `n=8` (byte ring) and iterate all `x in [0,255]`

    * ensures interval boundaries and BoolExpr evaluation are correct
* **Random mode**: `n=64`, random `x`, confirm invariants/properties

This is the right time to also test that your current GeLU spline “toy coefficients” match the SUF reference (so later your SUF→PFSS compilation can be validated).

---

# Milestone 4 — Mask-rewrite engine (§3.3) as code + proofs-by-tests

Milestone 4 turns your ad-hoc “rotate cutpoints by mask” into a **general predicate rewriting library** with tests.

### The key design constraint (very important)

The rewrite must **not leak wrap** via public booleans or via gate-key control flow. That means either:

* **Approach A (cleanest):** dealer programs PFSS/DCF to output the *final masked predicate bit* directly (no local wrap logic).
* **Approach B (works with your current design):** keep your two-threshold method but represent wrap as a **secret-shared bit**, used only in MPC selection.

Given your current ReluARS/GeLU already use multi-DCF pieces and local logic, **Approach B is the minimal refactor**, so I’ll push that.

---

## 4.1 Core primitive: rotated interval membership under masking

For `C_beta(x)=1[x<beta]`, under mask `hatx = x + r`, the set of `hatx` where `x<beta` is the rotated interval:

* `[r, r+beta) mod 2^64` (length beta)

Membership can be computed from two comparisons:

* `a = 1[hatx < r]`
* `b = 1[hatx < (r+beta mod 2^64)]`
* `wrap = 1[(r+beta) < r]` (in unsigned u64)

Then:

* if `wrap==0`: `m = (!a) & b`
* if `wrap==1`: `m = (!a) | b`

**Milestone 4 code should implement:**

* offline: compute `wrap` and secret-share it
* online: evaluate `a,b` via DCF, then compute `m` via MPC with shared wrap

Same applies to low-bit comparison `x mod 2^f < gamma` with `r_low = r mod 2^f`, in the ring `Z_2^f`.

---

## 4.2 What to implement (files)

Add:

1. `include/suf/mask_rewrite.hpp`

    * deterministic rewrite recipes computed by the dealer
2. `include/suf/mask_rewrite_eval.hpp`

    * reference evaluator for rewritten form (for tests)
3. `src/demo/test_mask_rewrite.cpp`

    * property tests: rewritten predicate on `hatx` matches original on `x`

### API shape (minimal, reusable)

In `mask_rewrite.hpp` define recipes that enumerate exactly what PFSS needs:

```cpp
struct RotCmp64Recipe {
  uint64_t theta0;   // r
  uint64_t theta1;   // r + beta (mod 2^64)
  uint8_t  wrap;     // 0/1, but MUST be secret-shared in real protocol
};

inline RotCmp64Recipe rewrite_lt_u64(uint64_t r, uint64_t beta);

struct RotLowRecipe {
  int f;
  uint64_t theta0;   // r_low
  uint64_t theta1;   // r_low + gamma (mod 2^f)
  uint8_t  wrap;     // in 2^f sense
};
inline RotLowRecipe rewrite_ltlow(uint64_t r, int f, uint64_t gamma);
```

For `MSB(x+c)` you reduce to a half-interval membership:

* Let `start = r - c` (mod (2^{64}))
* membership `h = 1[(hatx - start) mod < 2^63]`
* then `MSB(x+c) = 1 - h` (or `!h`)

So implement:

```cpp
inline RotCmp64Recipe rewrite_msb_add(uint64_t r, uint64_t c); // uses beta=2^63 and start=r-c
```

---

## 4.3 How to apply Milestone 4 to your current ReluARS code (delta-minimal)

Right now your tape order contains `[wrap_flag]` and your party key contains `bool wrap_sign`.

**Replace that with:**

* `u64 wrap_sign_share` (0/1 additive share)
* tape writes `U64 wrap_sign_share` at the same position

Then in evaluator:

* compute both “no-wrap” and “wrap” candidate bits
* use `sel(wrap_sign_share, wrap_case, nowrap_case)` to pick

You already have:

* arithmetic bit ops and BeaverMul64Batch
  So selection is one multiplication (plus those needed for AND/OR if not already computed).

---

# What I need from you to “push further” cleanly (without asking questions)

You said: “Milestone 3/4 not merged”. Perfect: do this next sequence **exactly**:

1. **Implement Milestone 3**: `suf/validate.hpp` + `suf/ref_eval.hpp` + tests
2. **Implement Milestone 4**: `suf/mask_rewrite.hpp` + property tests
3. **Patch ReluARS**: remove public wrap → secret share wrap in tape + online selection
4. **Patch GeLU step-DCF** similarly if it uses public wrap bits (same rule)

---

## Quick correctness checklist for your current progress (pass/fail)

* ✅ Canonical serialization single source: **good**
* ✅ Tape deterministic and typed records: **good**
* ✅ BeaverMul64Batch one-round: **good**
* ✅ Clear backend returns additive shares: **good**
* ✅ Harness uses plaintext references: **good**
* ⚠️ `wrap_flag` / `wrap_sign` public bool: **must fix** as above

---

If you want, paste the **exact code location** where `wrap_flag` is read and used in `reluars_online_complete.hpp` (the few lines), and I’ll give you a precise patch that converts it to `wrap_sign_share` and expresses the selection using your existing `bit_ring_ops` + `BeaverMul64Batch` (no new primitives needed).
