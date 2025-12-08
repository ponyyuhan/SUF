## Quick check on your Milestone 9 write-up (based on description)

From what you listed, you **very likely meet the Milestone 9 DoD**:

* ✅ Each required scalar gate exists (SiLU / nExp / reciprocal / rsqrt) with reusable fixed-point + piecewise scaffolding.
* ✅ Vector blocks exist (SoftmaxBlock, LayerNorm) wired to the scalar stack.
* ✅ You added cleartext reference evaluators and a 2PC correctness suite (`test_llm_gates`) that passes.
* ✅ You added a batched benchmark runner (`bench_llm_gates --gate=...`).
* ✅ You updated program metadata (GateKind entries) and milestone tracker.

That said, there are a few **high-risk “looks correct but could still bite later” items** worth sanity-checking now:

1. **Tolerance-based correctness**

    * If `test_llm_gates` only asserts `|err| < tol`, make sure `tol` is expressed in **ULP / LSB units** (e.g., ≤ 1–2 LSB at `frac_bits=16`) and not so loose that a rescale bug passes.
    * Best practice: keep **two modes**: *strict fixed-point ref equality* for same-algorithm ref, and *float accuracy* vs `double`.

2. **nExp(t) behavior for t < 0**

    * You clamp negatives to `1`. That’s acceptable for softmax-stable usage (since `t=max-x ≥ 0`), but it can mask upstream sign bugs. Consider an assertion/debug counter in softmax path that “t<0 occurrences should be 0”.

3. **NR (recip/rsqrt) rescale + overflow**

    * Verify each iteration does consistent `mul -> rescale -> clip` (or uses widened intermediate) so you don’t silently overflow ring64 on large inputs / bad eps.

4. **Masked outputs semantics**

    * Ensure downstream consumers (blocks, later layers) always use the *resolved* share value (mask applied), not a “masked pair” accidentally propagated into generic matmul.

5. **Backend coverage**

    * If `test_llm_gates` only runs SigmaFast, you may still want a small matrix of runs for Clear + SigmaFast to catch dispatch / tape-consumption mismatches.

If those checks hold, your Milestone 9 is in good shape.

---

## Milestone 10 — Linear Algebra + Attention (the true LLM core)

### Goal

Add the **linear algebra engine** (fixed-point GEMM + vector kernels) and build an **Attention block with KV cache**, then combine with existing LayerNorm/SiLU/GELU gates to run **one Transformer layer end-to-end** under 2PC, matching plaintext within approximation error.

### 10.0 Scope & design constraints

* Fixed-point ring64 everywhere (reuse your shared fixed-point tooling).
* Support two execution modes:

    1. **Public weights** (typical inference): no comm for linear layers (fast path).
    2. **Private weights** (optional but required by spec): matrix Beaver triples for secret-secret matmul.
* Provide both:

    * **Batch mode**: `B x T x D` (offline evaluation / tests)
    * **Step mode**: `B x 1 x D` + KV cache append (autoregressive inference)

---

## 10.1 Tensor/Linear primitives (foundation)

### Deliverables

Create a small, explicit tensor/view layer for contiguous ring buffers so all matmul/attention code is shape-safe and stride-correct.

**New headers**

* `include/nn/tensor_view.hpp`

    * `TensorView<T>` with `{data, shape[], strides[]}` (row-major default)
    * Helpers: `view3`, `view4`, `reshape`, `slice`, `transpose_view` (view-only)
* `include/nn/linops.hpp`

    * Elementwise: `add`, `sub`, `mul_const`, `axpy`, `hadamard` (secret-secret mul uses Beaver)
    * Reductions: `sum_lastdim`, `max_lastdim` (max uses compare+select; masks can be public)

**New sources**

* `src/nn/linops.cpp` (CPU reference + optimized loops)

**Acceptance**

* Unit tests for shape/stride correctness and simple ops under shares.

---

## 10.2 GEMM engine (public weights)

### What to implement

When weights are public, each party computes its output share locally:
[
Y^{(p)} = X^{(p)} \cdot W \quad (\text{and optionally } + b/2 \text{ if bias is public and you split it})
]

**Files**

* `include/nn/matmul_publicW.hpp`
* `src/nn/matmul_publicW.cpp`

**Required API**

* `matmul_publicW(const TensorView<u64>& X_share, const TensorView<i64>& W_public, TensorView<u64> Y_share, MatmulParams)`

    * Support 2D and batched 3D (`[B,M,K] x [K,N] -> [B,M,N]`)
    * Support optional `bias_public` and optional `W_transposed` flag
    * Fixed-point: `(x * w) >> frac_bits` with your standard rounding rule

**Bench**

* `src/bench/bench_gemm.cpp` (or merge into `bench_llm_blocks.cpp`)

    * Bench typical sizes: `(B*T, D) x (D, 3D)` and `(B*T, D) x (D, D)`.

**Optional stretch**

* `src/nn/cuda/matmul_publicW.cu` gated by `USE_CUDA` (cuBLAS), measuring overlap potential with comm-heavy ops.

---

## 10.3 GEMM engine (private weights) via **Matrix Beaver triples**

### What to implement

Secret-secret matmul needs Beaver preprocessing: generate secret A,B and C=A·B.

Online protocol for `Y = X·W`:

1. `E = open(X - A)`, `F = open(W - B)`
2. `Y = C + E·B + A·F + E·F`
   (then fixed-point rescale as required)

**Files**

* `include/nn/matmul_beaver.hpp`
* `src/nn/matmul_beaver.cpp`

**Tape integration**

* Add a dedicated tape section for matrix triples (or a structured record) so consumption is deterministic:

    * `A` shape `[M,K]`, `B` shape `[K,N]`, `C` shape `[M,N]`
* Dealer support:

    * `dealer_gen_matmul_triple(M,K,N, frac_bits, TapeWriter&)`

**Required API**

* `matmul_beaver(ctx, X_share, W_share, Y_share, MatmulParams, TapeReader&)`
* Must be batched and reuse your existing batched Beaver plumbing for the underlying multiplications.

**Tests**

* `src/demo/test_matmul_publicW.cpp` (sanity + rescale)
* `src/demo/test_matmul_beaver.cpp` (compare vs plaintext; include random shapes + edge sizes)

**Bench**

* Add to gemm bench: private W path (smaller sizes to keep runtime sane).

---

## 10.4 Attention block + KV cache

### Required functionality

Implement multi-head self-attention with KV cache in secret shares (or your masked-share format), supporting both batch and step mode.

**New headers**

* `include/nn/kv_cache.hpp`

    * `KVCache` structure:

        * storage: `K_share`, `V_share` shaped `[B, H, S_max, Dh]`
        * `cur_len` (public)
    * ops: `append(K_t, V_t)`, `view_prefix(len)`
* `include/nn/attention_block.hpp`

    * `AttentionConfig{D, H, Dh, S_max, frac_bits, use_causal_mask, use_rope(optional)}`

**Attention eval graph**

1. **QKV projection**: `X -> Q,K,V`

    * Prefer fused: one matmul `Wqkv` producing `[B,T,3D]` then split.
2. **(Optional) RoPE** (if you want LLaMA-style realism)

    * Since sin/cos are public, apply as local linear ops on shares.
3. **Update / read KV cache**
4. **Scores**: `S = (Q · K^T) * inv_sqrt(Dh)`

    * Core kernel: lots of dot-products between secret vectors:

        * implement `dot_product_beaver(q, k)` batched over `(B,H,T,S)`
5. **Masking**

    * Causal/padding masks are public: ensure masked positions contribute **exactly 0** after softmax; simplest:

        * multiply `exp` outputs by `{0,1}` mask **before** sum.
6. **SoftmaxBlock** (reuse Milestone 9)
7. **Context**: `P · V`

    * Another secret-secret matmul / reduction; implement a specialized kernel for `[T,S]x[S,Dh]`.
8. **Output projection**: `context -> D` via publicW or beaver matmul
9. **Residual add**: output + input

**Sources**

* `src/nn/attention_block.cpp`
* `src/demo/test_attention_block.cpp`
* `src/demo/demo_kv_cache_step.cpp` (step consistency)

**Correctness acceptance**

* For small sizes (e.g., `B=2, T=4..8, D=32, H=4, Dh=8`):

    * Batch attention matches plaintext fixed/float within defined tolerance.
    * Step mode with KV cache matches batch mode (same weights/inputs, token by token).

**Bench**

* `src/bench/bench_attention.cpp`

    * Measure:

        * batch mode throughput for modest `T`
        * step mode latency vs `S` (cache length sweep)

---

## 10.5 Transformer layer end-to-end (Attention + MLP + Norms)

### What to implement

A minimal “real” layer that exercises everything:

**Pre-norm transformer layer**

1. `u = LayerNorm(x)`
2. `a = Attention(u)` (with KV cache support for step mode)
3. `x1 = x + a`
4. `v = LayerNorm(x1)`
5. `m = MLP(v)`
6. `y = x1 + m`

**MLP variants (pick at least one, ideally both)**

* GELU MLP: `W1 -> GeLU -> W2`
* SwiGLU (SiLU-gated): `W1x` and `W3x`, `SiLU(W1x) ⊙ (W3x) -> W2`

**Files**

* `include/nn/mlp_block.hpp`, `src/nn/mlp_block.cpp`
* `include/nn/transformer_layer.hpp`, `src/nn/transformer_layer.cpp`
* Tests:

    * `src/demo/test_transformer_layer.cpp`

**Acceptance**

* One layer runs end-to-end under 2PC and matches plaintext within approximation error for randomized weights and inputs.
* Include a deterministic seed + at least one “stress” test with larger `T` (still modest D).

---

## 10.6 Definition of Done (Milestone 10)

You’re done when all are true:

1. **GEMM engine**

    * `matmul_publicW` implemented + tested + benchmarked.
    * `matmul_beaver` (private weights) implemented + tested + benchmarked.
    * Deterministic tape consumption for matrix triples.

2. **Attention + KV cache**

    * Multi-head attention works in **batch** and **step** modes.
    * Step mode (KV cache) matches batch mode on the same sequence.

3. **One transformer layer end-to-end**

    * Runs under 2PC and matches plaintext within defined tolerance (document the tolerance clearly).
    * Includes at least: LayerNorm, Attention(Softmax), MLP(GeLU or SiLU/SwiGLU), residuals.

4. **Bench coverage**

    * Bench for GEMM and Attention (and optionally full layer) added under `src/bench/`.

5. **Docs**

    * `docs/milestone_acceptance.md` updated with commands and acceptance notes:

        * build + tests + bench invocations
        * typical sizes + expected output metrics (even if rough)

---

If you want an extra-powerful safety net for Milestone 10: make your `test_transformer_layer` include **both** (a) “public weights” path (fast) and (b) “private weights” path (Beaver) on tiny dims. That catches 95% of integration mistakes early (scales, rescale points, shape bugs, cache indexing, masking).
