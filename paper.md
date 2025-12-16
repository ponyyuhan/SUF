## 1 Setting and High‑Level Design

We work in the standard two‑party computation (2PC) *preprocessing model*. In the offline phase the two servers receive input‑independent correlated randomness; in the online phase they run a fast, input‑dependent protocol using this randomness. This model underlies recent FSS‑based systems for secure ML inference such as SIGMA (semi‑honest GPT/transformer inference) and SHARK (actively secure inference).

Following prior FSS‑based work, the main invariant is that every intermediate value (x \in \mathbb{Z}*{2^n}) is represented by a *masked value*
[
\hat{x} = x + r*{\text{in}} \in \mathbb{Z}*{2^n},
]
where (r*{\text{in}}) is a random mask chosen during preprocessing. The masked value (\hat{x}) is *public* to both parties; the mask is shared as
[
r_{\text{in}} = r_{\text{in},0} + r_{\text{in},1} \pmod{2^n},
]
with each party (P_b) holding (r_{\text{in},b}). Linear layers are evaluated on additive secret shares ((x_0,x_1)) with (x = x_0 + x_1), using Beaver triples for multiplications. Non‑linear gates are evaluated via function secret sharing (FSS) on the public masked input (\hat{x}), combined with the existing shares ((x_0,x_1)) (or shares obtained via a masked‑to‑shares protocol described below).

Our contribution is an explicit *abstraction layer* and compiler on top of this invariant:

1. **Structured‑Univariate Functions (SUF).**
   A single, uniform function family that captures essentially all fixed‑point non‑linearities used in modern transformers (ReLU/ReluARS, GeLU, SiLU, reciprocal/rsqrt, nExp, softmax blocks, layer norm) *at the level of scalar inputs*. SUFs expose both arithmetic outputs (e.g., a non‑linear function value) and Boolean “helper bits” (e.g., sign bits, interval indices, clipping predicates) in a single typed object.

2. **SUF→FSS (programmable FSS backend) compilation.**
   A generic compilation pipeline that takes a symbolic SUF description and emits a small constant number of programs for a *DPF/DCF‑style FSS backend*. We refer to this abstract backend as a *programmable FSS backend* (PFSS): it only requires standard DPF/DCF functionality (comparison / interval selection and vector‑payload LUTs). Boyle‑style programmable DPF (PDPF) can be used *internally* as a PCG layer to generate these DPF/DCF keys, but is not assumed as the online backend.

3. **Composite‑FSS gates.**
   A gate‑level protocol abstraction where each gate type (\tau) (e.g., ReluARS, GeLU, SoftmaxBlock, LayerNorm) is specified once as a *Composite‑FSS* module
   [
   (\mathsf{CompGen}*\tau, \mathsf{CompEval}*{\tau,0}, \mathsf{CompEval}_{\tau,1})
   ]
   built mechanically from its SUF description and the FSS backend. Correctness and privacy of all gates reduce to the security of the underlying backend and of the triple‑based arithmetic layer.

Conceptually, SUF plays the role of a *domain‑specific intermediate representation (IR)* for fixed‑point non‑linearities. SIGMA and SHARK implicitly exploit much of the same structure—e.g., they design piecewise‑polynomial approximations and helper‑bit protocols for GeLU, SiLU, ReluARS, splines, reciprocal, etc.—but they do so by presenting a collection of *per‑primitive* FSS or IFSS protocols.

Our framework makes that structure explicit, proves that all these primitives fall into a single function family, and gives a compiler and security proof *once and for all* for the whole family.

---

## 2 Structured‑Univariate Functions (SUF)

### 2.1 Definition

Fix an integer (n \ge 1) and let (R = \mathbb{Z}_{2^n}). We work with unsigned representatives ({0,\dots,2^n-1}), but occasionally interpret them in two’s complement when talking about “sign” or “MSB” tests, as is standard in FSS‑based ML protocols.

A **Structured‑Univariate Function (SUF)** over (R) with (r) arithmetic outputs and (\ell) Boolean outputs is specified by:

* A bit‑width (n) and domain ring (R = \mathbb{Z}_{2^n}).
* A sequence of interval boundaries
  [
  0 \le \alpha_0 < \alpha_1 < \cdots < \alpha_m \le 2^n,
  ]
  inducing intervals (I_i = [\alpha_i,\alpha_{i+1})) for (i=0,\dots,m-1).
* For each interval (I_i), a vector of degree‑(\le d) polynomials
  [
  P_i(x) = \big(P_{i,1}(x), \dots, P_{i,r}(x)\big) \in R[x]^r,
  ]
  and a vector of Boolean formulas
  [
  B_i(x) = \big(B_{i,1}(x), \dots, B_{i,\ell}(x)\big) \in {0,1}^\ell,
  ]
  where each (B_{i,j}) is built from the following primitive predicates and Boolean connectives:

    * comparisons to public constants:

        * (\mathbf{1}[x < \beta]) for (\beta \in R),
        * (\mathbf{1}[x \bmod 2^f < \gamma]) for parameters (f \le n), (\gamma < 2^f);
    * most‑significant‑bit predicates:

        * (\mathrm{MSB}(x)), meaning the sign bit of (x) when viewed in two’s complement,
        * (\mathrm{MSB}(x + c)) for public (c \in R);
    * Boolean connectives (\neg, \wedge, \vee, \oplus); and constants 0,1.

The SUF (F : R \to R^r \times {0,1}^\ell) associated to this description is:
[
F(x) = \big(P_i(x),, B_i(x)\big)\quad\text{for the unique }i\text{ such that }x\in I_i.
]

We denote this family by (\mathsf{SUF}_{n}^{r,\ell,d}).

Intuitively, SUFs are *piecewise polynomial functions* on the ring (R), enriched with a bounded amount of structured control flow (comparisons, MSB tests, modular comparisons) whose outputs we expose as typed Boolean channels. This matches the structure of all non‑linearities used in FSS‑based secure inference protocols to date.

### 2.2 Closure properties

The family (\mathsf{SUF}_{n}^{r,\ell,d}) enjoys several closure properties that we rely on when expressing complex gates:

1. **Affine reparameterization.**
   If (F(x)) is a SUF and (a \in \mathbb{Z}_{2^n}^\times, b \in R), then (x \mapsto F(ax + b)) is also a SUF of the same degree. Affine changes of variables just translate and rescale the interval boundaries, and composition of a polynomial with an affine map is still a polynomial of the same degree.

2. **Cartesian products and Boolean combinations.**
   The Cartesian product of finitely many SUFs is a SUF: we can refine the interval partition to the union of all boundaries and concatenate the polynomial and Boolean components. Likewise, Boolean outputs are closed under further Boolean combination, since our predicate language already allows arbitrary Boolean formulas over the primitive tests.

3. **Linear post‑processing.**
   If (F(x)) is a SUF with arithmetic output (y), and we apply a fixed linear map (L: R^r \to R^{r'}) (e.g., scaling, adding a bias, or selecting some outputs), then (x \mapsto (L \circ F)(x)) is again a SUF: we just apply (L) to the polynomial coefficients on each interval.

These closure properties, together with the expressiveness of piecewise polynomials and comparison/MSB predicates, allow us to show that all non‑linear primitives used in modern transformer architectures fall inside SUF for fixed ((n,f)) and a chosen approximation scheme.

### 2.3 Examples: transformer non‑linearities as SUFs

We briefly indicate how representative non‑linearities belong to the SUF family. The detailed formulas follow the constructions used in SIGMA and SHARK (e.g., for ReluARS, GeLU, splines, reciprocal, softmax), but are expressed in our unified language.

* **ReLU and sign tests.**
  ReLU is the piecewise linear function
  [
  \mathrm{ReLU}(x) = \max(x,0) =
  \begin{cases}
  0 & \text{if } \mathrm{MSB}(x) = 1,\
  x & \text{otherwise}.
  \end{cases}
  ]
  Over the signed interpretation of (R), the sign bit is (\mathrm{MSB}(x)), which is directly expressible in our predicate language. Thus ReLU is a degree‑1 SUF with two intervals (“negative” and “non‑negative”) and an obvious Boolean helper (\mathbf{1}[x\ge 0]).

* **LRS/ARS and truncation.**
  Logical right shift (LRS) and arithmetic right shift (ARS) are fixed‑point truncation operations of the form
  [
  \mathrm{LRS}*{n,f}(x) = \left\lfloor \frac{x}{2^f} \right\rfloor,\qquad
  \mathrm{ARS}*{n,f}(x) = \left\lfloor \frac{x}{2^f} \right\rfloor \text{ (with sign extension)},
  ]
  potentially combined with rounding and “gap” corrections. Existing FSS protocols represent them using a small number of comparison/MSB bits and linear combinations of (x). We therefore treat them as *SUF‑compatible*: the SUF exposes helper bits (sign, low‑bits predicates, clipping tests) and simple polynomial functions of (x); the final ARS/LRS value is obtained by linear post‑processing inside the Composite‑FSS gate.

* **ReluARS.**
  SHARK fuses ReLU and ARS into an optimized protocol (\mathrm{ReluARS}_{n,f}) with three helper bits ((w,t,d)) and a small 8‑entry table of correction terms. At the SUF level, we expose:

    * an arithmetic ReLU‑like output (x^+ = \max(x,0)), and
    * Boolean helper bits ((w,t,d)) defined by simple comparisons and MSB tests on (x).

  A small, constant‑size arithmetic circuit in ((x^+,w,t,d)) then reconstructs the full ReluARS value; see §4.4.

* **Splines and smooth activations (GeLU, SiLU, tanh‑like).**
  Both SIGMA and SHARK approximate smooth activations by bounded‑range splines: on a compact interval they use low‑degree polynomials with pre‑computed coefficients; outside, they clip or fall back to simpler functions. These are piecewise polynomials with interval tests; extra control bits (e.g., whether the input is in the central approximation region, which interval index, etc.) are Boolean combinations of comparisons. Hence these functions fall into (\mathsf{SUF}_n^{1,\ell,d}) with small (d).

* **Reciprocal and rsqrt.**
  SHARK’s reciprocal and reciprocal‑square‑root protocols proceed by extracting a floating‑point‑like exponent and mantissa, clipping to a bounded range, looking up a small table of base approximations indexed by the mantissa, and optionally performing one or two Newton steps. Each step is either a low‑degree polynomial in (x) or a table lookup indexed by a small collection of bits derived from comparisons on (x); this fits naturally into SUF.

* **Softmax blocks and LayerNorm.**
  Softmax over a vector (x) is implemented by subtracting the maximum, computing approximate exponentials of each coordinate, summing them, and normalizing by a reciprocal; LayerNorm similarly reduces to polynomial statistics (mean, variance) and rsqrt. SIGMA and SHARK implement each elementwise exponential and reciprocal via splines/LUTs and helper bits. At the scalar level, each non‑linear sub‑primitive is a SUF. The vector‑level operations (max, sum) are linear and do not change the SUF family itself.

The key point is that after fixing (n) and the polynomial/LUT approximation scheme, *all* of these primitives are instances of SUF (or SUF plus a final linear map). This allows a single compiler and security argument to cover every non‑linear gate we need.

---

## 3 From SUF to a Programmable FSS Backend (SUF→FSS)

We now explain how to implement SUFs on masked inputs using a generic programmable FSS backend. Throughout this section we treat the backend purely abstractly and refer to it as a *programmable FSS backend* (PFSS). It can be instantiated using standard large‑domain DPF/DCF constructions (comparison and vector‑payload LUTs); Boyle‑style *programmable DPF (PDPF)* may be used internally as a PCG layer to generate these DPF/DCF keys, but does not appear in the online protocol.

### 3.1 Abstract FSS / PFSS backend

We assume access to any FSS/DPF/DCF‑style primitive that can realize multi‑output functions on \(\{0,1\}^n\) with per‑input evaluation.

Concretely, a **programmable FSS backend (PFSS)** for payload group \(G\) consists of:

* \(\mathsf{Setup}(1^\lambda)\) producing public parameters \(\text{pp}\).
* A programming interface
  \[
  \mathsf{ProgGen}(\text{pp}, \text{desc}) \to (k_0, k_1)
  \]
  that takes as input a *program description* \(\text{desc}\) (e.g., “for each input \(x\), output a payload in \(G\)”) and outputs keys for the two parties.
* A local evaluation algorithm
  \[
  \mathsf{Eval}_b(\text{pp}, k_b, x) \in G
  \]
  such that \(\mathsf{Eval}_0 + \mathsf{Eval}_1\) reconstructs the desired payload function on every input \(x \in \{0,1\}^n\).

The key property we require is that both key size and evaluation cost are polynomial in \(n\) and \(\lambda\), with **no dependence on \(2^n\)** beyond the cost of computing along a control tree of depth \(n\). This is satisfied by essentially all tree‑based large‑domain DPF/DCF schemes used in the FSS literature.:contentReference[oaicite:6]{index=6}

In practice, such a backend can be instantiated in several ways:

* Directly, using a “traditional” large‑domain DPF/DCF (e.g., FKOS‑style comparison DCF, or generic DPF for vector payloads).
* Via *packed* comparison and LUT engines built on such DPF/DCF schemes (as in Grotto, SIGMA, Pika, SHARK), where a single evaluation can yield several predicate bits or several payload words at once; we detail this in §3.1.1.
* Indirectly, using **Boyle‑style small‑domain programmable DPF** as a PCG layer to generate many keys for these traditional DPF/DCF instances. In this case Boyle’s PDPF never appears directly as our SUF backend; instead it compresses the correlated randomness used by the FSS layer.

Our SUF→FSS compiler and Composite‑FSS abstraction treat the backend *purely abstractly*. The correctness and security proofs only rely on the usual FSS security notion for the backend’s function family. Whether one implements the backend with pure DPF/DCF, with Boyle’s PDPF as a PCG layer, or with any other construction is an implementation choice.

#### 3.1.1 Packed comparison and packed LUT instantiations

For concreteness, we briefly describe how two widely‑used FSS patterns—*packed comparisons* and *packed LUTs*—fit our abstract PFSS backend and what concrete costs they have in existing systems.

##### Packed comparisons via Grotto‑style CDPF

Grotto implements high‑throughput comparisons over \(\mathbb{Z}_{2^n}\) using a specialized *comparison DPF* (CDPF) for signed 64‑bit values. The core idea is:

* Start from a standard binary‑tree DPF with VALUE\_BITS‑bit domain and 1‑bit outputs.
* Program the DPF such that the internal “flag” bit on each node encodes whether the unique “target” leaf lies in that subtree.
* Given a reconstructed difference \(S = \text{target} - x\), the parties can compute a bit share of \([x>0]\) by summing the flag bits on a *small* set of internal nodes whose subtrees cover a contiguous block of leaves (corresponding to the interval \(\{S+1,\dots,S+2^{n-1}-1\}\)). This requires time *linear in the tree depth* rather than in the domain size.:contentReference[oaicite:0]{index=0}

The CDPF implementation in Grotto’s open‑source code records the following concrete costs for 64‑bit inputs (VALUE\_BITS = 64):​:contentReference[oaicite:1]{index=1}

* **Key generation.** Generating a pair of CDPF keys with a (possibly random) target costs
  \[
  4\cdot \text{VALUE\_BITS} - 28 = 228
  \]
  local AES calls per party, and the resulting keys are *< 1KB* each.
* **Comparison.** Evaluating \([x>0]\) on additive shares of \(x\) costs
  \[
  2\cdot \text{VALUE\_BITS} - 14 = 114
  \]
  local AES calls per party, plus a single machine word of communication (to exchange additive “mask‑to‑target” offsets).

In our terminology, this CDPF is a concrete instantiation of the PFSS backend for the specific function
\[
g_{\text{cmp}}(x) = \big(\mathbf{1}[x<0],\mathbf{1}[x=0],\mathbf{1}[x>0]\big)
\]
over \(\mathbb{Z}_{2^{64}}\). The SUF primitive predicates \(\mathbf{1}[x<\beta]\), \(\mathbf{1}[x\bmod 2^f<\gamma]\) and \(\mathrm{MSB}(x+c)\) can all be reduced to comparisons of this form by simple local pre‑processing:

* To implement \(\mathbf{1}[x<\beta]\), compare \(x-\beta\) to 0.
* To implement \(\mathrm{MSB}(x)\) we test \([x\ge 2^{n-1}]\), which is again a comparison against a fixed public constant.
* For \(\mathbf{1}[x\bmod 2^f<\gamma]\), we first extract the low \(f\) bits and then compare to \(\gamma\).

Thus, when we say that the predicate PFSS program \(\Pi_{\mathsf{pred}}\) is implemented using a *packed comparison* backend, we mean that we instantiate it with a small number of CDPFs (or similar tree‑based comparison DPFs) that are re‑used across many wires and thresholds. The asymptotic parameters \(K_{\mathrm{FSS}}(n)\) and \(E_{\mathrm{FSS}}(n)\) in §3.6 can be concretized using the above formulas for \(n=64\) (or analogous ones for other bit‑widths).

##### Packed LUTs via vector‑payload DPF (Pika / SIGMA / SHARK)

For LUT‑style functionality—returning an entry from a small table based on a masked index—several works (Pika, SIGMA, SHARK, Curl) use *vector‑payload DPFs*:

* A base DPF on a domain of size \(2^k\) (with \(k\approx 12\!-\!20\)) provides the control tree; each leaf stores a payload block of \(w\) bits (or an entire *vector* of words).
* Evaluation on input \(x\in\{0,1\}^k\) walks the depth‑\(k\) tree, expanding PRG states along the path and combining payloads so that the parties obtain additive shares of the selected table row.:contentReference[oaicite:2]{index=2}

In this setting:

* The **control‑tree cost** (PRG/AES calls and key size) is essentially identical to that of a single DPF/DCF for a scalar payload: one PRG expansion per level and per party, so roughly \(O(\lambda k)\) bit‑operations and \(O(k)\) AES calls.
* The **payload cost** scales linearly with the number of payload words: each additional payload word adds a constant amount of work per level for masking and combining labels.

Pika’s and subsequent implementations report that, for typical parameters (e.g., \(k \in [12,20]\) and 32‑ or 64‑bit payload words), the DPF key size remains in the low‑KB range and the per‑evaluation FSS cost is on the order of a few dozen to a few hundred AES invocations—comparable to a single 64‑bit comparison or less—while returning an entire vector of payload values in one shot.:contentReference[oaicite:3]{index=3} SHARK builds its IFSS LUTs on a closely related vector‑payload DPF core and reports similar asymptotic behaviour, with small constant factors in practice.:contentReference[oaicite:4]{index=4}

In our SUF→FSS compiler, the *coefficient program* \(\Pi_{\mathsf{coeff}}\) is instantiated precisely as such a vector‑payload DPF:

* The domain bit‑width \(k\) is either the full input width \(n\) (for generic piecewise polynomials) or a smaller “mantissa” width obtained after normalization.
* Each leaf payload \(v[i]\) contains all coefficients for the active interval (and, if desired, any small gate‑local LUTs).
* A single evaluation of \(\Pi_{\mathsf{coeff}}\) yields additive shares of *all* coefficients needed by the SUF on that input.

Thus, when we speak of a *packed LUT* backend for \(\Pi_{\mathsf{coeff}}\), we mean exactly this: a vector‑payload DPF that outputs an entire coefficient vector in one FSS call. In §3.6, the parameters \(K_{\mathrm{FSS}}(n)\) and \(E_{\mathrm{FSS}}(n)\) can be instantiated using the concrete key sizes and AES counts for such vector‑payload DPFs as reported in Pika, SIGMA, and SHARK; our asymptotic analysis remains valid regardless of which particular implementation is chosen.

### 3.2 Masked SUFs and SUF descriptors

In the FSS setting we never work with the bare SUF (F); instead we work with *masked variants* tailored to the masking invariant. For each SUF
[
F(x) = (P(x), B(x)) \in \mathsf{SUF}*n^{r,\ell,d}
]
we define a corresponding family of offset functions
[
F^{[r*{\text{in}}, r_{\text{out}}]}(x)
= \big( P(x) + r_{\text{out}},, B(x) \big),
]
to be evaluated on the masked input (\hat{x} = x + r_{\text{in}}).

It is convenient to view this as a function of (\hat x):
[
\widehat{F}(\hat x)
= F^{[r_{\text{in}},r_{\text{out}}]}(x)
= \big(P(\hat x - r_{\text{in}}) + r_{\text{out}},, B(\hat x - r_{\text{in}})\big),
]
where (x = \hat x - r_{\text{in}} \pmod{2^n}).

A **SUF descriptor** is a typed object (\mathsf{desc}(F^{[r_{\text{in}},r_{\text{out}}]})) containing:

* the bit‑width (n) and the interval boundaries (\alpha_0,\dots,\alpha_m),
* the per‑interval polynomial vectors (P_i),
* the per‑interval Boolean formula vectors (B_i),
* the number of arithmetic and Boolean outputs ((r,\ell)),
* and the input/output masks ((r_{\text{in}}, r_{\text{out}})).

The descriptor lives only at key generation time (in the dealer); the online parties only see FSS keys and masked wire values.

### 3.3 Mask‑aware predicates

We briefly recall how the primitive predicates are rewritten under the masked input (\hat x = x + r_{\text{in}}). This section replaces ad‑hoc “(+r)” rules in prior work by a precise case analysis on the ring.

We work over (R = \mathbb{Z}*{2^n}) with canonical representatives ({0,\dots,2^n-1}). Let (r = r*{\text{in}}) and (\hat x = x + r \bmod 2^n).

The SUF predicate language uses the primitives:

* (C_\beta(x) = \mathbf{1}[x < \beta]),
* (D_{\gamma,f}(x) = \mathbf{1}[x\bmod 2^f < \gamma]),
* (\mathrm{MSB}(x)),
* (\mathrm{MSB}(x+c)) for constants (c).

We rewrite each as a Boolean formula in the masked variable (\hat x), using *only* predicates of the allowed forms
(\mathbf{1}[\hat x < \theta]),
(\mathbf{1}[\hat x\bmod 2^f < \theta]),
and (\mathrm{MSB}(\hat x + c')).

#### 3.3.1 Masked comparisons (\mathbf{1}[x < \beta])

Let
[
S_\beta = {\hat x \in R : (\hat x - r) \bmod 2^n < \beta }.
]
This is the image of ([0,\beta)) under addition of (r), hence
[
S_\beta = [r, r+\beta) \pmod{2^n},
]
a “rotated” interval of length (\beta). There are two cases:

* **No wrap‑around** ((r + \beta < 2^n)): then
  [
  S_\beta = {\hat x : r \le \hat x < r + \beta},
  ]
  so
  [
  \mathbf{1}[x < \beta]
  = \neg \mathbf{1}[\hat x < r];\wedge;\mathbf{1}[\hat x < r+\beta].
  ]

* **Wrap‑around** ((r + \beta \ge 2^n)): write (u = r+\beta - 2^n). Then
  [
  S_\beta = {\hat x : \hat x < u} ;\cup; {\hat x : \hat x \ge r},
  ]
  so
  [
  \mathbf{1}[x < \beta]
  = \mathbf{1}[\hat x < u] ;\vee; \neg \mathbf{1}[\hat x < r].
  ]

Thus (\mathbf{1}[x<\beta]) can always be expressed as a Boolean combination of primitive comparisons (\mathbf{1}[\hat x < \theta]) with public thresholds (\theta).

#### 3.3.2 Masked low‑bit comparisons (\mathbf{1}[x \bmod 2^f < \gamma])

Write
[
u = x\bmod 2^f,\quad s = \hat x\bmod 2^f,\quad \delta = r\bmod 2^f.
]
Then (s = (u + \delta) \bmod 2^f). As above, the set of (\hat x) for which (u<\gamma) is a rotated interval of length (\gamma) in ({0,\dots,2^f-1}), and we obtain:

* If (\delta + \gamma < 2^f),
  [
  \mathbf{1}[x \bmod 2^f < \gamma]
  = \neg \mathbf{1}[\hat x\bmod 2^f < \delta]
  ;\wedge;
  \mathbf{1}[\hat x\bmod 2^f < \delta+\gamma].
  ]

* If (\delta + \gamma \ge 2^f), with (u' = \delta + \gamma - 2^f),
  [
  \mathbf{1}[x \bmod 2^f < \gamma]
  = \mathbf{1}[\hat x\bmod 2^f < u'] ;\vee;
  \neg \mathbf{1}[\hat x\bmod 2^f < \delta].
  ]

#### 3.3.3 Masked MSB predicates

We have:

* (\mathrm{MSB}(x) = 1[x \ge 2^{n-1}] = 1 - \mathbf{1}[x < 2^{n-1}]), so we reduce to the previous case with (\beta = 2^{n-1}).
* For (\mathrm{MSB}(x+c)), define (y = x+c) and note
  [
  \mathrm{MSB}(x+c) = \mathrm{MSB}(y) = 1[y \ge 2^{n-1}] = 1 - \mathbf{1}[y < 2^{n-1}].
  ]
  Under the masking (\hat x = x + r), we have
  [
  y = (\hat x - r) + c = \hat x + (c-r) \pmod{2^n}.
  ]
  Our SUF predicate grammar already allows (\mathrm{MSB}(\hat x + c')) as a primitive, so we rewrite every (\mathrm{MSB}(x+c)) to (\mathrm{MSB}(\hat x + (c-r))).

Taken together, these rewrites show:

> **Proposition 3.1 (Closure under masking).**
> Fix (F \in \mathsf{SUF}*n^{r,\ell,d}) and masks ((r*{\text{in}},r_{\text{out}})).
> Then the masked function
> [
> \widehat{F}(\hat x)
> = F^{[r_{\text{in}},r_{\text{out}}]}(\hat x)
> = \big(P(\hat x - r_{\text{in}}) + r_{\text{out}},, B(\hat x - r_{\text{in}})\big)
> ]
> is again a SUF over (R) (possibly with a refined interval partition and modified constants in its primitive predicates).

Thus the SUF→FSS compiler can work entirely with SUF descriptors in the masked coordinate (\hat x), without ad‑hoc or incorrect “shift by (r_{\text{in}})” formulas.

### 3.4 Structure‑aware SUF→FSS compilation

We now give a *structure‑aware* compilation of masked SUFs to a small number of backend programs. The goal is to exploit the SUF’s interval and polynomial structure instead of materializing a full (2^n) table. Conceptually, the compilation produces:

* a **predicate program** (\Pi_{\mathsf{pred}}) that outputs all primitive predicate bits needed to evaluate the Boolean formulas and determine the active interval, and
* a **coefficient program** (\Pi_{\mathsf{coeff}}) that outputs the polynomial coefficients for the active interval.

Both programs are evaluated on the *masked* input (\hat x). The actual polynomial evaluation takes place locally on additive shares of the *unmasked* input (x), which come from the linear layer (or are obtained via a masked‑to‑shares protocol in §3.5).

#### 3.4.1 Collect primitive predicates and thresholds

Given a SUF descriptor for (\widehat{F}), we enumerate all *distinct* primitive predicates that appear in:

* the Boolean formulas (B_i(x)), and
* the interval tests (\alpha_i \le x < \alpha_{i+1}).

Let these be:

* (C_\beta(x) = \mathbf{1}[x<\beta]),
* (D_{\gamma,f}(x) = \mathbf{1}[x\bmod 2^f < \gamma]),
* (\mathrm{MSB}(x+c)),

for various (\beta,\gamma,f,c). Let the total number of distinct primitives be (T).

Using the mask‑aware rewrites of §3.3, we turn each primitive into a Boolean formula over:

* (\mathbf{1}[\hat x < \theta]) for a finite set of thresholds (\Theta),
* (\mathbf{1}[\hat x \bmod 2^f < \theta']) for sets (\Gamma_f),
* (\mathrm{MSB}(\hat x + c')) for a set (\mathcal C).

Interval membership (\mathbf{1}[x\in I_i]) is expressed as
[
\mathbf{1}[x<\alpha_{i+1}] \wedge \neg \mathbf{1}[x<\alpha_i],
]
and then rewritten in terms of the same primitive bits.

We assign each primitive a position in a global bit vector
[
\mathbf{b}(\hat x) = (b_1(\hat x),\dots,b_T(\hat x)) \in {0,1}^T.
]

#### 3.4.2 Predicate program (\Pi_{\mathsf{pred}})

We define a backend program (\Pi_{\mathsf{pred}}) whose payload is the vector of all primitive predicate bits:

* For each (\theta\in\Theta): one bit (b_\theta = \mathbf{1}[\hat x < \theta]).
* For each (f) and (\theta'\in\Gamma_f): one bit (b_{f,\theta'} = \mathbf{1}[\hat x\bmod 2^f < \theta']).
* For each (c'\in\mathcal C): one bit (m_{c'} = \mathrm{MSB}(\hat x + c')).

Concretely, this is implemented using the backend’s packed comparison templates of §3.1. Evaluating (\Pi_{\mathsf{pred}}) on (\hat x) yields additive shares of the primitive predicate vector (\mathbf{b}(\hat x)). From these bits, each party can locally (and in secret‑shared form):

1. Evaluate all mask‑rewritten primitive predicates (e.g., (\mathbf{1}[x<\beta])) as Boolean formulas in the bits.
2. Derive secret‑shared interval indicators (\mathbf{1}[x\in I_i]).
3. Derive secret‑shared Boolean outputs (B(x)) for the active interval by evaluating the Boolean formulas (B_i) using standard Boolean MPC (XOR + AND with Beaver triples).

#### 3.4.3 Coefficient program (\Pi_{\mathsf{coeff}})

The arithmetic SUF outputs are given by the piecewise polynomials
[
P_i(x) = (P_{i,1}(x),\dots,P_{i,r}(x)),\qquad
P_{i,j}(x) = \sum_{k=0}^{d} a_{i,j,k} x^k.
]

We define a payload for each interval (I_i):
[
v[i] = (a_{i,j,k})_{j\in[r],,k\in{0,\dots,d}} \in R^{r(d+1)}.
]

Using the masking invariant (\hat x = x + r_{\text{in}}), we convert each interval (I_i = [\alpha_i,\alpha_{i+1})) into its image (\hat I_i = [\alpha_i + r_{\text{in}}, \alpha_{i+1} + r_{\text{in}})\pmod{2^n}) in the (\hat x) domain. Each (\hat I_i) is either a single interval or the union of two wrap‑around intervals; in the latter case we split it into two. Overall we obtain a partition of ([0,2^n)) into at most (2m) disjoint intervals ([\hat p[t],\hat q[t]]), each with an associated payload (v[t]) corresponding to some original (I_i).

We then program an *interval‑lookup* program (\Pi_{\mathsf{coeff}}) such that, on input (\hat x), it returns additive shares of (v[t^*]) for the unique (t^*) such that (\hat x \in [\hat p[t^*],\hat q[t^*]]). This uses the backend’s interval/LUT template.

After evaluating (\Pi_{\mathsf{coeff}}), each party has additive shares of all polynomial coefficients (a_{i,j,k}) for the correct interval (i), but does *not* learn which interval index was selected.

#### 3.4.4 Local polynomial evaluation and correctness

Given the outputs of (\Pi_{\mathsf{pred}}) and (\Pi_{\mathsf{coeff}}), both parties proceed as follows:

1. Use the masked‑to‑shares protocol of §3.5 (if necessary) to convert the public masked input (\hat x) and mask shares ((r_{\text{in},0},r_{\text{in},1})) into additive shares ((x_0,x_1)) of the unmasked input (x).
2. For each arithmetic output (j), evaluate
   [
   y_j = \sum_{k=0}^d a_{i,j,k} x^k
   ]
   on shares, using Horner’s rule and Beaver triples. Powers of (x) can be shared across outputs to reduce the number of multiplications.
3. Add the corresponding output mask:
   [
   \hat y_j = y_j + r_{\text{out},j}.
   ]

By construction of the SUF and the payloads, this yields exactly the masked arithmetic outputs of (\widehat{F}); Boolean outputs are obtained directly from the predicate bits and Boolean MPC.

> **Lemma 3.2 (SUF→FSS correctness, masked case).**
> Let (F\in\mathsf{SUF}*n^{r,\ell,d}) and masks ((r*{\text{in}},r_{\text{out}})). Let (\widehat{F}) be its masked variant and (\Pi_{\mathsf{pred}},\Pi_{\mathsf{coeff}}) be the programs produced by the structure‑aware SUF→FSS compilation above. Let (\hat x = x + r_{\text{in}}).
> After running masked‑to‑shares on (\hat x), evaluating (\Pi_{\mathsf{pred}},\Pi_{\mathsf{coeff}}), and performing the local post‑processing described above, the parties obtain arithmetic and Boolean shares that reconstruct to
> [
> \widehat{F}(\hat x) = \big(P(x)+r_{\text{out}},,B(x)\big).
> ]

---

### 3.5 Masked↔shares conversions

The masked‑wire invariant involves *both* representations of a value (x):

* additive shares ((x_0,x_1)) with (x = x_0 + x_1 \pmod{2^n}), used by the arithmetic layer; and
* a public masked value (\hat x = x + r_{\text{in}}) together with mask shares ((r_{\text{in},0},r_{\text{in},1})), used by the FSS layer.

We therefore need *both* directions of conversion:

* **shares→masked**: from ((x_0,x_1)) to (\hat x) and mask shares; and
* **masked→shares**: from (\hat x) and mask shares back to ((x_0,x_1)).

These conversions are extremely simple; we spell them out for completeness.

#### 3.5.1 Shares→masked protocol

Assume:

* The parties hold additive shares ((x_0,x_1)) of (x\in\mathbb{Z}_{2^n}).
* They want to derive a fresh mask (r_{\text{in}}), its shares ((r_{\text{in},0},r_{\text{in},1})), and a public masked value (\hat x = x + r_{\text{in}}).

> **Protocol 3.3 (Shares→masked conversion).**
> Inputs: party (P_b) holds (x_b).
>
> 1. Each party (P_b) samples a uniform random share (r_{\text{in},b} \in \mathbb{Z}*{2^n}). Set
     >    [
     >    r*{\text{in}} := r_{\text{in},0} + r_{\text{in},1} \pmod{2^n}.
     >    ]
> 2. Each party computes a local masked share
     >    [
     >    \hat x_b := x_b + r_{\text{in},b} \pmod{2^n},
     >    ]
     >    broadcasts (\hat x_b), and reconstructs
     >    [
     >    \hat x = \hat x_0 + \hat x_1 = x_0 + x_1 + r_{\text{in},0} + r_{\text{in},1}
     >    = x + r_{\text{in}}.
     >    ]

> **Lemma 3.3 (Shares→masked correctness and privacy).**
> Correctness is immediate from the calculation above. For privacy, note that:
>
> * (r_{\text{in},0},r_{\text{in},1}) are uniform and independent of (x);
> * the broadcasted values (\hat x_b = x_b + r_{\text{in},b}) reveal only (\hat x), which is intended to be public;
> * the mask shares remain information‑theoretically hidden from the other party.

> Thus Protocol 3.3 reveals *exactly* the masked value (\hat x) and no additional information about the underlying (x).

In our overall design, a linear layer outputs ((x_0,x_1)); before feeding its result into a non‑linear gate, we apply Protocol 3.3 to obtain the masked value (\hat x) and mask shares ((r_{\text{in},0},r_{\text{in},1})), establishing the invariant required by the SUF→FSS layer.

#### 3.5.2 Masked→shares protocol

Now assume the inverse situation:

* The masked input (\hat x \in \mathbb{Z}_{2^n}) is public.
* The mask is shared as (r_{\text{in}} = r_{\text{in},0} + r_{\text{in},1} \pmod{2^n}), with each party (P_b) holding (r_{\text{in},b}).
* The true input is (x = \hat x - r_{\text{in}} \pmod{2^n}).

Our goal is to derive additive shares ((x_0,x_1)) of (x) such that (x = x_0 + x_1), using only local computation and no communication.

> **Protocol 3.4 (Masked→shares conversion).**
> Inputs: public (\hat x); party (P_0) holds (r_{\text{in},0}); party (P_1) holds (r_{\text{in},1}).
>
> 1. (P_0) sets
     >    [
     >    x_0 := \hat x - r_{\text{in},0} \pmod{2^n}.
     >    ]
> 2. (P_1) sets
     >    [
     >    x_1 := - r_{\text{in},1} \pmod{2^n}.
     >    ]

> **Lemma 3.4 (Masked→shares correctness and privacy).**
> We have
> [
> x_0 + x_1
> = (\hat x - r_{\text{in},0}) + (-r_{\text{in},1})
> = \hat x - (r_{\text{in},0}+r_{\text{in},1})
> = \hat x - r_{\text{in}}
> = x.
> ]
> If (r_{\text{in},0},r_{\text{in},1}) are uniformly random subject to (r_{\text{in},0} + r_{\text{in},1} = r_{\text{in}}), then conditioned on the public (\hat x), each share (x_b) is uniform from the other party’s point of view, so the protocol reveals nothing about (x) beyond (\hat x).

In our SUF→FSS design, each non‑linear gate first *optionally* applies Protocol 3.4 to obtain ((x_0,x_1)) if it does not already have them from the previous layer, then runs the predicate/coeff programs on (\hat x) and evaluates the SUF polynomials on ((x_0,x_1)).

### 3.6 Complexity of structure‑aware SUF→FSS

We summarize the complexity of the structure‑aware compilation for a SUF with parameters:

* input bit‑width (n),
* number of intervals (m),
* max polynomial degree (d),
* number of arithmetic outputs (r),
* number of Boolean outputs (\ell),
* number of distinct primitive predicates (T) (before masking),
* number of Boolean AND gates (A) in all (B_i) formulas.

Let (K_{\mathrm{FSS}}(n)) be the key size of the underlying backend for domain ({0,1}^n), and (E_{\mathrm{FSS}}(n)) its per‑input evaluation cost. For tree‑based DPF/DCF, (K_{\mathrm{FSS}}(n), E_{\mathrm{FSS}}(n) = \Theta(\lambda n)).

**Key sizes (per SUF‑gate type).**

* Predicate program:
  [
  \mathrm{keysize}(\Pi_{\mathsf{pred}})
  = O\big(K_{\mathrm{FSS}}(n) + \lambda T\big),
  ]
  since we need one control structure plus (O(T)) bits of payload.
* Coefficient program:
  [
  \mathrm{keysize}(\Pi_{\mathsf{coeff}})
  = O\big(K_{\mathrm{FSS}}(n) + \lambda r(d+1)\big),
  ]
  as the payload group has bit‑length (r(d+1)n) and multi‑word payloads only add a linear overhead in the payload size.

Additionally, we need at most (rd) Beaver triples for polynomial evaluation and (A) triples for Boolean ANDs.

**Online evaluation cost (per gate instance).**

* Two backend evaluations:

    * Predicate program: (E_{\mathrm{FSS}}(n) + O(|\Theta| + \sum_f|\Gamma_f|)).
    * Coefficient program: (E_{\mathrm{FSS}}(n) + O(m)) for the interval logic.
* Local work:

    * Boolean MPC: (O(T + \ell)) XORs and (A) ANDs with triples.
    * Arithmetic MPC: (O(rd)) ring additions and (rd) multiplications (each consuming one triple).

Critically, **all costs are polynomial** in the SUF description size; there is *no* dependence on (2^n) as in a naive LUT‑based compilation. For the transformer non‑linearities of interest, (m,d,T) are all modest (e.g., splines with (m\le 8), (d\le 3)), so the per‑gate overhead is moderate.

If the backend is instantiated with a *traditional single‑output DPF* without vector payloads, one would need a separate invocation for each predicate bit and each coefficient word, paying ((T + r(d+1))) FSS evaluations instead of 2; our analysis then essentially recovers the same complexity profile as a SIGMA‑style per‑primitive pipeline. If, as is typical in practice, the backend supports multi‑word payloads (vector‑DPF/DCF), then the SUF→FSS design achieves the 1–2 evaluations per gate claimed above.

---

## 4 Composite‑FSS Gates

We now explain how SUFs and the SUF→FSS backend fit into a gate‑level 2PC protocol. We first define the gate interface and its correctness/security, then instantiate it for ReluARS (§4.4) and give a fully expanded GeLU example (§4.5).

### 4.1 Gate interface

We work with additive secret sharing over (R=\mathbb{Z}_{2^n}). A wire value (x) is stored as shares ((x_0,x_1)) with (x = x_0 + x_1 \bmod 2^n). Linear operations are done locally; multiplications use Beaver triples generated in preprocessing.

For non‑linear gates, we additionally maintain a masked value
[
\hat x = x + r_{\text{in}},
]
where (r_{\text{in}}) is a mask sampled (and secret‑shared) during preprocessing. The masked value (\hat x) is reconstructed or refreshed as needed (via shares→masked) and serves as the input to the FSS backend.

A **Composite‑FSS gate type** (\tau) with parameters (\gamma) (e.g., bit‑width, scaling factors, approximation choice) is specified by:

* A SUF (F_{\tau,\gamma} \in \mathsf{SUF}_n^{r,\ell,d}) describing how the gate acts on a *single* input value (x).
* A key generation algorithm
  [
  \mathsf{CompGen}*\tau(\gamma,1^\lambda) \to
  (k*{\tau,\gamma,0}, k_{\tau,\gamma,1})
  ]
  that:

    1. samples masks ((r_{\text{in}}, r_{\text{out}})) for the gate’s arithmetic channels,
    2. builds the SUF descriptor of (\widehat{F}*{\tau,\gamma} = F*{\tau,\gamma}^{[r_{\text{in}},r_{\text{out}}]}),
    3. invokes the SUF→FSS compiler to obtain the predicate and coefficient programs for this SUF,
    4. packages these programs, masks, and any additional linear‑layer metadata (e.g., small LUTs embedded in the SUF) into the gate keys.
* Per‑party evaluation algorithms
  [
  \mathsf{CompEval}*{\tau,b}(k*{\tau,\gamma,b}, \hat{x}, x_b, \text{aux}_b) \to
  \left(\hat{y}_b, z_b, \text{aux}'_b\right),
  ]
  where (\hat{x}) is the public masked input, (x_b) is party (b)’s additive share of the unmasked input, (\hat{y}_b) is an arithmetic share of the *masked* output, (z_b) is a Boolean share vector of helper bits, and (\text{aux}_b) captures local state (e.g., cached triple indices).

Informally, (\mathsf{CompEval}) performs:

1. Two backend calls on (\hat{x}) to obtain shares of:

    * all primitive predicate bits and Boolean helper outputs, and
    * the polynomial coefficients for the active interval.
2. A fixed pattern of local linear operations and a *small number* of Beaver‑triple‑based multiplications to:

    * evaluate the polynomials on shares of (x),
    * implement any gate‑specific arithmetic logic (e.g., ARS rounding or GeLU correction),
    * and add masks, yielding the final masked arithmetic outputs and helper bits.

### 4.2 Gate correctness

Let (F_{\tau,\gamma} : R \to R^r \times {0,1}^\ell) be the ideal single‑input functionality for gate type (\tau), and let (F_{\tau,\gamma}^{[r_{\text{in}},r_{\text{out}}]}) be its masked SUF variant. Consider evaluating a single gate on an input wire (x) held as shares ((x_0,x_1)), with public masked value
[
\hat{x} = x + r_{\text{in}}.
]
Running ((\mathsf{CompEval}*{\tau,0},\mathsf{CompEval}*{\tau,1})) yields arithmetic outputs ((\hat{y}_0,\hat{y}_1)) and Boolean shares ((z_0,z_1)).

Correctness requires:
[
\hat{y}*0 + \hat{y}*1 = y + r*{\text{out}},\qquad
z_0 \oplus z_1 = b,
]
where ((y,b) = F*{\tau,\gamma}(x)).

> **Lemma 4.1 (Composite‑FSS gate correctness).**
> If SUF→FSS is correct (Lemma 3.2), the shares↔masked conversions are correct (Lemma 3.3–3.4), and the Beaver‑triple arithmetic layer realizes correct multiplications, then every Composite‑FSS gate type (\tau) satisfies the above correctness condition.

*Proof sketch.* The backend calls yield shares of the SUF outputs (P(x)+r_{\text{out}}) and (B(x)). Shares→masked and masked→shares conversions relate the representations of (x) and (\hat x) correctly. The remaining local computation in (\mathsf{CompEval}) consists only of linear operations and multiplications that implement known algebraic identities for the target function in terms of SUF outputs and helper bits. Additive sharing plus Beaver triples correctly implement these equations, hence the reconstructed values match (F_{\tau,\gamma}^{[r_{\text{in}},r_{\text{out}}]}(x)). ∎

### 4.3 Gate security

We adopt the standard indistinguishability‑based security definition for FSS‑based 2PC in the semi‑honest model. For each gate (\tau) we define a single‑gate *ideal functionality* (\mathcal{F}*\tau) that, on input masked value (\hat{x}) and secret shares of (x), internally samples ((r*{\text{in}},r_{\text{out}})), computes (F_{\tau,\gamma}^{[r_{\text{in}},r_{\text{out}}]}(x)), and returns additive shares of the output wire and Boolean bits to the two parties. An execution consisting of many such gates plus linear operations is then defined by standard sequential composition.

Assume:

* The backend is a secure FSS for the relevant function families: for any single program there exists a simulator that, given only the target function and its output on some input, can sample a view computationally indistinguishable from the adversary’s view of the key and evaluation.
* Beaver triple generation is modeled as an ideal functionality (or is provided by a standard semi‑honest secure precomputation protocol).

Then we obtain:

> **Theorem 4.2 (Composite‑FSS gate security, semi‑honest).**
> For any fixed gate type (\tau) and parameters (\gamma), under the above assumptions there exists a PPT simulator such that the real‑world view of a semi‑honest adversary corrupting either party in a single execution of (\mathsf{CompEval}*\tau) is computationally indistinguishable from its view in the ideal execution with (\mathcal{F}*\tau).

*Proof sketch.* Each Composite‑FSS execution consists of:

1. Local key generation using (\mathsf{CompGen}_\tau), which internally calls SUF→FSS and the backend’s programming interface.
2. Local evaluation steps that call the backend’s Eval on (\hat{x}), run shares↔masked conversions, and perform linear operations and triple‑based multiplications.

We replace each backend ((\mathsf{ProgGen},\mathsf{Eval})) pair by the ideal FSS oracle guaranteed by the backend’s security definition; by composition of FSS schemes this is simulatable for joint outputs. We then replace triple‑based multiplications with calls to the triple ideal functionality. Shares↔masked conversions are purely local and information‑theoretically secure given the public (\hat x). The resulting hybrid is exactly the ideal execution of (\mathcal{F}_\tau). Sequential composition over gates yields security of the whole circuit. ∎

We stress that, unlike SHARK, we *do not* attempt to obtain malicious security; we remain in the semi‑honest setting, closer to SIGMA. SHARK shows how to “lift” semi‑honest FSS protocols to malicious security via IFSS and authenticated secret sharing; our abstraction is compatible with such a lifting, but we do not rely on IFSS or authentication in the current work.

### 4.4 Example: ReluARS as SUF + Composite‑FSS

We briefly instantiate the framework for a representative fused gate: ReluARS. We model (\mathrm{ReluARS}_{n,f}) as a gate that, on fixed‑point input (x\in R) with (f) fractional bits, outputs:

* an arithmetic value
  [
  y = \mathrm{ReluARS}_{n,f}(x),
  ]
  typically something like (\left\lfloor \max(x,0) / 2^f \right\rfloor) plus small corrections, and
* three Boolean helper bits (w,t,d) capturing sign, rounding, and a small “gap correction” index, as in SHARK/SIGMA’s optimized protocol.

#### 4.4.1 ReluARS helper SUF

We define a SUF (F_{\mathsf{ReluHelp}} \in \mathsf{SUF}_n^{1,3,1}) capturing the common structure:

* **Intervals.**
  Two intervals split at the sign boundary:
  [
  I_0 = [0, 2^{n-1})\quad(\text{non‑negative}),\qquad
  I_1 = [2^{n-1}, 2^n)\quad(\text{negative}).
  ]

* **Arithmetic polynomials.**
  [
  P_0(x) = x,\quad P_1(x) = 0,
  ]
  so the arithmetic output is (x^+ = \max(x,0)).

* **Boolean outputs.**

    * Sign bit:
      [
      w(x) = \mathbf{1}[x\ge 0] = \neg \mathrm{MSB}(x).
      ]
    * Rounding bit:
      [
      t(x) = \mathbf{1}[x\bmod 2^f \ge 2^{f-1}] = \neg \mathbf{1}[x\bmod 2^f < 2^{f-1}].
      ]
    * Gap‑correction selector (d(x)), defined as a Boolean combination of low‑bit predicates on (x\bmod 2^f), matching the choice in SHARK/SIGMA.

Thus
[
F_{\mathsf{ReluHelp}}(x) = (x^+(x),,w(x),t(x),d(x)).
]

By Proposition 3.1 its masked variant (\widehat{F}_{\mathsf{ReluHelp}}) remains a SUF and can be compiled via SUF→FSS.

#### 4.4.2 Composite‑FSS ReluARS gate

The Composite‑FSS ReluARS gate (\mathsf{G}_{\mathsf{ReluARS}}) is defined as:

* **Key generation** (\mathsf{CompGen}_{\mathsf{ReluARS}}):

    1. Sample masks ((r_{\text{in}},r_{\text{out}})) for the single arithmetic output.
    2. Build the SUF descriptor of (\widehat{F}_{\mathsf{ReluHelp}}).
    3. Run SUF→FSS to obtain predicate and coefficient programs.
    4. Store these programs, the masks (or their shares), and the parameters of the small 8‑entry correction table indexed by ((w,t,d)) in the gate keys.

* **Evaluation** (\mathsf{CompEval}_{\mathsf{ReluARS},b}) on party (b) with public (\hat x) and share (x_b):

    1. If necessary, run masked→shares (§3.5.2) on (\hat x) and (r_{\text{in},b}) to derive (x_b).
    2. Evaluate (\Pi_{\mathsf{pred}},\Pi_{\mathsf{coeff}}) at (\hat x), obtaining shares of:

        * primitive predicate bits,
        * helper bits (w,t,d),
        * polynomial coefficients (trivial here).
    3. Locally evaluate (x^+) using Horner’s rule (degree 1) on shares of (x).
    4. Compute the truncated value (q = \lfloor x^+/2^f \rfloor) using the known truncation circuit with helper bit (t).
    5. Using ((w,t,d)), look up and add the correction term (\Delta[w,t,d]) in a small LUT realized as a local Boolean circuit.
    6. Add (r_{\text{out}}) to obtain (\hat y_b).

The combined output (\hat y_0 + \hat y_1) reconstructs to (\mathrm{ReluARS}*{n,f}(x) + r*{\text{out}}), and the helper bits reconstruct to the intended SHARK/SIGMA helper bits. The per‑gate FSS cost is:

* one predicate program evaluation (small (T) and (m=2)),
* one coefficient program evaluation (degree‑1 polynomials),
* a constant number of Beaver triples for truncation and LUT logic.

### 4.5 Fully‑expanded example: GeLU spline as SUF → (\Pi_{\mathsf{pred}},\Pi_{\mathsf{coeff}}) → Composite‑FSS

We now work out one gate in detail: a GeLU‑style activation implemented via splines. This section is deliberately explicit to illustrate how SUF descriptions, the SUF→FSS compiler, and Composite‑FSS gates interact in an actual instantiation, and how the resulting complexity compares to SIGMA/SHARK‑style hand‑written protocols (under the *same* DPF/DCF backend).

#### 4.5.1 GeLU SUF description

Let (x \in R) be a fixed‑point value with (f) fractional bits. Let (T > 0) be a clipping bound (e.g., (T \approx 4\cdot 2^f)). We approximate GeLU by:

* a central spline on ([-T,T]) using degree‑(d) polynomials ((d\le 3)),
* linear/constant tails outside ([-T,T]).

We define:

* Interval boundaries (in the signed interpretation)
  [
  -T = a_0 < a_1 < \dots < a_m = T,
  ]
  inducing central intervals (J_i = [a_i,a_{i+1})) for (i=0,\dots,m-1).

* An approximate GeLU (\widetilde{\mathrm{GeLU}}) by
  [
  \widetilde{\mathrm{GeLU}}(x) =
  \begin{cases}
  0              & x \le a_0,\
  Q_i(x)         & x \in J_i,\
  x              & x \ge a_m,
  \end{cases}
  ]
  where each (Q_i(x) = \sum_{k=0}^d c_{i,k} x^k) is a spline polynomial.

We express this as a SUF (F_{\mathsf{GeLU}} \in \mathsf{SUF}_n^{2,\ell,d}) with two arithmetic channels and a small number of Boolean helpers.

* **Intervals.**
  Three types:

    * left tail (I_{\text{left}} = (-\infty, a_0)),
    * central spline intervals (I_i = J_i = [a_i,a_{i+1})),
    * right tail (I_{\text{right}} = [a_m,\infty)),
      realized in (\mathbb{Z}_{2^n}) via comparisons and sign tests.

* **Arithmetic outputs.**
  We expose:

    1. (x^+(x) = \max(x,0)), implemented as in ReLU.
    2. A spline correction (\delta(x)):

        * (\delta(x) = 0) on (I_{\text{left}}) and (I_{\text{right}}),
        * (\delta(x) = Q_i(x)) on (I_i).

  The final GeLU‑like activation is
  [
  y = x^+ + \delta(x).
  ]

* **Boolean outputs.**
  We include helper bits:

    * (w = \mathbf{1}[x\ge 0]) (sign),
    * (c = \mathbf{1}[|x| < T]) (central‑region flag),
    * optionally a compact encoding of the spline interval index (i).

All predicates are Boolean formulas over (\mathbf{1}[x<\beta]) and MSB tests, so (F_{\mathsf{GeLU}}) is a SUF.

#### 4.5.2 Compiling GeLU SUF to (\Pi_{\mathsf{pred}}) and (\Pi_{\mathsf{coeff}})

We apply the SUF→FSS compiler of §3.4 to the masked variant (\widehat{F}_{\mathsf{GeLU}}).

**Step 1: collect primitives.**
From the Boolean formulas and interval tests we collect:

* comparisons (\mathbf{1}[x<a_0],\dots,\mathbf{1}[x<a_m]),
* sign tests (\mathrm{MSB}(x)),
* possibly additional low‑bit comparisons if we encode the interval index (i).

Let the total number of distinct primitives be (T_{\text{GeLU}}) (typically (T_{\text{GeLU}} \approx m+2)).

Using §3.3 we rewrite each primitive in terms of:

* primitive comparisons (\mathbf{1}[\hat x < \theta]) with thresholds (\theta\in\Theta),
* MSB shifts (\mathrm{MSB}(\hat x + c')) with constants (c'\in\mathcal C).

We assign each primitive an index in the global predicate vector (\mathbf{b}(\hat x)).

**Step 2: predicate program (\Pi_{\mathsf{pred}}^{\mathsf{GeLU}}).**
We program a backend instance so that on input (\hat x) it outputs the entire predicate vector (\mathbf{b}(\hat x)) in packed form (e.g., (\lceil T_{\text{GeLU}}/64\rceil) machine words). Evaluating this once at run time gives both parties additive shares of all primitive bits. From these, they derive:

* helper bits (w) and (c),
* interval indicators (\mathbf{1}[x\in I_{\text{left}}]), (\mathbf{1}[x\in I_i]), (\mathbf{1}[x\in I_{\text{right}}]).

**Step 3: coefficient program (\Pi_{\mathsf{coeff}}^{\mathsf{GeLU}}).**
For each interval we define the polynomial coefficients:

* left/right tails: (Q_{\text{left}}(x)=0), (Q_{\text{right}}(x)=x),
* central intervals: (Q_i(x)) with coefficients (c_{i,k}).

We then define payloads:

* (v[\text{left}] = (0,\dots,0)),
* (v[i] = (c_{i,0},\dots,c_{i,d})) for each central interval (i),
* (v[\text{right}] =) coefficients of the identity polynomial (x\mapsto x).

As in §3.4.3 we map each interval to the masked coordinate (\hat x) (splitting wrap‑around intervals if needed) and program a backend interval‑lookup so that on input (\hat x) it returns additive shares of the coefficient tuple (v[i^*]) for the active interval.

#### 4.5.3 Composite‑FSS GeLU gate: offline and online

We now define the Composite‑FSS GeLU gate (\mathsf{G}_{\mathsf{GeLU}}).

**Offline (key generation) (\mathsf{CompGen}_{\mathsf{GeLU}}):**

1. The dealer samples masks ((r_{\text{in}},r_{\text{out}})) for each arithmetic channel ((x^+,\delta)) and shares them as ((r_{\text{in},0},r_{\text{in},1}), (r_{\text{out},0},r_{\text{out},1})).
2. It builds the SUF descriptor of (\widehat{F}_{\mathsf{GeLU}}) with these masks.
3. It runs the SUF→FSS compiler to obtain:

    * predicate backend keys for (\Pi_{\mathsf{pred}}^{\mathsf{GeLU}}),
    * coefficient backend keys for (\Pi_{\mathsf{coeff}}^{\mathsf{GeLU}}).
4. It packages these keys and masks into gate keys (k_{\mathsf{GeLU},0},k_{\mathsf{GeLU},1}) and distributes them to the parties.

**Online (evaluation) (\mathsf{CompEval}_{\mathsf{GeLU},b}) on party (b):**

Inputs: public (\hat x), share (x_b) of (x), gate key (k_{\mathsf{GeLU},b}), and access to Beaver triples.

1. If the wire only stores (\hat x) and mask shares, run masked→shares (§3.5.2) to derive (x_b).
2. Evaluate (\Pi_{\mathsf{pred}}^{\mathsf{GeLU}}) at (\hat x) to obtain additive shares of all primitive predicate bits. From these, compute shares of:

    * helper bits (w,c),
    * interval indicators.
3. Evaluate (\Pi_{\mathsf{coeff}}^{\mathsf{GeLU}}) at (\hat x) to obtain shares of the coefficient tuple (v[i^*]) for the active interval.
4. Using shares of (x) and the coefficients, evaluate the spline polynomial (\delta(x)) on shares via Horner’s rule (degree (d)), consuming (d) triples per output.
5. Evaluate (x^+) on shares (if not already available) using the ReLU helper bit from step 2 and Beaver triples.
6. Compute (y = x^+ + \delta(x)) on shares, then add (r_{\text{out}}) to get (\hat y_b).

Both parties return (\hat y_b) and any helper bits they need to expose (e.g., (w,c)) to subsequent gates.

#### 4.5.4 Complexity and comparison to SIGMA/SHARK

Let:

* (m) be the number of spline intervals (central pieces) for GeLU,
* (d \le 3) be the spline degree,
* (r=2) arithmetic channels ((x^+,\delta)),
* (T_{\text{GeLU}} \approx m+2) primitive predicates.

Then GeLU via Composite‑FSS has:

* **Key generation:**

    * one predicate program with key size (O(K_{\mathrm{FSS}}(n) + \lambda T_{\text{GeLU}})),
    * one coefficient program with key size (O(K_{\mathrm{FSS}}(n) + \lambda r(d+1))).

* **Online per‑gate cost:**

    * one evaluation of (\Pi_{\mathsf{pred}}^{\mathsf{GeLU}}),
    * one evaluation of (\Pi_{\mathsf{coeff}}^{\mathsf{GeLU}}),
    * (O(T_{\text{GeLU}})) local Boolean operations and a handful of ANDs with triples,
    * (O(d)) Beaver multiplications for spline evaluation and a few more for (x^+) and final combination.

Assuming a traditional tree‑based DPF/DCF backend shared with SIGMA/SHARK (so (E_{\mathrm{FSS}}(n) = \Theta(\lambda n)) for each call), we use **two** backend evaluations per GeLU gate.

By contrast, in a typical SIGMA‑style design, a GeLU would be realized as a *pipeline* of several separate FSS protocols: e.g. one DReLU, one truncation, one spline LUT, and one linear‑select protocol—3–4 backend calls per gate, all with cost (E_{\mathrm{FSS}}(n)). SHARK’s semi‑honest core has a similar structure before adding IFSS/authentication.

Under the *same backend*, our SUF‑based design therefore reduces the number of FSS invocations for GeLU by about a factor (2) (sometimes more), while keeping the asymptotic cost per invocation unchanged. If the backend did **not** support multi‑word payloads, one would instead need (T_{\text{GeLU}} + r(d+1)) separate calls per gate; this collapses back to a SIGMA‑style overhead and SUF becomes mainly a conceptual unification rather than an efficiency win.

### 4.6 Protocol summary and per‑gate performance

For each gate type (\tau) (ReluARS, GeLU, SiLU, reciprocal, etc.) with SUF parameters ((m_\tau,d_\tau,r_\tau,\ell_\tau,T_\tau,A_\tau)), the Composite‑FSS instantiation has:

* **Preprocessing:**

    * SUF→FSS key generation for:

        * one predicate program of size (O(K_{\mathrm{FSS}}(n) + \lambda T_\tau)),
        * one coefficient program of size (O(K_{\mathrm{FSS}}(n) + \lambda r_\tau(d_\tau+1))),
    * generation of at most (r_\tau d_\tau + A_\tau) Beaver triples.

* **Online, per gate instance:**

    * two backend evaluations at cost (2E_{\mathrm{FSS}}(n) + O(m_\tau + T_\tau)),
    * (O(T_\tau + \ell_\tau)) Boolean ops and (A_\tau) ANDs with triples,
    * (O(r_\tau d_\tau)) ring operations and (r_\tau d_\tau) multiplications with triples.

This should be compared *under the same backend* (same DPF/DCF implementation and parameters) to SIGMA/SHARK‑style per‑primitive pipelines, where each complex non‑linear gate typically invokes 3–5 separate FSS protocols. Our SUF‑based Composite‑FSS gates reduce the number of FSS calls per gate by a small constant factor, while keeping the security assumptions and masked‑wire invariant unchanged.

---

## 5 Conceptual Comparison with SIGMA and SHARK

We briefly position SUF and Composite‑FSS relative to SIGMA and SHARK along several axes: target function family, abstraction level, protocol structure, security model, and extensibility.

### 5.1 SIGMA

SIGMA is a semi‑honest 2PC system for GPT‑style transformer inference built on FSS. It introduces specialized FSS protocols for a set of fixed‑point non‑linearities (ReLU, ARS, ReluARS, GeLU, SiLU, reciprocal, softmax, LayerNorm) and reports strong end‑to‑end performance on BERT‑like models.

**Target functions.**
SIGMA’s non‑linear protocols are all built around the same pattern: express the function as a combination of:

* univariate piecewise polynomials on bounded intervals (splines),
* simple branch conditions (sign, comparisons),
* and small LUTs for residual corrections.

This structure is implicit in their per‑primitive designs.

**Abstraction level.**

* SIGMA presents each non‑linear primitive as a *standalone FSS protocol* (\Pi_F), with its own key format and evaluation logic.
* There is no single unifying function family; instead, the fact that all these functions are “piecewise‑polynomial with helper bits” is left implicit.

In our framework, the same functions are formalized uniformly as SUFs; non‑linear gate implementations are Composite‑FSS instantiations derived from a single generic compiler.

**Protocol structure.**

* SIGMA builds each primitive from a small set of FSS building blocks (DReLU, TR/ARS, LRS, generic LUTs, linear‑select), each realized via DPF/DCF under the hood.
* A complex non‑linearity like GeLU is then realized as a *pipeline* of several FSS protocols: e.g., ReLU + TR + LUT + select.

In contrast, Composite‑FSS fuses these layers at the SUF level and compiles them into *one* or a small constant number of backend programs that emit all required helper bits and arithmetic outputs in a single shot; only lightweight local post‑processing remains.

**Extensibility.**

* Adding a new non‑linear function in SIGMA requires designing and proving a new FSS protocol (\Pi_F), even though structurally it will again be a univariate piecewise polynomial with a few helper bits.
* In our framework, adding a new function is done by:

    1. specifying its SUF description (intervals, approximating polynomials, predicates), and
    2. plugging it into the existing SUF→FSS and Composite‑FSS machinery.

No new FSS security proof is needed beyond the generic theorem for SUF.

### 5.2 SHARK

SHARK is an actively secure, preprocessing‑based 2PC system built on FSS and *interactive* FSS (IFSS). It targets the same class of functions as SIGMA (ReLU, ARS, ReluARS, splines, reciprocal) but in a malicious security model, and reports large speedups over previous maliciously secure inference protocols.

**Target functions and structure.**

* SHARK’s target functions are again fixed‑point ReLU, ARS, ReluARS, splines, and reciprocal, which are all univariate piecewise polynomials with helper bits; its softmax/transformer support relies on these.
* The paper explicitly uses helper bits (w,t,d) and small tables for ReluARS and details how reciprocal is implemented via clipping and a mantissa/exponent LUT. These constructions fit exactly into our SUF formalization.

**Security model and IFSS.**

* SHARK lifts semi‑honest FSS‑based protocols to malicious security by introducing interactive FSS and combining it with authenticated secret sharing over rings and Booleans.
* IFSS schemes allow interaction during evaluation and can output auxiliary information (e.g., MAC tags) that is later checked to ensure integrity.

Our work deliberately *does not use* IFSS or authentication: we remain in the semi‑honest model and rely only on non‑interactive FSS (our PFSS backend) plus an unauthenticated Beaver‑triple layer. This makes our protocols conceptually closer to SIGMA than to SHARK, but our abstraction is designed so that it could be lifted along similar lines to SHARK’s IFSS‑based approach if malicious security is desired in the future.

**Abstraction and composition.**

* Like SIGMA, SHARK presents per‑primitive (I)FSS schemes for ReLU, ARS, ReluARS, splines, reciprocal, etc., each with its own cost analysis.
* SHARK’s main abstraction innovation lies in *security*: it shows how to define and prove security for IFSS and how to combine IFSS schemes with authenticated secret sharing.

Our abstraction innovation is *functional*: we isolate the structured function family (SUF), give a generic SUF→FSS compiler, and define Composite‑FSS gates whose security is proved generically over that family. In other words, SHARK unifies the *security lifting* story, while we unify the *function‑family and compilation* story.

**Efficiency.**

* For key primitives, SHARK’s IFSS protocols reduce AES calls and preprocessing size relative to earlier actively secure protocols.
* Our Composite‑FSS cost analysis (built on the same class of DPF/DCF cores, with the same (K_{\mathrm{FSS}}(n)), and assuming a shared backend) shows that for the *semi‑honest* setting, we can match or improve the per‑gate constants by folding multiple helper functions into one or two backend programs—e.g., for ReluARS and GeLU our analytical estimates indicate about (2\times) reductions in key size and (4!-!6\times) reductions in AES calls for typical parameters such as (n=64,f=12).

It is important to interpret these comparisons correctly:

* SHARK has additional overheads due to malicious‑security machinery (MACs, consistency checks, IFSS interaction), which we do **not** bear.
* Our analytical comparisons therefore focus on the *FSS core* for the same functionality; they suggest that even in a semi‑honest setting, SUF‑based Composite‑FSS can realize the same gates with strictly fewer backend calls than both SIGMA’s per‑primitive pipelines and SHARK’s IFSS protocols, when all three are instantiated on the *same* DPF/DCF backend.

### 5.3 Summary of differences

| Axis                 | SIGMA                                                                  | SHARK                                                      | This work (SUF + Composite‑FSS)                                                 |
| -------------------- | ---------------------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Threat model         | Semi‑honest                                                            | Malicious 2PC w/ preprocessing                             | Semi‑honest (but compatible with lifting)                                       |
| Core primitive       | Non‑interactive FSS (DPF/DCF)                                          | Interactive FSS + authenticated sharing                    | Non‑interactive FSS backend (PFSS interface)                                    |
| Target functions     | Transformer non‑linearities (ReLU, ARS, GeLU, SiLU, softmax, LN, etc.) | Same plus reciprocal for softmax, under malicious security | Same set, formalized in one SUF family                                          |
| Abstraction          | Per‑primitive FSS protocols (no unified IR)                            | Per‑primitive IFSS schemes + unified malicious lifting     | Unified SUF function family + SUF→FSS compiler + Composite‑FSS gate abstraction |
| Use of helper bits   | Function‑specific (DReLU, TR, LUT, select)                             | Function‑specific (IFSS with auxiliary info)               | Unified helper‑bit predicates in SUF; shared across gates/layers                |
| Adding new primitive | Design new FSS protocol + proof                                        | Design new IFSS scheme + malicious proof                   | Specify SUF description; security follows from generic SUF theorem              |
| Role of structure    | Implicit in per‑primitive formulas                                     | Explicit in security lifting (IFSS)                        | Explicit in function family, compilation, and gate fusion                       |

---

## 6 Role of SUF in the Overall Design

To make the perspective crystal‑clear:

* **What is SUF in our design?**
  SUF is a *mathematical function family* of structured univariate functions over (\mathbb{Z}_{2^n}), encompassing both arithmetic outputs and Boolean helper bits. It is *not* tied to any particular implementation (DPF, DCF, IFSS); it is the semantic target that all of our non‑linear gates implement.

* **What role does SUF play?**
  SUF is the *interface* between:

    1. high‑level, approximate fixed‑point descriptions of non‑linearities (piecewise polynomials, LUT‑based corrections, helper bits), and
    2. the low‑level FSS implementation (our PFSS‑style backend) plus the Beaver‑triple arithmetic layer.

  It allows us to:

    * prove once that ReLU/ARS/ReluARS/GeLU/SiLU/reciprocal/rsqrt/nExp/softmax‑block are all SUFs (or SUF‑compatible) for fixed precision;
    * write a single SUF→FSS compiler and a single Composite‑FSS security theorem that applies to all of them;
    * systematically exploit shared structure across gates (e.g., global helper‑bit layouts, layer‑level batching) because everything is described in the same IR.

* **How is this different in spirit from SIGMA and SHARK?**
  SIGMA and SHARK both *implicitly* exploit structured univariate behavior: they design efficient FSS/IFSS protocols by hand for each such function. However:

    * They do **not** abstract this into a common function family; each protocol’s security and efficiency analysis is bespoke.
    * SIGMA optimizes semi‑honest efficiency; SHARK optimizes malicious lifting via IFSS. Neither introduces an intermediate IR whose main purpose is to *unify* these primitives and enable generic compilation.
    * Our SUF + Composite‑FSS framework is therefore more of a *programming and compilation abstraction* on top of the same FSS “hardware”. It is orthogonal to SHARK’s IFSS‑based security lifting and to SIGMA’s engineering optimizations.

In short, SUF is the “language of non‑linearity” our system speaks. Composite‑FSS is the “calling convention” that connects SUFs to the PFSS‑style backend and to the secret‑sharing arithmetic layer. Together they give a clean, theoretically justified way to say:

> **Any univariate, piecewise‑polynomial non‑linearity used in transformers can be described once as a SUF, compiled automatically to a small number of FSS programs built from standard DPF/DCF templates, and then plugged into a generic Composite‑FSS gate whose correctness and privacy reduce to standard FSS assumptions.**

---

## 7 Prototype Mapping (This Repository)

This paper is accompanied by a prototype implementation in this repo. The main conceptual components map to code as follows:

- **SUF IR + semantics**: `include/suf/` (IR, mask rewrite, `ref_eval`, validation)
- **SUF→PFSS compiler + analyses**: `include/compiler/`, `src/compiler/` (PFSS program descriptions, range/gap analysis, trunc lowering, layer graph passes)
- **Composite‑FSS gates**: `include/gates/` (nonlinear gates and post-processing) and `include/gates/composite_fss.hpp`
- **Runtime scheduling + batching**: `include/runtime/`, `src/runtime/` (`PhaseExecutor`, `PfssSuperBatch`, `OpenCollector`, planners, async/staged execution)
- **Transformer stack**: `include/nn/`, `src/nn/` (attention/MLP/transformer layer, `LayerContext`)
- **Backends**: `include/proto/` (clear, myl7/fss adapter, SigmaFast, GPU) and `cuda/` (CUDA PFSS kernels + backend)
- **Tests/benches**: `src/demo/`, `src/bench/`

Build and test instructions are kept in `README.md`.
