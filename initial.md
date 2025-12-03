Below is a **full, implementable codebase plan (CPU + GPU)** for your **SUF → PFSS compiler + Composite-FSS gates** design. Everything is **in English**, and structured so you can let Codex fill in backend-specific crypto details while still having a working end-to-end system (with an **insecure cleartext backend for functional testing** and a **secure backend adapter interface** for plugging in Grotto/SIGMA/SHARK-style DPF/DCF engines).

---

## 0) Design goals encoded into the repo

1. **Separation of concerns**

    * `suf_ir/`: SUF language (intervals, polynomials, predicates, boolean formulas).
    * `compiler/`: SUF→FSS compilation (masked rewrite, predicate program, coefficient program).
    * `pfss/`: abstract programmable-FSS interface (ProgGen/Eval), plus adapters.
    * `mpc/`: additive ring sharing, Beaver triples, Boolean MPC.
    * `gates/`: Composite-FSS gate types (ReLU/ReluARS/GeLU/…).
2. **Two modes**

    * `CLEARTEST` mode: no crypto; dealer stores the desc, parties get random shares. Used to validate correctness of SUF compilation + gate algebra.
    * `SECURE` mode: same APIs, but PFSS adapter calls a real DPF/DCF implementation.
3. **CPU + GPU**

    * CPU: reference eval, batching, multithreading, AES-NI/ChaCha PRG hooks (backend-defined).
    * GPU: batched PFSS eval and polynomial evaluation kernels (backend-defined), with a clean ABI.

---

## 1) Repository layout

```text
suf-fss/
  CMakeLists.txt
  include/
    core/
      types.hpp
      ring.hpp
      serialization.hpp
    mpc/
      shares.hpp
      beaver.hpp
      boolean_mpc.hpp
      arithmetic_mpc.hpp
      masked_wire.hpp
    suf/
      suf_ir.hpp
      predicates.hpp
      bool_expr.hpp
      polynomial.hpp
      fixedpoint.hpp
    compiler/
      mask_rewrite.hpp
      suf_collect.hpp
      suf_to_pfss.hpp
    pfss/
      pfss.hpp
      program_desc.hpp
      backend_cleartext.hpp
      backend_external_adapter.hpp
    gates/
      gate_api.hpp
      relu_gate.hpp
      reluars_gate.hpp
      gelu_spline_gate.hpp
  src/
    demo/
      demo_relu.cpp
      demo_gelu.cpp
  cuda/                 # optional
    pfss_cuda_api.hpp
    pfss_kernels.cu
    poly_kernels.cu
```

---

## 2) Core types: ring arithmetic and shares (CPU)

### `include/core/ring.hpp`

```cpp
#pragma once
#include <cstdint>
#include <type_traits>

namespace core {

// Ring: Z_{2^n}. If Bits=64, uint64 overflow is already mod 2^64.
template<int Bits>
struct Z2n {
  static_assert(Bits >= 1 && Bits <= 64);
  using word = uint64_t;

  word v;

  static constexpr word mask() {
    if constexpr (Bits == 64) return ~word(0);
    return (word(1) << Bits) - 1;
  }

  static inline word norm(word x) {
    if constexpr (Bits == 64) return x;
    return x & mask();
  }

  Z2n() : v(0) {}
  explicit Z2n(word x) : v(norm(x)) {}

  friend inline Z2n operator+(Z2n a, Z2n b) { return Z2n(norm(a.v + b.v)); }
  friend inline Z2n operator-(Z2n a, Z2n b) { return Z2n(norm(a.v - b.v)); }
  friend inline Z2n operator*(Z2n a, Z2n b) { return Z2n(norm(a.v * b.v)); }

  Z2n& operator+=(Z2n o){ v = norm(v + o.v); return *this; }
  Z2n& operator-=(Z2n o){ v = norm(v - o.v); return *this; }
  Z2n& operator*=(Z2n o){ v = norm(v * o.v); return *this; }
};

} // namespace core
```

### `include/mpc/shares.hpp`

```cpp
#pragma once
#include "core/ring.hpp"
#include <cstdint>

namespace mpc {

template<typename RingT>
struct AddShare { RingT s; };        // x = x0 + x1 mod 2^n

struct XorShare { uint8_t b; };      // bit share: b = b0 XOR b1

} // namespace mpc
```

---

## 3) Beaver triples (arithmetic + boolean)

### `include/mpc/beaver.hpp`

```cpp
#pragma once
#include "mpc/shares.hpp"
#include <random>

namespace mpc {

// Arithmetic Beaver triple: a,b uniform; c=a*b in ring. Shared additively.
template<typename RingT>
struct BeaverTripleA {
  AddShare<RingT> a, b, c;
};

// Boolean Beaver triple for AND over GF(2): u,v uniform bits; w=u&v. Shared by XOR.
struct BeaverTripleB {
  XorShare u, v, w;
};

// Dealer-side generation (in preprocessing model)
template<typename RingT, class URBG>
inline std::pair<BeaverTripleA<RingT>, BeaverTripleA<RingT>>
dealer_make_tripleA(URBG& g) {
  using W = typename RingT::word;
  std::uniform_int_distribution<W> dist;
  RingT a(dist(g)), b(dist(g)), c(a * b);

  RingT a0(dist(g)), b0(dist(g)), c0(dist(g));
  RingT a1 = a - a0, b1 = b - b0, c1 = c - c0;

  BeaverTripleA<RingT> t0{{a0},{b0},{c0}};
  BeaverTripleA<RingT> t1{{a1},{b1},{c1}};
  return {t0, t1};
}

template<class URBG>
inline std::pair<BeaverTripleB, BeaverTripleB>
dealer_make_tripleB(URBG& g) {
  std::uniform_int_distribution<int> dist(0,1);
  uint8_t u = dist(g), v = dist(g), w = uint8_t(u & v);

  uint8_t u0 = dist(g), v0 = dist(g), w0 = dist(g);
  uint8_t u1 = u ^ u0, v1 = v ^ v0, w1 = w ^ w0;

  return {BeaverTripleB{{u0},{v0},{w0}}, BeaverTripleB{{u1},{v1},{w1}}};
}

} // namespace mpc
```

### `include/mpc/arithmetic_mpc.hpp` (1 multiplication)

```cpp
#pragma once
#include "mpc/beaver.hpp"

// You provide transport externally (TCP, shared memory, etc.).
// This interface is minimal and supports one round-trip for d,e openings.
namespace net {
struct Chan {
  virtual void send_u64(uint64_t) = 0;
  virtual uint64_t recv_u64() = 0;
  virtual ~Chan() = default;
};
}

namespace mpc {

template<typename RingT>
inline AddShare<RingT> mul_share(
  int party, net::Chan& ch,
  AddShare<RingT> x, AddShare<RingT> y,
  BeaverTripleA<RingT> t
){
  RingT d_local = x.s - t.a.s;
  RingT e_local = y.s - t.b.s;

  // Open d,e (additive)
  ch.send_u64(d_local.v); uint64_t d_other = ch.recv_u64();
  ch.send_u64(e_local.v); uint64_t e_other = ch.recv_u64();

  RingT d = RingT(d_local.v + d_other);
  RingT e = RingT(e_local.v + e_other);

  RingT z = t.c.s + RingT(d.v) * t.b.s + RingT(e.v) * t.a.s;
  if (party == 0) z += RingT(d.v) * RingT(e.v);
  return {z};
}

} // namespace mpc
```

### `include/mpc/boolean_mpc.hpp` (XOR + AND)

```cpp
#pragma once
#include "mpc/beaver.hpp"

namespace mpc {

inline XorShare bxor(XorShare a, XorShare b) { return {uint8_t(a.b ^ b.b)}; }
inline XorShare bnot(XorShare a) { return {uint8_t(a.b ^ 1)}; }

// AND via Beaver triple over GF(2) with open(d), open(e)
inline XorShare band_share(
  int party, net::Chan& ch,
  XorShare x, XorShare y,
  BeaverTripleB t
){
  uint8_t d_local = x.b ^ t.u.b;
  uint8_t e_local = y.b ^ t.v.b;

  ch.send_u64(d_local); uint64_t d_other = ch.recv_u64();
  ch.send_u64(e_local); uint64_t e_other = ch.recv_u64();

  uint8_t d = uint8_t(d_local ^ uint8_t(d_other));
  uint8_t e = uint8_t(e_local ^ uint8_t(e_other));

  uint8_t z = uint8_t(t.w.b ^ (d & t.v.b) ^ (e & t.u.b));
  if (party == 0) z ^= uint8_t(d & e);
  return {z};
}

} // namespace mpc
```

---

## 4) Masked wire invariant helpers (shares↔masked)

### `include/mpc/masked_wire.hpp`

```cpp
#pragma once
#include "mpc/shares.hpp"

namespace mpc {

template<typename RingT>
struct MaskedWire {
  RingT x_hat_public;          // public masked value
  AddShare<RingT> r_share;     // each party stores its share of r_in
};

// shares -> masked (Protocol 3.3): requires one broadcast each
template<typename RingT>
inline MaskedWire<RingT> shares_to_masked(
  net::Chan& ch,
  AddShare<RingT> x_share,
  AddShare<RingT> r_share
){
  RingT xh_local = x_share.s + r_share.s;
  ch.send_u64(xh_local.v);
  uint64_t xh_other = ch.recv_u64();
  RingT x_hat = RingT(xh_local.v + xh_other);
  return {x_hat, r_share};
}

// masked -> shares (Protocol 3.4): no communication
template<typename RingT>
inline AddShare<RingT> masked_to_shares(int party, MaskedWire<RingT> w){
  if (party == 0) return { w.x_hat_public - w.r_share.s };
  return { RingT(0) - w.r_share.s };
}

} // namespace mpc
```

---

## 5) SUF IR: intervals, polynomials, predicates, boolean formulas

### `include/suf/polynomial.hpp`

```cpp
#pragma once
#include <vector>
#include "core/ring.hpp"
#include "mpc/shares.hpp"
#include "mpc/arithmetic_mpc.hpp"

namespace suf {

// coeffs[k] corresponds to x^k
template<typename RingT>
struct Poly {
  std::vector<RingT> coeffs; // size d+1

  int degree() const { return (int)coeffs.size() - 1; }
};

// Horner evaluation on additive shares: consumes (deg) Beaver triples.
template<typename RingT>
inline mpc::AddShare<RingT> eval_poly_horner_shared(
  int party, net::Chan& ch,
  const Poly<RingT>& p,
  mpc::AddShare<RingT> x,
  const std::vector<mpc::BeaverTripleA<RingT>>& triples // >= degree
){
  // result = a_d
  mpc::AddShare<RingT> acc{ p.coeffs.back() };
  for (int i = (int)p.coeffs.size() - 2; i >= 0; --i) {
    acc = mpc::mul_share(party, ch, acc, x, triples[(size_t)i]);
    acc.s += p.coeffs[(size_t)i];
  }
  return acc;
}

} // namespace suf
```

### `include/suf/predicates.hpp`

```cpp
#pragma once
#include <cstdint>
#include <variant>

namespace suf {

// Primitive predicates in the UNMASKED x-domain (your SUF definition)
struct Pred_X_lt_const { uint64_t beta; };                    // 1[x < beta]
struct Pred_X_mod2f_lt { int f; uint64_t gamma; };            // 1[x mod 2^f < gamma]
struct Pred_MSB_x { };                                        // MSB(x)
struct Pred_MSB_x_plus { uint64_t c; };                        // MSB(x + c)

using PrimitivePred = std::variant<
  Pred_X_lt_const, Pred_X_mod2f_lt, Pred_MSB_x, Pred_MSB_x_plus
>;

} // namespace suf
```

### `include/suf/bool_expr.hpp`

```cpp
#pragma once
#include <memory>
#include <variant>
#include <vector>
#include "suf/predicates.hpp"

namespace suf {

// AST for boolean formulas built from primitive preds and connectives.
struct BoolExpr;

struct BConst { bool v; };
struct BVar  { int pred_idx; }; // index into "primitive predicate vector"
struct BNot  { std::unique_ptr<BoolExpr> a; };
struct BXor  { std::unique_ptr<BoolExpr> a, b; };
struct BAnd  { std::unique_ptr<BoolExpr> a, b; };
struct BOr   { std::unique_ptr<BoolExpr> a, b; };

struct BoolExpr {
  std::variant<BConst,BVar,BNot,BXor,BAnd,BOr> node;
};

} // namespace suf
```

### `include/suf/suf_ir.hpp`

```cpp
#pragma once
#include <vector>
#include "suf/polynomial.hpp"
#include "suf/bool_expr.hpp"

namespace suf {

// One interval piece: [alpha_i, alpha_{i+1})
template<typename RingT>
struct SufPiece {
  std::vector<Poly<RingT>> polys;      // r arithmetic outputs
  std::vector<BoolExpr> bool_outs;     // ℓ boolean outputs
};

template<typename RingT>
struct SUF {
  int n_bits;                    // n
  int r_out;                     // # arithmetic outputs
  int l_out;                     // # boolean outputs
  int degree;                    // max d
  std::vector<uint64_t> alpha;   // boundaries: size m+1, 0<=...<=2^n

  std::vector<PrimitivePred> primitive_preds; // global list before masking
  std::vector<SufPiece<RingT>> pieces;        // size m
};

} // namespace suf
```

---

## 6) PFSS abstraction (programmable FSS backend)

### `include/pfss/pfss.hpp`

```cpp
#pragma once
#include <cstdint>
#include <vector>
#include <string>

namespace pfss {

// Opaque key blob (serialize-friendly)
struct Key {
  std::vector<uint8_t> bytes;
};

// Public parameters (if needed by backend)
struct PublicParams {
  std::string backend_name;
  int lambda_bits = 128;
};

// Generic “program description” (filled by compiler & read by dealer only)
struct ProgramDesc {
  std::string kind; // "predicates", "coeff_lut", ...
  std::vector<uint8_t> dealer_only_desc; // backend-specific encoding
};

// Backend interface: ProgGen in dealer, Eval in online parties.
template<typename PayloadT>
struct Backend {
  virtual PublicParams setup(int lambda_bits) = 0;

  virtual std::pair<Key,Key> prog_gen(const PublicParams& pp,
                                      const ProgramDesc& desc) = 0;

  virtual PayloadT eval(int party,
                        const PublicParams& pp,
                        const Key& key,
                        uint64_t x_hat_public) const = 0;

  // Optional batched API (CPU multithread / GPU)
  virtual void eval_batch(int party,
                          const PublicParams& pp,
                          const Key& key,
                          const uint64_t* x_hat,
                          PayloadT* out,
                          size_t count) const {
    for (size_t i=0;i<count;i++) out[i] = eval(party, pp, key, x_hat[i]);
  }

  virtual ~Backend() = default;
};

} // namespace pfss
```

**Payload choices**

* Predicate program payload: **bit-vector shares**, e.g. `std::vector<uint64_t>` where each word packs 64 predicate bits in GF(2).
* Coefficient program payload: `std::vector<uint64_t>` (additive shares in Z₂⁶⁴) holding `(r*(d+1))` coefficients.

---

## 7) SUF→PFSS compiler (critical glue)

This is where your paper’s value lives: **mask-aware rewrite + two programs**.

### 7.1 Mask rewrite building blocks (rotated intervals)

### `include/compiler/mask_rewrite.hpp`

```cpp
#pragma once
#include <cstdint>
#include <vector>
#include <utility>

namespace compiler {

// Represents a subset of Z_{2^k} as 1 or 2 half-open intervals [L,U).
struct RotInterval {
  int k_bits;
  std::vector<std::pair<uint64_t,uint64_t>> ranges; // each is [L,U) in [0,2^k)

  // Evaluate in clear (for testing only!)
  bool contains(uint64_t x) const {
    uint64_t mask = (k_bits==64)?~0ull:((1ull<<k_bits)-1);
    x &= mask;
    for (auto [L,U] : ranges) {
      if (L <= U) { if (L <= x && x < U) return true; }
      else { // wrap (should not occur if we normalize to 2 ranges)
        if (x >= L || x < U) return true;
      }
    }
    return false;
  }
};

// Image of [0,beta) under +r mod 2^k
inline RotInterval rotate_prefix(int k_bits, uint64_t r, uint64_t beta){
  uint64_t mod = (k_bits==64) ? 0 : (1ull<<k_bits);
  auto norm = [&](uint64_t x){ return (k_bits==64)?x:(x % mod); };

  RotInterval out{ k_bits, {} };
  r = norm(r); beta = norm(beta);

  uint64_t end = norm(r + beta);
  if (beta == 0) return out;

  if (k_bits==64) {
    // In mod 2^64 arithmetic, overflow acts like mod. Decide wrap by comparing.
    uint64_t r_plus_beta = r + beta;
    bool wrap = (r_plus_beta < r);
    if (!wrap) out.ranges.push_back({r, r_plus_beta});
    else { out.ranges.push_back({0, r_plus_beta}); out.ranges.push_back({r, ~0ull}); }
    return out;
  }

  if (r + beta < mod) out.ranges.push_back({r, r+beta});
  else {
    out.ranges.push_back({0, (r+beta)-mod});
    out.ranges.push_back({r, mod});
  }
  return out;
}

// Rotate low-bit comparison: u = x mod 2^f < gamma, with s = (u + delta) mod 2^f
inline RotInterval rotate_lowbits(int f_bits, uint64_t delta, uint64_t gamma){
  // same logic but in k=f_bits domain
  return rotate_prefix(f_bits, delta, gamma);
}

} // namespace compiler
```

### 7.2 Compiler output types

### `include/compiler/suf_to_pfss.hpp`

```cpp
#pragma once
#include "pfss/pfss.hpp"
#include "suf/suf_ir.hpp"
#include <vector>

namespace compiler {

// What the dealer gives each party for one SUF gate instance/type.
struct CompiledSUFKeys {
  pfss::Key pred_key0, pred_key1;
  pfss::Key coeff_key0, coeff_key1;

  // masks (shared). r_in is on input wire; r_out per arithmetic output.
  std::vector<uint64_t> r_out_share0, r_out_share1;
  uint64_t r_in_share0, r_in_share1;

  // metadata
  int n_bits;
  int r_out;
  int l_out;
  int degree;
  int num_primitive_preds;
  int packed_pred_words; // e.g. ceil(T/64)
};

} // namespace compiler
```

### 7.3 Program descriptions (dealer-only serialized)

You’ll keep the PFSS descriptions backend-specific but compiler-owned:

* **Predicate program desc**: list of rotated-interval tests you want as primitive bits.
* **Coeff program desc**: disjoint masked-domain intervals with vector payload (coefficients).

### `include/pfss/program_desc.hpp` (simple encoding)

```cpp
#pragma once
#include <cstdint>
#include <vector>
#include <utility>

namespace pfss_desc {

// A “predicate bit” is membership in a rotated interval, in either full n bits or low f bits.
struct PredBitDesc {
  int k_bits; // n or f
  // up to two [L,U) ranges
  std::vector<std::pair<uint64_t,uint64_t>> ranges;
};

// Piecewise constant vector payload: disjoint intervals in Z_{2^n}
struct PiecewiseVectorDesc {
  int n_bits;
  struct Piece { uint64_t L, U; std::vector<uint64_t> payload; };
  std::vector<Piece> pieces; // must partition domain (or cover all with default)
};

} // namespace pfss_desc
```

### 7.4 The compilation procedure (dealer-side)

### `include/compiler/suf_collect.hpp` (collect + mask)

```cpp
#pragma once
#include "suf/suf_ir.hpp"
#include "compiler/mask_rewrite.hpp"
#include "pfss/program_desc.hpp"
#include <vector>

namespace compiler {

// Collect primitive predicate bits needed, and rewrite each primitive into one RotInterval over hat{x}.
inline std::vector<pfss_desc::PredBitDesc>
compile_primitive_pred_bits(
  const suf::SUF<core::Z2n<64>>& F,
  uint64_t r_in  // dealer knows
){
  std::vector<pfss_desc::PredBitDesc> out;
  out.reserve(F.primitive_preds.size());

  for (const auto& pr : F.primitive_preds) {
    using namespace suf;
    if (std::holds_alternative<Pred_X_lt_const>(pr)) {
      auto beta = std::get<Pred_X_lt_const>(pr).beta;
      auto R = rotate_prefix(F.n_bits, r_in, beta); // hat{x} in rotated interval
      out.push_back({F.n_bits, R.ranges});
    } else if (std::holds_alternative<Pred_X_mod2f_lt>(pr)) {
      auto p = std::get<Pred_X_mod2f_lt>(pr);
      uint64_t delta = (p.f == 64) ? r_in : (r_in & ((1ull<<p.f)-1));
      auto R = rotate_lowbits(p.f, delta, p.gamma);
      out.push_back({p.f, R.ranges});
    } else if (std::holds_alternative<Pred_MSB_x>(pr)) {
      // MSB(x) = 1 - 1[x < 2^{n-1}] ; treat as rotated prefix beta=2^{n-1}
      uint64_t beta = (F.n_bits==64) ? (1ull<<63) : (1ull<<(F.n_bits-1));
      auto R = rotate_prefix(F.n_bits, r_in, beta); // this gives 1[x<beta]
      out.push_back({F.n_bits, R.ranges});
    } else if (std::holds_alternative<Pred_MSB_x_plus>(pr)) {
      // MSB(x+c) = 1 - 1[(x+c) < 2^{n-1}] . Under hat{x}: (hat{x} - r_in + c).
      // Equivalent to rotate by r_in - c (mod 2^n) or bake as a separate desc.
      auto c = std::get<Pred_MSB_x_plus>(pr).c;
      uint64_t beta = (F.n_bits==64) ? (1ull<<63) : (1ull<<(F.n_bits-1));
      uint64_t r_eff = r_in - c; // because test is on (x+c), so masked variable shifts by (r_in - c)
      auto R = rotate_prefix(F.n_bits, r_eff, beta);
      out.push_back({F.n_bits, R.ranges});
    }
  }
  return out;
}

} // namespace compiler
```

### `include/compiler/suf_to_pfss.hpp` (the dealer’s main function)

```cpp
#pragma once
#include "compiler/suf_collect.hpp"
#include "compiler/suf_to_pfss.hpp"
#include "pfss/pfss.hpp"
#include "pfss/program_desc.hpp"
#include <random>
#include <cstring>

namespace compiler {

// Helper: split a ring element into two additive shares
inline std::pair<uint64_t,uint64_t> split_u64(std::mt19937_64& g, uint64_t x){
  uint64_t s0 = g();
  uint64_t s1 = x - s0;
  return {s0,s1};
}

// Build coefficient LUT pieces in masked hat{x}-domain (split wrap-around intervals).
inline pfss_desc::PiecewiseVectorDesc
compile_coeff_piecewise(
  const suf::SUF<core::Z2n<64>>& F,
  uint64_t r_in
){
  // Payload per interval: concat coefficients for each arithmetic output.
  // We assume F.pieces[i].polys[j].coeffs size = d+1 for all.
  pfss_desc::PiecewiseVectorDesc desc;
  desc.n_bits = F.n_bits;

  auto shift = [&](uint64_t a)->uint64_t { return a + r_in; }; // mod 2^64

  for (size_t i=0;i<F.pieces.size();i++){
    uint64_t A = F.alpha[i];
    uint64_t B = F.alpha[i+1];
    uint64_t L = shift(A);
    uint64_t U = shift(B);

    std::vector<uint64_t> payload;
    payload.reserve((size_t)F.r_out * (size_t)(F.degree+1));
    for (int j=0;j<F.r_out;j++){
      const auto& poly = F.pieces[i].polys[(size_t)j];
      for (auto c : poly.coeffs) payload.push_back(c.v);
    }

    // Split wrap-around in hat{x} domain by checking overflow L->U.
    bool wrap = (U < L); // for 64-bit; for smaller bits you’d compare with modulus
    if (!wrap) {
      desc.pieces.push_back({L, U, payload});
    } else {
      desc.pieces.push_back({0, U, payload});
      desc.pieces.push_back({L, ~0ull, payload});
    }
  }
  return desc;
}

template<typename PredPayloadT, typename CoeffPayloadT>
inline CompiledSUFKeys
dealer_compile_suf_gate(
  pfss::Backend<PredPayloadT>& pred_backend,
  pfss::Backend<CoeffPayloadT>& coeff_backend,
  const pfss::PublicParams& pp_pred,
  const pfss::PublicParams& pp_coeff,
  const suf::SUF<core::Z2n<64>>& F,
  std::mt19937_64& g
){
  // 1) sample masks
  auto [r0,r1] = split_u64(g, g()); // r_in random; represented as shares
  uint64_t r_in = r0 + r1;

  std::vector<uint64_t> r_out0(F.r_out), r_out1(F.r_out);
  for (int j=0;j<F.r_out;j++){
    auto [s0,s1] = split_u64(g, g());
    r_out0[(size_t)j] = s0;
    r_out1[(size_t)j] = s1;
  }

  // 2) compile predicate desc in masked coordinate (dealer-only)
  auto pred_bits = compile_primitive_pred_bits(F, r_in);

  pfss::ProgramDesc pred_prog;
  pred_prog.kind = "predicates";
  // backend-specific serialization:
  // You can serialize pred_bits with your own stable encoding.
  pred_prog.dealer_only_desc = {/*...*/};

  // 3) compile coefficient piecewise LUT desc
  auto coeff_desc = compile_coeff_piecewise(F, r_in);
  pfss::ProgramDesc coeff_prog;
  coeff_prog.kind = "coeff_lut";
  coeff_prog.dealer_only_desc = {/*...*/};

  // 4) ProgGen -> keys
  auto [pk0,pk1] = pred_backend.prog_gen(pp_pred, pred_prog);
  auto [ck0,ck1] = coeff_backend.prog_gen(pp_coeff, coeff_prog);

  CompiledSUFKeys out;
  out.pred_key0 = std::move(pk0); out.pred_key1 = std::move(pk1);
  out.coeff_key0 = std::move(ck0); out.coeff_key1 = std::move(ck1);
  out.r_in_share0 = r0; out.r_in_share1 = r1;
  out.r_out_share0 = std::move(r_out0);
  out.r_out_share1 = std::move(r_out1);
  out.n_bits = F.n_bits; out.r_out = F.r_out; out.l_out = F.l_out;
  out.degree = F.degree;
  out.num_primitive_preds = (int)F.primitive_preds.size();
  out.packed_pred_words = (out.num_primitive_preds + 63)/64;
  return out;
}

} // namespace compiler
```

**Why this is “paper-faithful”:**

* Dealer knows `r_in` ⇒ can compute rotated-interval descriptions without revealing `r_in`.
* Online phase only sees `(key_b, x_hat_public)`.

---

## 8) Composite-FSS gate API

### `include/gates/gate_api.hpp`

```cpp
#pragma once
#include "compiler/suf_to_pfss.hpp"
#include "mpc/masked_wire.hpp"
#include "suf/polynomial.hpp"

namespace gates {

template<typename RingT>
struct GateEvalResult {
  mpc::AddShare<RingT> y_hat_share;      // masked arithmetic output share
  // Optional boolean/helper outputs (XOR shares) etc.
};

// Generic “compiled SUF gate evaluation”: 2 PFSS evals + Horner + add r_out.
template<typename RingT, typename PredPayloadT, typename CoeffPayloadT>
inline GateEvalResult<RingT> eval_compiled_suf_gate(
  int party,
  net::Chan& ch,
  const pfss::Backend<PredPayloadT>& pred_backend,
  const pfss::Backend<CoeffPayloadT>& coeff_backend,
  const pfss::PublicParams& pp_pred,
  const pfss::PublicParams& pp_coeff,
  const compiler::CompiledSUFKeys& k,
  uint64_t x_hat_public,
  mpc::AddShare<RingT> x_share, // if not available, derive from masked->shares using r_in_share
  const std::vector<mpc::BeaverTripleA<RingT>>& triples
){
  // 1) Evaluate PFSS programs
  const pfss::Key& pred_key = (party==0)?k.pred_key0:k.pred_key1;
  const pfss::Key& coeff_key = (party==0)?k.coeff_key0:k.coeff_key1;

  PredPayloadT pred_payload = pred_backend.eval(party, pp_pred, pred_key, x_hat_public);
  CoeffPayloadT coeff_payload = coeff_backend.eval(party, pp_coeff, coeff_key, x_hat_public);

  // 2) Parse coeff_payload -> polynomial coeffs for each arithmetic output
  // For simplicity assume 1 arithmetic output (extend easily).
  // coeff_payload is a vector<uint64_t> length r*(d+1)
  suf::Poly<RingT> poly;
  poly.coeffs.resize((size_t)k.degree + 1);
  for (int i=0;i<=k.degree;i++){
    poly.coeffs[(size_t)i] = RingT(coeff_payload[(size_t)i]);
  }

  // 3) Polynomial evaluation on shares
  auto y_share = suf::eval_poly_horner_shared(party, ch, poly, x_share, triples);

  // 4) Add output mask share r_out
  uint64_t r_out_b = (party==0)? k.r_out_share0[0] : k.r_out_share1[0];
  y_share.s += RingT(r_out_b);

  return { y_share };
}

} // namespace gates
```

---

## 9) Example gates you can ship immediately

### 9.1 ReLU as SUF (degree-1, 2 intervals, helper bit)

### `include/gates/relu_gate.hpp`

```cpp
#pragma once
#include "suf/suf_ir.hpp"
#include "core/ring.hpp"

namespace gates {

// ReLU(x) = max(x,0) in two’s complement.
// We expose arithmetic output y and helper bit w = 1[x>=0] (optional).
inline suf::SUF<core::Z2n<64>> make_relu_suf_64(){
  using R = core::Z2n<64>;
  suf::SUF<R> F;
  F.n_bits = 64;
  F.r_out = 1;
  F.l_out = 1;
  F.degree = 1;

  // boundaries split by sign: [0,2^63) and [2^63,2^64)
  F.alpha = {0ull, (1ull<<63), 0ull}; // NOTE: last boundary "2^64" can't fit in u64.
  // Practical approach: store boundaries as u128 in real code, or represent sign split specially.
  // For now, treat as two pieces keyed by MSB predicate rather than alpha covering 2^64.

  // Primitive preds: MSB(x)
  F.primitive_preds.push_back(suf::Pred_MSB_x{});

  // Two pieces:
  // if MSB(x)=0: y=x; if MSB(x)=1: y=0
  suf::SufPiece<R> nonneg;
  nonneg.polys = { suf::Poly<R>{{ R(0), R(1) }} }; // 0 + 1*x
  nonneg.bool_outs = { suf::BoolExpr{ suf::BConst{true} } };

  suf::SufPiece<R> neg;
  neg.polys = { suf::Poly<R>{{ R(0) }} };
  neg.bool_outs = { suf::BoolExpr{ suf::BConst{false} } };

  F.pieces = { nonneg, neg };
  return F;
}

} // namespace gates
```

**Implementation note:** for 64-bit, the “boundary at 2^64” needs `u128` or a special-cased “last interval ends at modulus”. In production, represent intervals as:

* either `std::optional<uint64_t> end` where `nullopt` means `2^n`,
* or store boundaries as `unsigned __int128`.

---

## 10) CPU vs GPU: what changes?

### CPU version

* PFSS eval is run on CPU, optionally batched across wires:

    * `pred_backend.eval_batch(...)`
    * `coeff_backend.eval_batch(...)`
* Polynomial eval is cheap; biggest cost is PFSS (AES expansions).
* Use:

    * thread pool (OpenMP/TBB)
    * aligned key storage
    * AES-NI PRG (backend-specific)

### GPU version

You keep the **exact same compiler + gate algebra**, but provide:

* `pfss::Backend<T>::eval_batch(...)` override that routes to CUDA kernels
* optional GPU polynomial evaluation kernel for large batches

#### `cuda/pfss_cuda_api.hpp` (clean ABI for backends)

```cpp
#pragma once
#include <cstdint>
#include <cstddef>

namespace cuda_pfss {

// backend-defined opaque device key handle
struct DeviceKey { void* ptr; };

// Upload key bytes -> device resident key
DeviceKey upload_key(const uint8_t* key_bytes, size_t key_len);
void free_key(DeviceKey);

// Evaluate many x_hat in batch
// out is backend-defined payload layout (e.g., uint64_t words).
void eval_batch_pred(int party, DeviceKey key,
                     const uint64_t* d_xhat, uint64_t* d_out,
                     size_t count);

void eval_batch_coeff(int party, DeviceKey key,
                      const uint64_t* d_xhat, uint64_t* d_out,
                      size_t count);

} // namespace cuda_pfss
```

#### `cuda/poly_kernels.cu` (Horner in GPU, additive shares)

```cpp
// Sketch only: each thread evaluates one polynomial output for one element.
// In practice, fuse across outputs, prefetch coefficients, and keep x_share in registers.
extern "C" __global__
void horner_eval_u64(const uint64_t* x_share,
                     const uint64_t* coeffs, // [count*(d+1)]
                     uint64_t* out_share,
                     int d, size_t count){
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const uint64_t* c = coeffs + i*(size_t)(d+1);
  uint64_t acc = c[d];
  uint64_t x = x_share[i];
  for (int k=d-1;k>=0;k--) acc = acc * x + c[k];
  out_share[i] = acc;
}
```

**Where GPU helps most:** batched PFSS evaluation (tree PRG expansions). Your backend will implement that kernel; the SUF framework doesn’t care.

---

## 11) Cleartext backend (to validate everything now)

This backend is **NOT secure**, but ensures your compiler + gate logic is correct quickly.

### `include/pfss/backend_cleartext.hpp`

```cpp
#pragma once
#include "pfss/pfss.hpp"
#include "pfss/program_desc.hpp"
#include <random>
#include <unordered_map>

namespace pfss {

// Payload types for cleartext demo
using PredPayload = std::vector<uint64_t>;   // packed bits
using CoeffPayload = std::vector<uint64_t>;  // coeff words

struct CleartextBackendPred final : Backend<PredPayload> {
  struct Stored { std::vector<pfss_desc::PredBitDesc> bits; };
  std::unordered_map<uint64_t, Stored> programs; // key_id -> desc
  uint64_t next_id = 1;

  PublicParams setup(int lambda_bits) override { return {"CLEAR_PRED", lambda_bits}; }

  std::pair<Key,Key> prog_gen(const PublicParams&, const ProgramDesc&) override {
    // In a real implementation, decode dealer_only_desc -> PredBitDesc list.
    // Here: placeholder storing nothing; you will wire this for tests.
    uint64_t id = next_id++;
    Key k0, k1;
    k0.bytes.resize(sizeof(id)); std::memcpy(k0.bytes.data(), &id, sizeof(id));
    k1.bytes = k0.bytes;
    return {k0,k1};
  }

  PredPayload eval(int party, const PublicParams&, const Key& key, uint64_t x_hat) const override {
    (void)party;
    uint64_t id; std::memcpy(&id, key.bytes.data(), sizeof(id));
    // For demo: return empty
    return {};
  }
};

} // namespace pfss
```

In your local tests you’ll expand this to actually store decoded `PredBitDesc` and `PiecewiseVectorDesc`, then:

* compute the real output in clear,
* sample random share for party0,
* return share0 (party0) and share1=out^share0 (party1) for GF(2) bits, or out-share0 for ring payloads.

That gives you a fully working SUF pipeline before plugging secure FSS.

---

## 12) External secure backend adapter (where you plug real DPF/DCF)

### `include/pfss/backend_external_adapter.hpp`

```cpp
#pragma once
#include "pfss/pfss.hpp"

// This is the only file that should mention a concrete library (Grotto/SIGMA-style).
// The rest of the framework is backend-agnostic.

namespace pfss {

template<typename PayloadT>
struct ExternalBackendAdapter final : Backend<PayloadT> {
  PublicParams setup(int lambda_bits) override {
    // call library setup
    return {"EXTERNAL_BACKEND", lambda_bits};
  }

  std::pair<Key,Key> prog_gen(const PublicParams& pp, const ProgramDesc& desc) override {
    // 1) decode desc.dealer_only_desc into your backend’s native description
    // 2) call backend ProgGen
    // 3) return key blobs
    return {};
  }

  PayloadT eval(int party, const PublicParams& pp, const Key& key, uint64_t x_hat) const override {
    // call backend Eval in constant-time
    return {};
  }

  void eval_batch(int party, const PublicParams& pp, const Key& key,
                  const uint64_t* x_hat, PayloadT* out, size_t count) const override {
    // optional: batch API (multi-thread or GPU)
    for (size_t i=0;i<count;i++) out[i] = eval(party, pp, key, x_hat[i]);
  }
};

} // namespace pfss
```

---

## 13) What you get “for free” with this structure

* Adding a new nonlinearity is **just**:

    1. Build its SUF (`alpha[]`, `polys`, `bool formulas`, primitive pred list).
    2. Compile with `dealer_compile_suf_gate`.
    3. Eval with `eval_compiled_suf_gate`, plus small custom post-processing if needed (ReluARS corrections, SoftmaxBlock reductions, etc).

* Your paper’s **abstraction layer** is literally a stable API:

    * `suf::SUF` as IR
    * `compiler::dealer_compile_suf_gate(...)`
    * `gates::eval_compiled_suf_gate(...)`

---

## 14) Next steps you can ask for (I’ll respond with code, not questions)

In your next message, tell me which direction to expand first, and I’ll output the full code for it:

1. **ReluARS gate**: helper bits `(w,t,d)` + correction LUT + ARS/truncation circuit.
2. **GeLU spline SUF**: interval encoding, coefficient payload packing, and the exact gate post-processing (`x^+ + δ(x)`).
3. **Batching + packing**: packed predicate words, SIMD-friendly CPU layout, and CUDA batch interfaces.
4. **A concrete secure backend**: implement a minimal DPF/DCF-style backend (reference, not optimized) or wire an adapter to an existing DPF/DCF engine you already use.

Just say e.g. “Expand ReluARS (CPU), then GPU batching.”
