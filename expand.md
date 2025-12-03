### Backend choice (DPF/DCF)

For a drop-in **open-source** backend that already has **2-party DPF + DCF** and also contains **CUDA code**, I’d base the PFSS layer on **`myl7/fss`** (C/C++ + CUDA). Its README states it implements 2-party DPF and DCF, fixes inputs to bits and outputs to bytes (with customizable “group addition”), and includes CUDA in the codebase.

Two good “alternatives but incomplete for SUF-on-masked-x by themselves” are:

* Google’s `distributed_point_functions` (DPF only; no DCF)
* Facebook Research `GPU-DPF` (GPU-focused DPF; again not a full DCF stack) ([GitHub][1])

So: **use `myl7/fss` as the concrete PFSS backend**, and keep your SUF/Composite-FSS code behind a clean interface so you can swap in a SIGMA-style packed CDPF/vector-DPF later. (SIGMA’s paper explicitly discusses DCF and also reductions from DPF → comparison in some settings.)

Below is a concrete, code-oriented design for:

1. **ReluARS** (helper bits `w,t,d` + correction LUT + truncation/ARS circuit)
2. **GeLU spline SUF** (interval encoding + coefficient payload packing + post-processing `x^+ + δ(x)`)

---

## 0) Minimal MPC + PFSS interfaces (C++17)

This is the “spine” both gates plug into.

```cpp
// include/mpc/ring64.h
#pragma once
#include <cstdint>

namespace mpc {

// Z_{2^64} “ring”: uint64 overflow is exactly mod 2^64.
using u64 = uint64_t;

struct Share64 { u64 v; };      // additive share: x = x0 + x1 mod 2^64
struct BitShare64 { u64 v; };   // arithmetic bit-share: bit in {0,1} but shared in Z_{2^64}

inline Share64 add(Share64 a, Share64 b) { return {a.v + b.v}; }
inline Share64 sub(Share64 a, Share64 b) { return {a.v - b.v}; }
inline Share64 add_const(Share64 a, u64 c) { return {a.v + c}; }
inline Share64 sub_const(Share64 a, u64 c) { return {a.v - c}; }
inline Share64 mul_const(Share64 a, u64 c) { return {a.v * c}; } // linear

inline BitShare64 bit_add(BitShare64 a, BitShare64 b) { return {a.v + b.v}; } // still in ring
inline BitShare64 bit_sub(BitShare64 a, BitShare64 b) { return {a.v - b.v}; }

} // namespace mpc
```

```cpp
// include/mpc/beaver.h
#pragma once
#include "mpc/ring64.h"

namespace mpc {

// Each party holds shares of (a,b,c=a*b).
struct TripleShare64 { Share64 a, b, c; };

// Abstract comm: implement over TCP/GRPC/etc. For demo, keep it simple.
struct IChannel {
  virtual ~IChannel() = default;
  virtual void send_u64(u64 x) = 0;
  virtual u64  recv_u64() = 0;
};

// Beaver multiply: z = x*y (additively shared)
inline Share64 mul(Share64 x, Share64 y, const TripleShare64& t, int party_id, IChannel& ch) {
  // locally compute d_i = x_i - a_i, e_i = y_i - b_i
  u64 d_i = x.v - t.a.v;
  u64 e_i = y.v - t.b.v;

  // open d = d0+d1, e = e0+e1 (1 round)
  ch.send_u64(d_i);
  ch.send_u64(e_i);
  u64 d_j = ch.recv_u64();
  u64 e_j = ch.recv_u64();
  u64 d = d_i + d_j;
  u64 e = e_i + e_j;

  // z_i = c_i + d*b_i + e*a_i + (party0 ? d*e : 0)
  u64 z = t.c.v + d * t.b.v + e * t.a.v;
  if (party_id == 0) z += d * e;
  return {z};
}

// Multiplying by a shared bit is just mul() with y being a BitShare64 as Share64.
inline Share64 mul_by_bit(Share64 x, BitShare64 b, const TripleShare64& t, int party_id, IChannel& ch) {
  return mul(x, Share64{b.v}, t, party_id, ch);
}

} // namespace mpc
```

```cpp
// include/pfss/backend.h
#pragma once
#include <cstdint>
#include <vector>

namespace pfss {

using u64 = uint64_t;

// A “key” for one PFSS program for one party.
// Concrete backends will define the actual blob format; we treat it as bytes.
struct KeyBlob { std::vector<uint8_t> bytes; };

// Program types we need for these gates:
struct DcfLtDesc {
  int nbits;     // input bit-width (e.g., 64 or f)
  u64 alpha;     // threshold: outputs 1 if x < alpha else 0
};

struct IntervalLutDesc {
  int nbits;                       // usually 64
  std::vector<u64> cutpoints;      // increasing, defines intervals [c[i], c[i+1])
  int out_words;                   // payload word count
  std::vector<u64> payload_words;  // flattened: interval_count * out_words
};

// Backend interface.
struct IBackend {
  virtual ~IBackend() = default;

  virtual void setup(int lambda) = 0;

  // 2-party keygen
  virtual void gen_dcf_lt(const DcfLtDesc& desc, KeyBlob& k0, KeyBlob& k1) = 0;
  virtual void gen_interval_lut(const IntervalLutDesc& desc, KeyBlob& k0, KeyBlob& k1) = 0;

  // eval: returns additive share words in Z_{2^64}
  virtual u64 eval_dcf_lt(int party_id, const KeyBlob& kb, u64 x) = 0;
  virtual void eval_interval_lut(int party_id, const KeyBlob& kb, u64 x, u64* out_words) = 0;

  // Optional: batch/GPU variants (can be no-ops in CPU-only backend).
  virtual void eval_dcf_lt_batch_gpu(int party_id, const KeyBlob& kb, const u64* d_x, u64* d_out, size_t n) {}
  virtual void eval_interval_lut_batch_gpu(int party_id, const KeyBlob& kb, const u64* d_x, u64* d_out_words, size_t n) {}
};

} // namespace pfss
```

> **Where `myl7/fss` fits:** implement `pfss::IBackend` using its DCF/DPF APIs (and its CUDA PRG path) . The gates below won’t change.

---

## 1) ReluARS gate (w,t,d + LUT + truncation)

### 1.1 What we compute online

We implement **rounded truncation after ReLU** using the classic “masked open” identity, but all secret-dependent comparisons are done via DCF on the **public masked value**.

Let `n=64`, fractional bits `f>=1`, offset `off = 2^{f-1}` for rounding, and public:

* `x_hat = x + r (mod 2^64)` (mask `r` is secret-shared)
* define `z = x + off` (for rounding), so `z_hat = z + r = x_hat + off` (public)

Then for **unsigned truncation** we use:
[
\left\lfloor \frac{z}{2^f}\right\rfloor
= (z_hat \gg f) - (r \gg f) - b + u\cdot 2^{64-f}
]
where

* `u = 1[z_hat < r]`  (wrap bit)
* `b = 1[(z_hat mod 2^f) < (r mod 2^f)]` (borrow from low bits)

Finally apply ReLU:

* `w = 1[x >= 0]` (non-negative)
* output `y = w * floor(z/2^f)`.

We expose helper bits `(w,t,d)` where we take:

* `d := u`
* `t := b`
* `w` computed as “membership in a rotated half-domain interval” (needs two DCFs + 2 tiny AND multiplications; see below).

### 1.2 Key material (offline)

We need:

* Secret-shared mask `r` and also **secret-shared** `r_hi = r >> f`, and **secret-shared** `rho = MSB(r)` (never reconstruct `rho`).
* DCF keys for:

    * `LtR64`: `1[x < r]` on 64-bit inputs (used for both `x_hat` and `z_hat`)
    * `LtRPlus64`: `1[x < (r + 2^63) mod 2^64]` (needed for sign under masking)
    * `LtRlow`: `1[x_low < r_low]` on `f`-bit domain (borrow bit `t`)

### 1.3 “Correction LUT”

You asked for the LUT explicitly. We’ll define:

* `corr(t,d) = (-t) + d*2^(64-f)` in Z2^64
  and embed an 8-entry table indexed by `(w<<2)|(t<<1)|d`.
  (Entries for `w=0` can be `0` since we still multiply by `w` to ReLU-clip.)

### 1.4 Code: ReluARS gate

```cpp
// include/gates/reluars.h
#pragma once
#include <array>
#include "mpc/ring64.h"
#include "mpc/beaver.h"
#include "pfss/backend.h"

namespace gates {

struct ReluARSParams {
  int f; // fractional bits, 1..63
};

// Party-b key material for ONE wire instance of the gate.
struct ReluARSGateKey {
  ReluARSParams p;

  // Shares of mask material (all are additive shares in Z2^64):
  mpc::Share64 r_share;      // share of r
  mpc::Share64 r_hi_share;   // share of (r >> f) provided by dealer (do NOT compute by shifting share!)
  mpc::BitShare64 rho_share; // share of MSB(r) as a bit in {0,1}

  // Output mask for producing masked output hat{y} = y + r_out:
  mpc::Share64 r_out_share;

  // PFSS keys (one blob per party)
  pfss::KeyBlob k_lt_r_64;
  pfss::KeyBlob k_lt_rplus_64; // threshold = (r + 2^63) mod 2^64
  pfss::KeyBlob k_lt_rlow_f;   // threshold = (r mod 2^f)

  // Beaver triples: for bit-logic and final multiply by w.
  // - c1*c2 to compute XOR
  // - xor*rho to conditional-not
  // - w*base to apply ReLU
  mpc::TripleShare64 tri_c1c2;
  mpc::TripleShare64 tri_xorrho;
  mpc::TripleShare64 tri_wbase;
};

// Helper: boolean XOR on arithmetic bit shares (needs one multiplication).
inline mpc::BitShare64 xor_bitshares(mpc::BitShare64 a, mpc::BitShare64 b,
                                    const mpc::TripleShare64& tri,
                                    int pid, mpc::IChannel& ch) {
  // a XOR b = a + b - 2ab
  auto ab = mpc::mul(mpc::Share64{a.v}, mpc::Share64{b.v}, tri, pid, ch);
  return mpc::BitShare64{a.v + b.v - 2 * ab.v};
}

// Helper: NOT on arithmetic bit share: (1 - x) in ring.
inline mpc::BitShare64 not_bitshare(mpc::BitShare64 x, int pid) {
  // represent constant 1 as (party0:1, party1:0)
  return mpc::BitShare64{ (pid==0 ? 1ULL : 0ULL) - x.v };
}

// 8-entry LUT for corr(w,t,d). We keep it explicit for clarity/debug.
inline std::array<mpc::u64,8> make_corr_lut(int f) {
  std::array<mpc::u64,8> lut{};
  const mpc::u64 BIG = (f==0) ? 0 : (1ULL << (64 - f)); // 2^(64-f), well-defined for f in 1..63
  for (int w=0; w<=1; ++w) {
    for (int t=0; t<=1; ++t) {
      for (int d=0; d<=1; ++d) {
        int idx = (w<<2) | (t<<1) | d;
        if (w==0) { lut[idx] = 0; continue; }
        // corr = (-t) + d*2^(64-f)
        mpc::u64 corr = (t ? (mpc::u64)(0ULL - 1ULL) : 0ULL);
        corr += (d ? BIG : 0ULL);
        lut[idx] = corr;
      }
    }
  }
  return lut;
}

// Online eval:
// inputs: public x_hat, backend, comm channel
// output: (masked y share, helper bit shares w,t,d)
inline void eval_reluars(const ReluARSGateKey& K,
                         int party_id,
                         pfss::IBackend& be,
                         mpc::IChannel& ch,
                         mpc::u64 x_hat_public,
                         mpc::Share64& out_hat_y_share,
                         mpc::BitShare64& out_w,
                         mpc::BitShare64& out_t,
                         mpc::BitShare64& out_d) {
  const int f = K.p.f;
  const mpc::u64 off = 1ULL << (f-1);
  const mpc::u64 z_hat = x_hat_public + off; // public: z_hat = x_hat + off

  // --- DCF evals (shares in Z2^64, but values are 0/1) ---
  // c1 = 1[x_hat < r]
  mpc::BitShare64 c1{ be.eval_dcf_lt(party_id, K.k_lt_r_64, x_hat_public) };
  // c2 = 1[x_hat < (r + 2^63) mod 2^64]
  mpc::BitShare64 c2{ be.eval_dcf_lt(party_id, K.k_lt_rplus_64, x_hat_public) };

  // xor = c1 XOR c2
  mpc::BitShare64 xorb = xor_bitshares(c1, c2, K.tri_c1c2, party_id, ch);

  // w = (rho ? NOT(xor) : xor)  == xor XOR rho   (since rho=1 flips) as boolean,
  // but we compute it arithmetically: w = xor + rho - 2*xor*rho
  auto xor_rho = mpc::mul(mpc::Share64{xorb.v}, mpc::Share64{K.rho_share.v},
                          K.tri_xorrho, party_id, ch);
  out_w = mpc::BitShare64{ xorb.v + K.rho_share.v - 2 * xor_rho.v };

  // d = u = 1[z_hat < r]  (same key as lt_r_64, but evaluated at z_hat)
  out_d = mpc::BitShare64{ be.eval_dcf_lt(party_id, K.k_lt_r_64, z_hat) };

  // t = b = 1[(z_hat mod 2^f) < (r mod 2^f)]  : DCF on f-bit domain
  const mpc::u64 z_low = (f==64) ? z_hat : (z_hat & ((1ULL<<f)-1));
  out_t = mpc::BitShare64{ be.eval_dcf_lt(party_id, K.k_lt_rlow_f, z_low) };

  // --- truncation arithmetic ---
  // base = (z_hat >> f) - (r >> f) + corr(t,d)
  const mpc::u64 z_shift = (f==0) ? z_hat : (z_hat >> f);
  mpc::Share64 z_shift_share{ party_id==0 ? z_shift : 0ULL };

  // corr(t,d) = (-t) + d*2^(64-f) ; we keep LUT explicit.
  const auto lut = make_corr_lut(f);

  // We need an index (w,t,d) to *debug* LUT; online we just compute corr linearly.
  // corr_share = (-t_share) + d_share*2^(64-f)
  const mpc::u64 BIG = (1ULL << (64 - f));
  mpc::Share64 corr_share = mpc::add( mpc::Share64{0ULL - out_t.v},
                                      mpc::mul_const(mpc::Share64{out_d.v}, BIG) );

  mpc::Share64 base_share = mpc::add( mpc::sub(z_shift_share, K.r_hi_share), corr_share );

  // Apply ReLU via multiplication by w (1 Beaver multiply)
  mpc::Share64 y_share = mpc::mul_by_bit(base_share, out_w, K.tri_wbase, party_id, ch);

  // Mask output: hat{y}_share = y_share + r_out_share
  out_hat_y_share = mpc::add(y_share, K.r_out_share);

  // Optional sanity: you can reconstruct LUT index locally if you ever open bits (don’t in prod).
  (void)lut;
}

} // namespace gates
```

**Notes that match your writeup:**

* `w,t,d` are produced via PFSS over (masked) public inputs; `t,d` depend on secret mask constants (`r`, `r_low`) but those are “baked into DCF keys”.
* The “correction LUT” is explicit (`make_corr_lut`) but the computation uses the linear form `(-t) + d*2^(64-f)` (same semantics, cheaper).

---

## 2) GeLU spline SUF gate (interval encoding + packed payload + post-processing)

Here we implement exactly what you asked:

* **interval encoding** (secret shared)
* **coefficient payload packing** into one interval-LUT PFSS program
* **post-processing:** `y = x^+ + δ(x)`

### 2.1 Interval design in unsigned Z2^64 (signed fixed-point interpretation)

Let signed fixed-point `x` be stored in two’s complement in `uint64_t`.

Define:

* sign boundary: `SIGN = 2^63`
* clipping bound `T` (already scaled by 2^f as an integer)

We represent the signed regions using unsigned intervals:

* negative central region `[-T, 0)` maps to `[2^64 - T, 2^64)`
* non-negative central `[0, T]` maps to `[0, T]`
* tails split by sign boundary to avoid mixing positives and negatives:

    * `[T, 2^63)` (large positive tail)
    * `[2^63, 2^64 - T)` (large negative tail)

So your spline intervals naturally split into two “bands” near 0 on both sides.

### 2.2 Payload layout (single PFSS call)

We pack, for each interval:

1. coefficients for `x_plus` (degree 1): `[a0, a1]` such that `a0 + a1*x` = either `x` or `0`
2. coefficients for `delta` (degree d): `[c0..cd]`
3. helper bits as arithmetic shares (0/1) but **packed as words**:

    * `w` (nonneg)
    * `c` (central-region flag)
    * `idx_bits[k]` (binary encoding of interval id), k = ceil(log2(numIntervals))

Thus `out_words = 2 + (d+1) + 2 + k`.

### 2.3 Code: packing + gate eval

```cpp
// include/gates/gelu_spline.h
#pragma once
#include <cstdint>
#include <vector>
#include <cmath>
#include <stdexcept>
#include "mpc/ring64.h"
#include "mpc/beaver.h"
#include "pfss/backend.h"

namespace gates {

struct GeluSplineParams {
  int frac_bits;     // f
  int poly_deg;      // d (<=3 typical)
  mpc::u64 T;        // clipping bound in *scaled integer units*
};

// One interval’s delta polynomial coeffs (degree d): c0..cd in Z2^64 fixed-point.
struct Poly {
  std::vector<mpc::u64> c; // size = d+1
};

struct GeluSplineSpec {
  GeluSplineParams p;

  // Spline nodes in signed domain, already scaled: a0 < ... < am, with a0=-T, am=+T.
  // Provide as int64_t in scaled units.
  std::vector<int64_t> knots_signed;           // size m+1
  std::vector<Poly>    delta_piece;            // size m (for central pieces only)

  // Tail policy:
  // left tail: delta=0, x_plus=0  => y=0
  // right tail: delta=0, x_plus=x => y=x
};

// Convert signed int64 to ring element in uint64 two's complement.
inline mpc::u64 to_u64_twos(int64_t x) {
  return static_cast<mpc::u64>(x);
}

inline int ceil_log2(int x) {
  int k=0; int v=1;
  while (v < x) { v <<= 1; k++; }
  return k;
}

// Dealer-side: build ONE interval-LUT description over masked coordinate x_hat.
struct GeluSplineGateKey {
  GeluSplineParams p;

  // input/output masks for masked-wire invariant
  mpc::Share64 r_in_share;
  mpc::Share64 r_out_share;

  // interval LUT PFSS key
  pfss::KeyBlob k_lut;

  // Beaver triples for polynomial eval:
  // Horner for x_plus degree1 uses 1 mul; delta degree d uses d muls.
  // Total per element: 1 + d multiplications.
  std::vector<mpc::TripleShare64> horner_triples; // size = 1 + d
};

// Pack per-interval payload words.
inline void append_payload(std::vector<mpc::u64>& payload,
                           const std::vector<mpc::u64>& words) {
  payload.insert(payload.end(), words.begin(), words.end());
}

// Construct unsigned-domain cutpoints and payloads.
// We create intervals in unsigned order with cutpoints increasing in [0,2^64).
// Central negative band: [2^64 - T, 2^64) split by negative knots.
// Central nonneg band: [0, T] split by nonneg knots.
// Other tails split at SIGN.
inline void build_gelu_interval_lut_desc(const GeluSplineSpec& spec,
                                        mpc::u64 r_in, // full mask value (dealer knows)
                                        pfss::IntervalLutDesc& out_desc) {
  const auto& P = spec.p;
  const int d = P.poly_deg;
  const mpc::u64 SIGN = (1ULL << 63);
  const mpc::u64 T = P.T;

  if (spec.knots_signed.size() < 2) throw std::runtime_error("need knots");
  if ((int)spec.delta_piece.size() != (int)spec.knots_signed.size()-1)
    throw std::runtime_error("delta_piece size mismatch");

  // Collect central knots split into negative and non-negative parts.
  std::vector<mpc::u64> neg_knots_u; // in [2^64-T, 2^64]
  std::vector<mpc::u64> pos_knots_u; // in [0, T]
  for (auto a : spec.knots_signed) {
    if (a < 0) neg_knots_u.push_back(to_u64_twos(a));
    else      pos_knots_u.push_back(to_u64_twos(a));
  }
  // Ensure 0 and -T and +T exist.
  // (In real code, validate and sort carefully.)

  // Build cutpoints in unsigned order:
  // 0 ... pos knots ... T ... SIGN ... (2^64 - T) ... neg knots ... 2^64
  std::vector<mpc::u64> cut;
  cut.push_back(0);

  // pos knots (>=0), sorted already if input is sorted in signed order.
  for (mpc::u64 ku : pos_knots_u) {
    if (ku != 0 && ku != T) cut.push_back(ku);
  }
  cut.push_back(T);
  cut.push_back(SIGN);

  const mpc::u64 NEG_START = (mpc::u64)(0ULL - T); // 2^64 - T
  cut.push_back(NEG_START);
  // neg knots are in increasing unsigned order near the top of the ring.
  for (mpc::u64 ku : neg_knots_u) {
    if (ku != NEG_START) cut.push_back(ku);
  }
  cut.push_back(0ULL); // interpret as 2^64 endpoint (will be handled as wrap in compiler)

  // Now define intervals between consecutive cutpoints in this cyclic list.
  // For an interval, we assign:
  // - x_plus poly: either x or 0
  // - delta poly: either spline piece (central) or 0 (tails)
  // - helper bits: w, c, idx_bits
  //
  // We flatten into a classic LUT with cutpoints in [0,2^64) increasing.
  // To avoid cyclic endpoint, we expand into [0..2^64) by dropping the last 0 and instead
  // using 2^64 as endpoint conceptually. Implementation detail: store cutpoints without final 0.
  cut.pop_back();
  // Ensure cut is strictly increasing; in real code sort+dedup with explicit construction.
  // Here assume it is.

  // Number of intervals:
  const int num_int = (int)cut.size(); // with implicit endpoint 2^64; intervals count = cut.size()
  const int idx_bits = ceil_log2(num_int);

  const int out_words = /*x_plus*/2 + /*delta*/(d+1) + /*w,c*/2 + idx_bits;

  out_desc.nbits = 64;
  out_desc.cutpoints = cut;
  out_desc.out_words = out_words;
  out_desc.payload_words.clear();
  out_desc.payload_words.reserve((size_t)num_int * out_words);

  auto is_central = [&](mpc::u64 x_u)->bool {
    // central if x in [0,T] or x in [2^64-T,2^64)
    return (x_u < T) || (x_u >= NEG_START);
  };
  auto is_nonneg = [&](mpc::u64 x_u)->bool {
    return x_u < SIGN;
  };

  // Helper to choose spline coeffs:
  // We map interval representative to signed x and pick the correct delta_piece.
  // For a production compiler, you’d build an explicit mapping from unsigned interval -> spline index.
  auto get_delta_coeffs_for_interval = [&](mpc::u64 interval_start_u)->std::vector<mpc::u64> {
    // Fallback: 0 poly
    std::vector<mpc::u64> z((size_t)d+1, 0ULL);

    // If interval_start_u is in central band, pick the correct knot region by signed compare.
    if (!is_central(interval_start_u)) return z;

    // Convert to signed for searching knots (two’s complement cast works for boundaries).
    int64_t xs = (int64_t)interval_start_u;

    // Find i s.t. knots[i] <= xs < knots[i+1] in signed order.
    int m = (int)spec.knots_signed.size() - 1;
    int idx = 0;
    while (idx+1 < (int)spec.knots_signed.size() && spec.knots_signed[idx+1] <= xs) idx++;
    if (idx >= m) idx = m-1;
    return spec.delta_piece[idx].c;
  };

  // Payload per interval
  for (int i=0; i<num_int; ++i) {
    mpc::u64 start = cut[i];

    // x_plus: if non-negative -> x (a0=0,a1=1), else 0 (a0=a1=0)
    mpc::u64 a0 = 0, a1 = is_nonneg(start) ? 1ULL : 0ULL;

    // delta coeffs
    auto delta = get_delta_coeffs_for_interval(start);

    // helper bits
    mpc::u64 w = is_nonneg(start) ? 1ULL : 0ULL;
    mpc::u64 c = is_central(start) ? 1ULL : 0ULL;

    // idx bits
    std::vector<mpc::u64> bits((size_t)idx_bits, 0ULL);
    for (int b=0; b<idx_bits; ++b) bits[b] = ( (i >> b) & 1 ) ? 1ULL : 0ULL;

    std::vector<mpc::u64> words;
    words.reserve((size_t)out_words);
    words.push_back(a0);
    words.push_back(a1);
    words.insert(words.end(), delta.begin(), delta.end());
    words.push_back(w);
    words.push_back(c);
    words.insert(words.end(), bits.begin(), bits.end());
    append_payload(out_desc.payload_words, words);
  }

  // Finally move to masked coordinate x_hat: add r_in to all cutpoints mod 2^64.
  // Full SUF→FSS needs wrap splitting; here we keep LUT simple and do a “rotate then sort”
  // which is equivalent (dealer-side).
  for (auto& cp : out_desc.cutpoints) cp = cp + r_in;
  // In production: sort cutpoints and rotate payloads accordingly; also split wrap if needed.
  // (Codex can finish this is as a dealer-side stable transformation.)
}

// Online Horner evaluation on shares.
inline mpc::Share64 horner_shared(const std::vector<mpc::Share64>& coeff, // size d+1
                                 mpc::Share64 x,
                                 const std::vector<mpc::TripleShare64>& tri,
                                 int party_id, mpc::IChannel& ch) {
  int d = (int)coeff.size() - 1;
  mpc::Share64 acc = coeff[d];
  for (int k=d-1; k>=0; --k) {
    acc = mpc::mul(acc, x, tri[(size_t)(d-1-k)], party_id, ch);
    acc = mpc::add(acc, coeff[k]);
  }
  return acc;
}

// Online gate eval: y = x_plus + delta, output masked.
inline void eval_gelu_spline(const GeluSplineGateKey& K,
                             int party_id,
                             pfss::IBackend& be,
                             mpc::IChannel& ch,
                             mpc::u64 x_hat_public,
                             mpc::Share64& out_hat_y_share,
                             mpc::BitShare64& out_w,
                             mpc::BitShare64& out_c,
                             std::vector<mpc::BitShare64>& out_idx_bits) {
  const int d = K.p.poly_deg;

  // masked->shares (Protocol 3.4):
  // party0: x0 = x_hat - r0 ; party1: x1 = -r1
  mpc::Share64 x_share{ (party_id==0) ? (x_hat_public - K.r_in_share.v) : (0ULL - K.r_in_share.v) };

  // Evaluate interval LUT: outputs packed words as additive shares
  const int out_words = /*x_plus*/2 + /*delta*/(d+1) + /*w,c*/2 + /*idx_bits*/(int)out_idx_bits.size();
  std::vector<mpc::u64> buf((size_t)out_words);
  be.eval_interval_lut(party_id, K.k_lut, x_hat_public, buf.data());

  // Unpack: x_plus coeffs
  mpc::Share64 a0{buf[0]}, a1{buf[1]};

  // delta coeffs
  std::vector<mpc::Share64> delta_coeff((size_t)d+1);
  for (int i=0;i<=d;++i) delta_coeff[i] = mpc::Share64{ buf[2 + i] };

  // helper bits
  out_w = mpc::BitShare64{ buf[2 + (d+1) + 0] };
  out_c = mpc::BitShare64{ buf[2 + (d+1) + 1] };

  // idx bits
  int base = 2 + (d+1) + 2;
  for (size_t i=0;i<out_idx_bits.size();++i) out_idx_bits[i] = mpc::BitShare64{ buf[base + (int)i] };

  // Compute x_plus = a0 + a1*x (degree 1): one multiplication
  // Reuse K.horner_triples[0] for (a1*x)
  mpc::Share64 a1x = mpc::mul(a1, x_share, K.horner_triples[0], party_id, ch);
  mpc::Share64 x_plus = mpc::add(a0, a1x);

  // delta(x) via Horner: needs d multiplications, use K.horner_triples[1..d]
  std::vector<mpc::TripleShare64> tri_delta;
  tri_delta.reserve((size_t)d);
  for (int i=0;i<d;++i) tri_delta.push_back(K.horner_triples[1 + i]);
  mpc::Share64 delta = (d==0) ? delta_coeff[0] : horner_shared(delta_coeff, x_share, tri_delta, party_id, ch);

  // y = x_plus + delta
  mpc::Share64 y = mpc::add(x_plus, delta);

  // output masked: hat{y} share
  out_hat_y_share = mpc::add(y, K.r_out_share);
}

} // namespace gates
```

**What this gives you (exactly what you asked):**

* **Interval encoding:** `idx_bits` are part of the LUT payload (no extra PFSS calls).
* **Coefficient payload packing:** `out_words` is a single contiguous payload vector per interval (again, one PFSS call).
* **Post-processing:** `y = x_plus + delta` is explicit.

---

## 3) CPU vs GPU versions (how to structure it)

### CPU path

* Evaluate PFSS per element (or batched) on CPU
* Run Beaver multiplications and Horner on CPU
* This is already supported by the code above.

### GPU path (practical approach)

Do *not* try to “remove interaction” (Beaver still needs one round). Instead:

1. Run PFSS evaluation in batch on GPU (backend-dependent; `myl7/fss` has CUDA code)
2. For each Beaver multiplication layer:

    * CUDA kernel computes local `d_i, e_i` arrays
    * copy `(d_i,e_i)` to host, exchange with peer, reconstruct `(d,e)` public arrays
    * copy `(d,e)` back to device
    * CUDA kernel finishes `z_i = c_i + d*b_i + e*a_i (+ d*e if party0)`
3. Horner is then just repeated Beaver multiplications + adds (all massively parallel).

Your `pfss::IBackend` already has `*_batch_gpu(...)` hooks, so the gates don’t change—only the backend + multiplication implementation do.

If you want SIGMA-like efficiency later: replace “many DCF calls” with **packed comparison** and replace interval LUT with **vector-payload LUT** (same `IntervalLutDesc`, but one key, wide payload). That’s a backend optimization; the gate logic stays exactly the same.

---

If you want, in the next step I can:

* write the **dealer/offline keygen** code for both gates (sampling `r_in,r_out`, generating DCF/LUT keys, and generating all Beaver triples),
* and give a concrete `myl7/fss` adapter skeleton (`Myl7FssBackend`) matching its “bits-in/bytes-out” model  while still reconstructing arithmetic shares in `uint64_t`.

[1]: https://github.com/facebookresearch/GPU-DPF?utm_source=chatgpt.com "GPU-based Distributed Point Functions (DPF) and 2- ..."
