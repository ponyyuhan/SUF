Below is a **concrete, code-level offline/dealer design** for **(1) ReluARS** and **(2) GeLU-spline SUF**, plus a **`Myl7FssBackend` adapter skeleton** that matches **myl7/fss’s “bits-in / bytes-out”** restriction (2-party only, fixed), while reconstructing outputs as **`uint64_t` ring shares**. ([GitHub][1])

I’ll use **`R = Z_{2^64}`** everywhere (so `uint64_t` arithmetic, wrapping as mod (2^{64})).

---

## 0) What we assume from `myl7/fss` (adapter contract)

`myl7/fss` explicitly fixes: **2-party**, **input=bits**, **output=bytes** (users define how to interpret/add outputs). ([GitHub][1])
So the adapter below exposes:

* `GenDCF(threshold α, payload bytes)` → `(k0,k1)`
* `EvalDCF(kb, x_bits)` → `out_bytes` (share)
* same idea for `DPF` if you later want it.

This is enough to implement:

* “predicate programs”: return 1-byte shares (GF(2) XOR shares) or 8-byte shares (Z_{2^64} additive shares),
* “coefficient/LUT programs”: return `8 * num_words` bytes interpreted as `num_words` `uint64_t` additive shares.

---

## 1) Common utilities (ring ops, packing, randomness, Beaver triples)

### `common.h`

```cpp
#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using u8  = uint8_t;
using u64 = uint64_t;
using u128 = unsigned __int128;

inline u64 add_mod(u64 a, u64 b) { return a + b; }     // wraps mod 2^64
inline u64 sub_mod(u64 a, u64 b) { return a - b; }     // wraps mod 2^64
inline u64 mul_mod(u64 a, u64 b) { return (u64)((u128)a * (u128)b); }

inline u64 rot_add(u64 x, u64 r) { return x + r; }     // mod 2^64

inline u64 mask_low(u64 x, int f) {
  if (f <= 0) return 0;
  if (f >= 64) return x;
  return x & ((u64(1) << f) - 1);
}

inline u64 one_hot_bit(u8 b) { return (u64)(b & 1); }

// Little-endian packing helpers
inline std::vector<u8> pack_u64_le(u64 x) {
  std::vector<u8> out(8);
  std::memcpy(out.data(), &x, 8);
  return out;
}

inline u64 unpack_u64_le(const u8* p) {
  u64 x;
  std::memcpy(&x, p, 8);
  return x;
}

inline std::vector<u8> pack_u64_vec_le(const std::vector<u64>& ws) {
  std::vector<u8> out(ws.size() * 8);
  for (size_t i = 0; i < ws.size(); i++) {
    std::memcpy(out.data() + 8*i, &ws[i], 8);
  }
  return out;
}

inline std::vector<u64> unpack_u64_vec_le(const std::vector<u8>& bytes) {
  if (bytes.size() % 8 != 0) throw std::runtime_error("unpack_u64_vec_le: size not multiple of 8");
  size_t n = bytes.size() / 8;
  std::vector<u64> ws(n);
  for (size_t i = 0; i < n; i++) ws[i] = unpack_u64_le(bytes.data() + 8*i);
  return ws;
}
```

### `secure_rand.h` (simple “good enough” randomness for a prototype)

For a paper artifact you’ll likely swap this for OS `getrandom()` / libsodium / OpenSSL.

```cpp
#pragma once
#include "common.h"
#include <random>

struct SecureRand {
  std::random_device rd;

  u64 rand_u64() {
    // Note: std::random_device quality varies by platform; replace for production.
    u64 x = 0;
    for (int i = 0; i < 4; i++) {
      x = (x << 16) ^ (u64)(rd() & 0xFFFFu);
    }
    return x;
  }

  u8 rand_bit() { return (u8)(rand_u64() & 1u); }
};
```

### `beaver.h`

```cpp
#pragma once
#include "common.h"
#include "secure_rand.h"

struct BeaverTriple64Share {
  u64 a, b, c; // additive share in Z_2^64
};
struct BeaverTripleBitShare {
  u8 a, b, c;  // XOR share in GF(2)
};

struct BeaverDealer {
  SecureRand rng;

  // Sample (a,b,c=a*b) in Z_2^64, split into 2 additive shares.
  std::pair<BeaverTriple64Share, BeaverTriple64Share> gen_triple64() {
    u64 a = rng.rand_u64();
    u64 b = rng.rand_u64();
    u64 c = mul_mod(a, b);

    u64 a0 = rng.rand_u64(); u64 a1 = sub_mod(a, a0);
    u64 b0 = rng.rand_u64(); u64 b1 = sub_mod(b, b0);
    u64 c0 = rng.rand_u64(); u64 c1 = sub_mod(c, c0);

    return {BeaverTriple64Share{a0,b0,c0}, BeaverTriple64Share{a1,b1,c1}};
  }

  // Sample (a,b,c=a&b) in GF(2), split into 2 XOR shares.
  std::pair<BeaverTripleBitShare, BeaverTripleBitShare> gen_triple_bit() {
    u8 a = rng.rand_bit();
    u8 b = rng.rand_bit();
    u8 c = (u8)(a & b);

    u8 a0 = rng.rand_bit(); u8 a1 = (u8)(a ^ a0);
    u8 b0 = rng.rand_bit(); u8 b1 = (u8)(b ^ b0);
    u8 c0 = rng.rand_bit(); u8 c1 = (u8)(c ^ c0);

    return {BeaverTripleBitShare{a0,b0,c0}, BeaverTripleBitShare{a1,b1,c1}};
  }

  // Split a ring element into 2 additive shares
  std::pair<u64,u64> split_add(u64 x) {
    u64 x0 = rng.rand_u64();
    u64 x1 = sub_mod(x, x0);
    return {x0,x1};
  }

  // Split a bit into 2 XOR shares
  std::pair<u8,u8> split_xor(u8 b) {
    u8 b0 = rng.rand_bit();
    u8 b1 = (u8)(b ^ b0);
    return {b0,b1};
  }
};
```

---

## 2) PFSS backend interface + `Myl7FssBackend` skeleton

### `pfss_backend.h`

```cpp
#pragma once
#include "common.h"

struct FssKey {
  std::vector<u8> bytes;
};

struct DcfKeyPair {
  FssKey k0;
  FssKey k1;
};

struct PfssBackend {
  virtual ~PfssBackend() = default;

  // Program a DCF for: f(x)=payload if x < alpha else 0.
  virtual DcfKeyPair gen_dcf(
      int in_bits,
      const std::vector<u8>& alpha_bits,   // bits-in model
      const std::vector<u8>& payload_bytes // bytes-out model
  ) = 0;

  // Evaluate one party's share.
  virtual std::vector<u8> eval_dcf(
      int in_bits,
      const FssKey& kb,
      const std::vector<u8>& x_bits
  ) const = 0;

  // Helpers: encode uint64->bits, bits->uint64, etc.
  virtual std::vector<u8> u64_to_bits_msb(u64 x, int in_bits) const = 0;
};
```

### `myl7_fss_backend.h` (adapter skeleton)

The only “myl7-specific” part is **how to call its DCF gen/eval APIs**; everything else is real.

```cpp
#pragma once
#include "pfss_backend.h"

// myl7/fss states: input=bits, output=bytes, 2-party. :contentReference[oaicite:2]{index=2}
// You must wire these wrappers to the actual library symbols/classes.

class Myl7FssBackend final : public PfssBackend {
public:
  struct Params {
    // Typical FSS security parameter in bytes (must be multiple of 8 in myl7/fss). :contentReference[oaicite:3]{index=3}
    int lambda_bytes = 16; // 128-bit
    bool bits_msb_first = true; // adjust to match myl7's bit ordering
  };

  explicit Myl7FssBackend(Params p) : params_(p) {}

  std::vector<u8> u64_to_bits_msb(u64 x, int in_bits) const override {
    std::vector<u8> bits(in_bits);
    for (int i = 0; i < in_bits; i++) {
      int shift = (in_bits - 1 - i);
      bits[i] = (u8)((x >> shift) & 1u);
    }
    return bits;
  }

  // Convenience: low-f bits only
  std::vector<u8> u64_low_to_bits_msb(u64 x, int f) const {
    return u64_to_bits_msb(mask_low(x, f), f);
  }

  DcfKeyPair gen_dcf(int in_bits,
                     const std::vector<u8>& alpha_bits,
                     const std::vector<u8>& payload_bytes) override {
    // TODO: Replace with real myl7/fss DCF Gen call
    // Expected behavior: returns (k0,k1) where Eval(k0,x)+Eval(k1,x) = (x<alpha?payload:0)
    // and output is "bytes-out".
    (void)in_bits; (void)alpha_bits; (void)payload_bytes;
    throw std::runtime_error("Myl7FssBackend::gen_dcf not wired to myl7/fss yet");
  }

  std::vector<u8> eval_dcf(int in_bits,
                           const FssKey& kb,
                           const std::vector<u8>& x_bits) const override {
    // TODO: Replace with real myl7/fss DCF Eval call
    (void)in_bits; (void)kb; (void)x_bits;
    throw std::runtime_error("Myl7FssBackend::eval_dcf not wired to myl7/fss yet");
  }

private:
  Params params_;
};
```

> Why this matches your “uint64 reconstruction” requirement:
>
> * If you set `payload_bytes = pack_u64_vec_le(coeff_words)`, each party’s `eval_dcf()` returns a byte string; interpret each `8` bytes as a `uint64_t` **additive** share in `Z_2^64`.
> * If you set `payload_bytes = {1}` and interpret as **1 byte**, you can reconstruct a boolean bit via XOR or via mod-2 addition, depending on how you choose to represent predicate shares.

---

## 3) Dealer/offline keygen for **ReluARS**

We generate:

* masks `r_in, r_out` and per-party additive shares,
* DCF keys to derive helper bits `(w,t,d)` on **public `hatx`**,
* Beaver triples needed by the eventual truncation + correction circuit.

### Helper-bit plan (practical, mask-friendly)

We treat:

* `w` = “non-negative” bit of **unmasked** signed `x` (after two’s complement).
* `t` = truncation carry/borrow bit for dividing by `2^f` via “open hatx” style formulas (fast and common in FSS pipelines).
* `d` = a configurable predicate on the low bits to select small “gap correction”; below we implement `d = 1[low_bits == 0]` as a **placeholder** (wire in SHARK/SIGMA’s exact predicate later).

All of these can be implemented with DCFs on:

* `hatx` (64-bit) and
* `hatx_low = hatx mod 2^f` (f-bit).

### `reluars_dealer.h`

```cpp
#pragma once
#include "beaver.h"
#include "pfss_backend.h"

struct ReluARSParams {
  int f = 12;                 // fractional bits for ARS/trunc
  std::array<u64, 8> delta;   // correction LUT Δ[w,t,d] (public constants; placeholder)
};

// DCF program handles (only what dealer must ship).
struct ReluARSPartyKey {
  // Mask shares
  u64 r_in_share = 0;
  u64 r_out_share = 0;

  // Wrap flags (depend only on random r_in, safe to reveal)
  bool wrap_sign = false;
  bool wrap_half = false; // for any half-range rotations you choose

  // DCF keys for comparisons:
  // 64-bit domain (hatx): comparisons to r_in and r_in+2^63 (mod 2^64)
  FssKey dcf_hat_lt_r;
  FssKey dcf_hat_lt_r_plus_2p63;

  // f-bit domain (hatx_low): comparison to r_low (carry) and for d predicate
  FssKey dcf_low_lt_r_low;
  FssKey dcf_low_lt_r_low_plus1;

  // Beaver triples (ring + bit) the gate will consume online.
  std::vector<BeaverTriple64Share> triples64;
  std::vector<BeaverTripleBitShare> triplesBit;

  // Public parameters for online evaluation
  ReluARSParams params;
};

struct ReluARSDealerOut {
  ReluARSPartyKey k0;
  ReluARSPartyKey k1;
};

class ReluARSDealer {
public:
  // Dealer generates everything offline.
  static ReluARSDealerOut keygen(const ReluARSParams& p, PfssBackend& fss, BeaverDealer& dealer) {
    ReluARSDealerOut out;
    out.k0.params = p;
    out.k1.params = p;

    // 1) Sample masks
    u64 r_in  = dealer.rng.rand_u64();
    u64 r_out = dealer.rng.rand_u64();

    auto [r_in0, r_in1]   = dealer.split_add(r_in);
    auto [r_out0, r_out1] = dealer.split_add(r_out);

    out.k0.r_in_share  = r_in0;
    out.k1.r_in_share  = r_in1;
    out.k0.r_out_share = r_out0;
    out.k1.r_out_share = r_out1;

    // 2) Program DCFs for helper bits on public hatx.

    // 2a) Sign boundary at 2^63: compute using masked-interval logic
    // We will provide parties sufficient comparison shares to assemble w.
    const u64 TWO63 = (u64(1) << 63);
    u64 thr1 = r_in;                     // r
    u64 thr2 = r_in + TWO63;             // r + 2^63 (wraps automatically mod 2^64)
    bool wrap = (thr2 < thr1);           // wrap iff overflow

    out.k0.wrap_sign = wrap;
    out.k1.wrap_sign = wrap;

    // DCF outputs 1-byte payload (GF(2) share); f(x)=1 if x<thr else 0.
    auto one_byte = std::vector<u8>{1u};

    // NOTE: alpha_bits encodes the *threshold*; x_bits encodes the *input*.
    auto thr1_bits = fss.u64_to_bits_msb(thr1, 64);
    auto thr2_bits = fss.u64_to_bits_msb(thr2, 64);

    auto kp1 = fss.gen_dcf(64, thr1_bits, one_byte);
    auto kp2 = fss.gen_dcf(64, thr2_bits, one_byte);

    out.k0.dcf_hat_lt_r = kp1.k0;
    out.k1.dcf_hat_lt_r = kp1.k1;

    out.k0.dcf_hat_lt_r_plus_2p63 = kp2.k0;
    out.k1.dcf_hat_lt_r_plus_2p63 = kp2.k1;

    // 2b) Trunc carry bit: t = 1[(hatx mod 2^f) < (r_in mod 2^f)]
    u64 r_low = mask_low(r_in, p.f);
    auto rlow_bits = fss.u64_to_bits_msb(r_low, p.f);
    auto kpt = fss.gen_dcf(p.f, rlow_bits, one_byte);

    out.k0.dcf_low_lt_r_low = kpt.k0;
    out.k1.dcf_low_lt_r_low = kpt.k1;

    // 2c) Placeholder d: d = 1[ (x_low == 0) ]  => in masked low bits:
    // if low_hat = (x_low + r_low) mod 2^f, then x_low==0 <=> low_hat == r_low.
    // Equality can be tested by: (low_hat < r_low+1) AND NOT(low_hat < r_low).
    u64 r_low_plus1 = (p.f == 64) ? (r_low + 1) : ((r_low + 1) & ((u64(1)<<p.f)-1));
    auto rlow1_bits = fss.u64_to_bits_msb(r_low_plus1, p.f);
    auto kpd = fss.gen_dcf(p.f, rlow1_bits, one_byte);

    out.k0.dcf_low_lt_r_low_plus1 = kpd.k0;
    out.k1.dcf_low_lt_r_low_plus1 = kpd.k1;

    // 3) Beaver triples: conservative counts (tune to your exact online circuit)
    // - a few ring mults for combining (w,t,d) bits with arithmetic,
    // - a few bit ANDs for correction LUT selection.
    //
    // If you implement everything in the ring (treat bits as 0/1 in Z_2^64),
    // you can drop bit-triples and just use 64-bit triples.
    const int need_triples64 = 16;
    const int need_triplesBit = 32;

    out.k0.triples64.reserve(need_triples64);
    out.k1.triples64.reserve(need_triples64);
    for (int i = 0; i < need_triples64; i++) {
      auto [t0, t1] = dealer.gen_triple64();
      out.k0.triples64.push_back(t0);
      out.k1.triples64.push_back(t1);
    }

    out.k0.triplesBit.reserve(need_triplesBit);
    out.k1.triplesBit.reserve(need_triplesBit);
    for (int i = 0; i < need_triplesBit; i++) {
      auto [t0, t1] = dealer.gen_triple_bit();
      out.k0.triplesBit.push_back(t0);
      out.k1.triplesBit.push_back(t1);
    }

    return out;
  }
};
```

**Notes for “SIGMA-like efficiency”:**
This dealer is already “compiler-friendly”: all thresholds are known at keygen, and outputs are small (1 byte). The big performance win (SIGMA-style) comes from:

* batched eval for many `hatx` in parallel,
* packed comparison (CDPF-style) so one tree walk outputs many threshold bits,
* a fast PRG (AES-NI or CUDA PRG). `myl7/fss` has CUDA in the repo, and is intentionally a primitive library (good baseline). ([GitHub][1])

---

## 4) Dealer/offline keygen for **GeLU spline SUF** (intervals + packed coefficient payload via “DCFs with vector payloads”)

We implement the coefficient “program” as a **piecewise-constant vector function** over the **masked, biased** input:

* biased value: `x_bias = x + 2^63` (so signed order becomes unsigned order),
* public masked input for predicates/LUT: `hatx_bias = hatx + 2^63` (just add a public constant),
* rotate by `r_in`, split wrap segments, sort.

Then we represent the piecewise-constant coefficient vector function by a prefix-sum of steps, and instantiate each step by a DCF with **multi-word payload bytes**.

### Data model

* degree `d <= 3`,
* coefficients vector = `[a0, a1, ..., ad]` as `uint64_t` in `Z_2^64`,
* each piece has one such vector for δ(x),
* outside central region, δ(x)=0 vector.

### `gelu_spline_dealer.h`

```cpp
#pragma once
#include "beaver.h"
#include "pfss_backend.h"
#include <algorithm>

struct GeluSplineParams {
  int f = 12;
  int d = 3;         // spline degree
  u64 T = 0;         // clipping bound in fixed-point (positive) as uint64 (interpreted signed)

  // Central spline boundaries in signed space: a0=-T < a1 < ... < am=T
  // Store as int64_t in two's complement, but dealer keeps in u64 representation.
  std::vector<int64_t> a;                 // size m+1, with a.front()=-T, a.back()=+T

  // For each central interval i in [0..m-1], polynomial coeffs for δ(x):
  // coeffs[i] is vector length d+1 of uint64_t in Z_2^64
  std::vector<std::vector<u64>> coeffs;   // size m, each length (d+1)
};

struct DcfVecProgramPartyKey {
  FssKey dcf_key;
};

struct StepCut {
  u64 start;                 // cutpoint in hatx_bias domain
  std::vector<u64> delta;    // Δ vector (public constant, used as “party0 adds Δ” trick)
  // Each cut has 1 DCF programmed to output Δ when x < start else 0 (bytes-out).
  DcfVecProgramPartyKey party0;
  DcfVecProgramPartyKey party1;
};

struct GeluSplinePartyKey {
  // mask shares
  u64 r_in_share = 0;
  u64 r_out_share = 0;

  // coefficient program: a list of step cuts (start points, deltas, and DCF keys)
  std::vector<StepCut> cuts;

  // triples for Horner (δ(x), degree d) + x^+ multiply-by-bit + misc
  std::vector<BeaverTriple64Share> triples64;
  std::vector<BeaverTripleBitShare> triplesBit;

  GeluSplineParams params;
};

struct GeluSplineDealerOut {
  GeluSplinePartyKey k0;
  GeluSplinePartyKey k1;
};

namespace gelu_internal {

// signed int64 -> biased uint64 (map [-2^63,2^63-1] to [0,2^64-1])
inline u64 bias(int64_t x) {
  return (u64)x + (u64(1) << 63);
}

// A single non-wrapping segment [start,end) with payload v
struct Segment {
  u64 start;
  u64 end; // start < end, non-wrapping
  std::vector<u64> v;
};

inline std::vector<Segment> rotate_and_split_segments(
    const std::vector<u64>& boundaries_bias,              // includes 0 and 2^64 as conceptual end
    const std::vector<std::vector<u64>>& payloads,        // payloads per interval
    u64 r_in                                               
) {
  // boundaries_bias length = N+1, payloads length = N
  if (boundaries_bias.size() < 2) throw std::runtime_error("need >=2 boundaries");
  if (payloads.size() + 1 != boundaries_bias.size()) throw std::runtime_error("payloads/boundaries mismatch");

  std::vector<Segment> segs;
  segs.reserve(payloads.size() * 2);

  for (size_t i = 0; i < payloads.size(); i++) {
    u64 a = boundaries_bias[i];
    u64 b = boundaries_bias[i+1];

    // rotate by r_in in biased domain: hatx_bias = x_bias + r_in
    u64 s = a + r_in;
    u64 e = b + r_in;

    // note: (a,b) are in [0,2^64], but stored as u64; b might be "2^64" conceptually.
    // We treat b==0 as wrap end. In practice, require boundaries end at 0 for u64 wrap,
    // so we handle segment end with split rules by comparing.
    if (s < e) {
      segs.push_back(Segment{s, e, payloads[i]});
    } else if (s > e) {
      // wrapping: split into [s,2^64) and [0,e)
      segs.push_back(Segment{s, (u64)0, payloads[i]}); // end=0 means 2^64 in u64 wrap sense; handled later
      segs.push_back(Segment{0, e, payloads[i]});
    } else {
      // s==e means interval length 0 (should not happen if boundaries strictly increasing)
      throw std::runtime_error("degenerate rotated segment");
    }
  }

  // normalize end=0 to mean 2^64 by sorting with special handling:
  // We'll just sort by start; evaluation can treat cutpoints list; we only need starts.
  std::sort(segs.begin(), segs.end(), [](const Segment& x, const Segment& y){
    return x.start < y.start;
  });

  return segs;
}

} // namespace gelu_internal

class GeluSplineDealer {
public:
  static GeluSplineDealerOut keygen(const GeluSplineParams& p, PfssBackend& fss, BeaverDealer& dealer) {
    if ((int)p.a.size() < 2) throw std::runtime_error("need a0..am boundaries");
    if ((int)p.coeffs.size() != (int)p.a.size() - 1) throw std::runtime_error("coeffs size mismatch");
    for (auto& cv : p.coeffs) {
      if ((int)cv.size() != p.d + 1) throw std::runtime_error("each coeff vector must be d+1");
    }

    GeluSplineDealerOut out;
    out.k0.params = p;
    out.k1.params = p;

    // 1) masks
    u64 r_in  = dealer.rng.rand_u64();
    u64 r_out = dealer.rng.rand_u64();
    auto [r_in0, r_in1]   = dealer.split_add(r_in);
    auto [r_out0, r_out1] = dealer.split_add(r_out);

    out.k0.r_in_share  = r_in0;
    out.k1.r_in_share  = r_in1;
    out.k0.r_out_share = r_out0;
    out.k1.r_out_share = r_out1;

    // 2) Build piecewise δ(x) coefficient vectors in biased domain.
    //
    // Domain in signed space is [-2^63, 2^63). In biased space it becomes [0,2^64).
    // We create a full partition:
    //   [0, bias(a0))        : δ=0
    //   [bias(a_i), bias(a_{i+1})) : δ=coeffs[i]
    //   [bias(am), 2^64)     : δ=0
    //
    // Note: a0 should be -T, am should be +T as int64_t.
    std::vector<u64> boundaries;
    boundaries.reserve(p.a.size() + 2);
    boundaries.push_back(0);

    for (size_t i = 0; i < p.a.size(); i++) {
      boundaries.push_back(gelu_internal::bias(p.a[i]));
    }

    // Ensure last boundary equals bias(+T), but still need a conceptual end=2^64.
    // Represent "2^64" as 0 in u64 wrap world; we handle by ending with 0 and requiring split logic.
    // We'll instead explicitly handle the tail by adding a final "end" marker as 0 and treat it as wrap.
    // For simplicity in dealer logic, we create payloads as if boundaries are strictly increasing mod 2^64,
    // and rotate/split later.
    //
    // We'll enforce increasing in biased domain:
    for (size_t i = 1; i < boundaries.size(); i++) {
      if (boundaries[i] <= boundaries[i-1]) throw std::runtime_error("boundaries must be strictly increasing in biased space");
    }

    // payloads per interval among boundaries:
    // interval count = boundaries.size() - 1, but we also add final tail to 2^64
    // We'll construct intervals:
    // 0) [0, bias(a0)) : 0
    // 1..m) [bias(ai), bias(a_{i+1})) : coeffs[i]
    // tail) [bias(am), 2^64) : 0
    std::vector<std::vector<u64>> payloads;

    auto zero_vec = std::vector<u64>(p.d + 1, 0);

    // left tail
    payloads.push_back(zero_vec);

    // central pieces
    for (size_t i = 0; i < p.coeffs.size(); i++) payloads.push_back(p.coeffs[i]);

    // right tail
    payloads.push_back(zero_vec);

    // Now extend boundaries with conceptual end=2^64.
    // We'll represent end as 0 and handle wrap after rotation.
    boundaries.push_back(0); // conceptual 2^64 end marker (wrap)

    // payloads should be boundaries.size()-1
    if (payloads.size() + 1 != boundaries.size()) {
      throw std::runtime_error("payloads/boundaries internal mismatch");
    }

    // 3) Rotate by r_in and split wrap segments in hatx_bias domain.
    auto segs = gelu_internal::rotate_and_split_segments(boundaries, payloads, r_in);

    // Normalize the "end=0 means 2^64" only for carry; for step cuts we only need starts & ordering.
    // Ensure we have a segment that starts at 0:
    if (segs.empty() || segs[0].start != 0) {
      // If rotation never produced a segment starting at 0, it means all segments were in (0,2^64),
      // but after wrap split we should always have one at 0. If not, something is off.
      throw std::runtime_error("expected a segment starting at 0 after rotate/split");
    }

    // 4) Convert segments into step cuts for a DCF-prefix construction:
    // f(x) = v0 + Σ_{j>=1} Δ_j * 1[x >= start_j]
    //
    // Implement 1[x>=s] using DCF(x < s) that outputs Δ:
    // term = Δ - DCF_out.
    // To realize "Δ" additively without extra sharing overhead, we do:
    //   party0 adds Δ, party1 adds 0, and both subtract DCF_out share.
    //
    // DCF outputs bytes interpreted as vector<u64> shares.
    out.k0.cuts.clear(); out.k1.cuts.clear();
    out.k0.cuts.reserve(segs.size() > 0 ? segs.size()-1 : 0);
    out.k1.cuts.reserve(segs.size() > 0 ? segs.size()-1 : 0);

    // base v0 is segs[0].v; party0 will start with v0, party1 starts with 0.
    // We don't store v0 as a cut; we store it as an implicit "init add".
    // For simplicity, store v0 in params? Here: just rely on online code to do it.
    // (If you prefer, you can store it in the party key.)
    //
    // We'll store v0 explicitly as a special cut with start=0 and delta=v0 and no DCF;
    // but then the online code must handle it specially anyway. We'll keep it implicit.
    const std::vector<u64> v0 = segs[0].v;

    // Build cuts for j>=1:
    for (size_t j = 1; j < segs.size(); j++) {
      const u64 start = segs[j].start;
      // Δ = v_j - v_{j-1} in ring, component-wise
      std::vector<u64> delta(v0.size());
      for (size_t t = 0; t < delta.size(); t++) {
        delta[t] = sub_mod(segs[j].v[t], segs[j-1].v[t]);
      }

      // Program DCF with threshold=start (in hatx_bias domain) and payload=Δ (bytes-out).
      auto start_bits = fss.u64_to_bits_msb(start, 64);
      auto payload_bytes = pack_u64_vec_le(delta);
      auto kp = fss.gen_dcf(64, start_bits, payload_bytes);

      StepCut cut;
      cut.start = start;
      cut.delta = delta;
      cut.party0.dcf_key = kp.k0;
      cut.party1.dcf_key = kp.k1;

      out.k0.cuts.push_back(cut);
      out.k1.cuts.push_back(cut);
      // (Note: both party keys store the same start/delta, but only their own dcf_key differs.)
      out.k0.cuts.back().party1.dcf_key.bytes.clear(); // optional: strip other party's key
      out.k1.cuts.back().party0.dcf_key.bytes.clear(); // optional: strip other party's key
    }

    // 5) Beaver triples (degree-d Horner for δ + a few extras)
    // Horner needs d multiplications for δ(x).
    // x^+ typically needs 1 multiplication by sign-bit if you do it that way.
    const int need_triples64 = p.d + 8;
    const int need_triplesBit = 32; // if you compute interval flags via GF(2) ANDs, etc.

    out.k0.triples64.reserve(need_triples64);
    out.k1.triples64.reserve(need_triples64);
    for (int i = 0; i < need_triples64; i++) {
      auto [t0, t1] = dealer.gen_triple64();
      out.k0.triples64.push_back(t0);
      out.k1.triples64.push_back(t1);
    }

    out.k0.triplesBit.reserve(need_triplesBit);
    out.k1.triplesBit.reserve(need_triplesBit);
    for (int i = 0; i < need_triplesBit; i++) {
      auto [t0, t1] = dealer.gen_triple_bit();
      out.k0.triplesBit.push_back(t0);
      out.k1.triplesBit.push_back(t1);
    }

    return out;
  }
};
```

### What “payload packing” means here

Each DCF has **payload bytes** equal to `8*(d+1)` bytes (or more if you pack multiple outputs).
Those bytes decode to `uint64_t` vector shares, so you can run Horner in `Z_2^64` directly.

---

## 5) Where this becomes “SIGMA-like” fast (without changing your gate interfaces)

The above is a correct *compiler-friendly* offline structure. For SIGMA-like throughput, you upgrade the backend without changing gate code:

1. **Batch eval API** (critical): add

```cpp
eval_dcf_batch(key, xs_bits[]) -> outs_bytes[]
```

so you can evaluate an entire tensor layer at once. Your gate evaluation becomes “one kernel per DCF program per layer”.

2. **Packed comparisons**: produce many predicate bits with one tree walk (CDPF style). Your own writeup mentions this path; it’s exactly the “one control tree, multi-bit payload” approach SIGMA-style systems use.

3. **GPU**: if you want a known reference point, GPU DPF implementations show order-of-magnitude speedups by doing the tree expansion on GPU. ([GitHub][2])
   `myl7/fss` already has CUDA in-tree, so it’s a natural place to hang a GPU batch evaluator. ([GitHub][1])

---

## 6) One important practicality: representation of predicate outputs

Because `myl7/fss` is “bytes-out”, you get to choose how predicates are represented:

* **GF(2) XOR-shared bits**: easiest for Boolean circuits; use 1 output byte with LSB = bit.
* **Z_{2^64} additive shares of {0,1}**: easiest to multiply into arithmetic without a B2A conversion; use 8 output bytes, decode to `u64`.

For “SIGMA-like speed”, many systems output predicate bits as machine-word shares to reduce conversions.

---

If you want, in the next step I can drop in:

* a concrete **online evaluator** for both gates (including the correction-LUT selection circuit for `(w,t,d)` and Horner for GeLU),
* a **batching-oriented API** shape that cleanly maps to CPU SIMD and a CUDA kernel launch strategy,
* and a second backend stub `SigmaFastBackend` that keeps the same `PfssBackend` interface but supports packed multi-threshold comparisons.

[1]: https://github.com/myl7/fss "GitHub - myl7/fss: Function secret sharing (FSS) primitives including distributed point functions (DPF) and distributed comparison functions (DCF)"
[2]: https://github.com/facebookresearch/GPU-DPF?utm_source=chatgpt.com "GPU-based Distributed Point Functions (DPF) and 2-server ..."
