Below is **drop‑in**, header‑only code for:

1. `include/proto/common.hpp`
2. `include/proto/tape.hpp`

…and then a **patch‑style outline** showing **exactly** how `reluars_dealer.hpp` writes a per‑party tape and how the online evaluator consumes it **deterministically**, without changing your existing in‑memory key path.

Finally, I’ll give you **Milestone 3/4** (SUF IR hardening + mask‑rewrite correctness) as the next concrete steps in *your* repo.

---

## 1) Drop‑in: `include/proto/common.hpp`

This is intentionally minimal and should coexist with your existing `core/*`. If you already have one, you can either replace it or merge the bits you need.

```cpp
// include/proto/common.hpp
#pragma once

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace proto {

using u8  = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

// Fixed-point for your current build (Option A + f=12).
static constexpr int kFracBits = 12;

// --------------------------
// Exceptions / checking
// --------------------------
[[noreturn]] inline void fail(const char* msg) {
  throw std::runtime_error(std::string("proto: ") + msg);
}
inline void ensure(bool ok, const char* msg) { if (!ok) fail(msg); }

// --------------------------
// Endian-stable encoding
// (explicit little-endian, works on any host)
// --------------------------
inline void store_u32_le(u32 x, u8 out[4]) {
  out[0] = (u8)(x >> 0);
  out[1] = (u8)(x >> 8);
  out[2] = (u8)(x >> 16);
  out[3] = (u8)(x >> 24);
}
inline u32 load_u32_le(const u8 in[4]) {
  return (u32(in[0]) << 0) | (u32(in[1]) << 8) | (u32(in[2]) << 16) | (u32(in[3]) << 24);
}

inline void store_u64_le(u64 x, u8 out[8]) {
  out[0] = (u8)(x >> 0);
  out[1] = (u8)(x >> 8);
  out[2] = (u8)(x >> 16);
  out[3] = (u8)(x >> 24);
  out[4] = (u8)(x >> 32);
  out[5] = (u8)(x >> 40);
  out[6] = (u8)(x >> 48);
  out[7] = (u8)(x >> 56);
}
inline u64 load_u64_le(const u8 in[8]) {
  return (u64(in[0]) << 0)  | (u64(in[1]) << 8)  | (u64(in[2]) << 16) | (u64(in[3]) << 24) |
         (u64(in[4]) << 32) | (u64(in[5]) << 40) | (u64(in[6]) << 48) | (u64(in[7]) << 56);
}

template <class T>
inline u32 checked_u32_len(T n) {
  static_assert(std::is_integral_v<T>);
  ensure(n >= 0, "negative length");
  ensure((std::uint64_t)n <= std::numeric_limits<u32>::max(), "length overflow u32");
  return (u32)n;
}

} // namespace proto
```

---

## 2) Drop‑in: `include/proto/tape.hpp`

This provides:

* `TapeWriter` / `TapeReader` over an abstract `ITapeSink` / `ITapeSource`
* `VecTapeSink/Source` for in‑memory tests
* `FileTapeSink/Source` for real disk tapes
* typed records: `u64`, `bytes`, `triple64`, `u64_vec(words)`

It’s designed to be *inserted* under `include/proto/` and used by your current dealer/evaluator code.

```cpp
// include/proto/tape.hpp
#pragma once

#include "proto/common.hpp"

#include <cstddef>
#include <fstream>
#include <span>
#include <string>
#include <vector>

namespace proto {

// Keep tags minimal; your whole system only needs a few.
enum class TapeTag : u32 {
  kU64      = 1,
  kU64Vec   = 2, // payload is 8*words
  kBytes    = 3, // payload arbitrary
  kTriple64 = 4  // payload = 24 bytes (a,b,c shares)
};

struct Triple64Share { u64 a=0, b=0, c=0; };

// ---------------------------------
// IO interfaces
// ---------------------------------
struct ITapeSink {
  virtual ~ITapeSink() = default;
  virtual void write(std::span<const u8> bytes) = 0;
  virtual void flush() {}
};

struct ITapeSource {
  virtual ~ITapeSource() = default;
  virtual void read_exact(std::span<u8> out) = 0;
  virtual bool eof() const = 0;
};

// ---------------------------------
// In-memory tape
// ---------------------------------
class VecTapeSink final : public ITapeSink {
public:
  std::vector<u8> buf;

  void write(std::span<const u8> bytes) override {
    buf.insert(buf.end(), bytes.begin(), bytes.end());
  }
};

class VecTapeSource final : public ITapeSource {
public:
  explicit VecTapeSource(const std::vector<u8>& b) : buf_(b) {}

  void read_exact(std::span<u8> out) override {
    ensure(off_ + out.size() <= buf_.size(), "tape: read past end");
    std::memcpy(out.data(), buf_.data() + off_, out.size());
    off_ += out.size();
  }

  bool eof() const override { return off_ >= buf_.size(); }

  std::size_t offset() const { return off_; }

private:
  const std::vector<u8>& buf_;
  std::size_t off_ = 0;
};

// ---------------------------------
// File tape
// ---------------------------------
class FileTapeSink final : public ITapeSink {
public:
  explicit FileTapeSink(const std::string& path)
      : out_(path, std::ios::binary | std::ios::trunc) {
    ensure(out_.good(), "tape: failed to open output file");
  }

  void write(std::span<const u8> bytes) override {
    out_.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    ensure(out_.good(), "tape: file write failed");
  }

  void flush() override {
    out_.flush();
    ensure(out_.good(), "tape: file flush failed");
  }

private:
  std::ofstream out_;
};

class FileTapeSource final : public ITapeSource {
public:
  explicit FileTapeSource(const std::string& path)
      : in_(path, std::ios::binary) {
    ensure(in_.good(), "tape: failed to open input file");
  }

  void read_exact(std::span<u8> out) override {
    in_.read(reinterpret_cast<char*>(out.data()),
             static_cast<std::streamsize>(out.size()));
    ensure(in_.good(), "tape: file read failed");
  }

  bool eof() const override { return in_.eof(); }

private:
  std::ifstream in_;
};

// ---------------------------------
// TapeWriter / TapeReader
// ---------------------------------
class TapeWriter {
public:
  explicit TapeWriter(ITapeSink& sink) : sink_(sink) {}

  void write_u64(u64 x) {
    u8 hdr[8], payload[8];
    store_u32_le((u32)TapeTag::kU64, hdr);
    store_u32_le(8, hdr + 4);
    store_u64_le(x, payload);
    sink_.write(hdr);
    sink_.write(payload);
  }

  void write_u64_vec(std::span<const u64> v) {
    const u32 bytes = checked_u32_len<std::size_t>(v.size() * 8);
    u8 hdr[8];
    store_u32_le((u32)TapeTag::kU64Vec, hdr);
    store_u32_le(bytes, hdr + 4);
    sink_.write(hdr);

    // Stream payload: no temporary large buffer needed if you want.
    // Here we chunk for safety; customize chunk size if desired.
    constexpr std::size_t kChunkWords = 1024;
    u8 tmp[8 * kChunkWords];
    std::size_t i = 0;
    while (i < v.size()) {
      std::size_t take = (v.size() - i > kChunkWords) ? kChunkWords : (v.size() - i);
      for (std::size_t j = 0; j < take; j++) store_u64_le(v[i + j], tmp + 8 * j);
      sink_.write(std::span<const u8>(tmp, 8 * take));
      i += take;
    }
  }

  void write_bytes(std::span<const u8> b) {
    const u32 bytes = checked_u32_len<std::size_t>(b.size());
    u8 hdr[8];
    store_u32_le((u32)TapeTag::kBytes, hdr);
    store_u32_le(bytes, hdr + 4);
    sink_.write(hdr);
    if (!b.empty()) sink_.write(b);
  }

  void write_triple64(const Triple64Share& t) {
    u8 hdr[8];
    store_u32_le((u32)TapeTag::kTriple64, hdr);
    store_u32_le(24, hdr + 4);
    sink_.write(hdr);

    u8 payload[24];
    store_u64_le(t.a, payload);
    store_u64_le(t.b, payload + 8);
    store_u64_le(t.c, payload + 16);
    sink_.write(payload);
  }

  void flush() { sink_.flush(); }

private:
  ITapeSink& sink_;
};

class TapeReader {
public:
  explicit TapeReader(ITapeSource& src) : src_(src) {}

  // For debugging / assertions.
  TapeTag peek_tag() {
    u8 hdr[8];
    src_.read_exact(hdr);
    // NOTE: this consumes; used only if you implement a buffering source.
    // In practice, prefer not to call peek_tag() unless you wrap a buffering source.
    fail("peek_tag() requires buffering source; do not call in production");
    return TapeTag::kU64;
  }

  u64 read_u64() {
    auto [tag, len] = read_header_();
    ensure(tag == TapeTag::kU64, "tape: expected U64");
    ensure(len == 8, "tape: bad U64 len");
    u8 payload[8];
    src_.read_exact(payload);
    return load_u64_le(payload);
  }

  std::vector<u64> read_u64_vec_words(std::size_t words) {
    auto [tag, len] = read_header_();
    ensure(tag == TapeTag::kU64Vec, "tape: expected U64Vec");
    ensure(len == words * 8, "tape: U64Vec length mismatch");
    std::vector<u64> out(words);
    // Read as bytes then decode (robust / endian stable)
    constexpr std::size_t kChunkWords = 1024;
    u8 tmp[8 * kChunkWords];
    std::size_t i = 0;
    while (i < words) {
      std::size_t take = (words - i > kChunkWords) ? kChunkWords : (words - i);
      src_.read_exact(std::span<u8>(tmp, 8 * take));
      for (std::size_t j = 0; j < take; j++) out[i + j] = load_u64_le(tmp + 8 * j);
      i += take;
    }
    return out;
  }

  std::vector<u8> read_bytes() {
    auto [tag, len] = read_header_();
    ensure(tag == TapeTag::kBytes, "tape: expected Bytes");
    std::vector<u8> out(len);
    if (len) src_.read_exact(out);
    return out;
  }

  Triple64Share read_triple64() {
    auto [tag, len] = read_header_();
    ensure(tag == TapeTag::kTriple64, "tape: expected Triple64");
    ensure(len == 24, "tape: bad Triple64 len");
    u8 payload[24];
    src_.read_exact(payload);
    return Triple64Share{
      load_u64_le(payload),
      load_u64_le(payload + 8),
      load_u64_le(payload + 16),
    };
  }

  bool eof() const { return src_.eof(); }

private:
  struct Header { TapeTag tag; u32 len; };

  Header read_header_() {
    u8 hdr[8];
    src_.read_exact(hdr);
    const u32 tag = load_u32_le(hdr);
    const u32 len = load_u32_le(hdr + 4);
    return Header{ (TapeTag)tag, len };
  }

  ITapeSource& src_;
};

} // namespace proto
```

**Notes**

* This format is aligned with your “bits‑in/bytes‑out” PFSS model: any backend key is serialized as a `Bytes` record.
* It’s deterministic and streamable: you can write two tapes (`P0`, `P1`) and replay them exactly.

---

# 3) Patch‑style outline: ReluARS dealer writes tapes, evaluator consumes tapes

I’ll do this in a **unified diff style** with minimal assumptions:

* You currently have:

    * `include/proto/reluars_dealer.hpp`
    * `include/proto/reluars_online_complete.hpp`
* Your dealer currently outputs per‑party in‑memory key structs (or equivalent).
* Your evaluator currently takes `(party_key, public_hatx, channel, beaver stuff, backend)`.

We’ll add a **tape path** that does *not* break the old path.

---

## 3.1 `reluars_dealer.hpp` patch outline

### Assumptions about your current structs

You likely have something like (names may differ):

* `ReluARSKeyParty` containing:

    * `u64 r_in_share, r_hi_share, r_out_share`
    * DCF keys for predicate bits: `k_w, k_t, k_d` (each is `std::vector<u8>`)
    * Beaver triples: `std::vector<Triple64Share> triples` (or compatible)

If your actual struct names differ, keep the *record order*; mapping is mechanical.

```diff
--- a/include/proto/reluars_dealer.hpp
+++ b/include/proto/reluars_dealer.hpp
@@
 #pragma once
+#include "proto/tape.hpp"
 
 namespace proto {
 
+// IMPORTANT: Fix the exact tape layout so online consumption is deterministic.
+// Record order per *instance*, per party tape:
+//   1) U64  r_in_share
+//   2) U64  r_hi_share        (f=12 truncation helper, per your prototype)
+//   3) U64  r_out_share
+//   4) BYTES dcf_key_w
+//   5) BYTES dcf_key_t
+//   6) BYTES dcf_key_d
+//   7) TRIPLE64 * kNumMulTriples
+//
+// If you later add more predicate DCFs (e.g., low-bit boundaries), append them
+// BEFORE triples, and keep the same order forever.
+
+struct ReluARSTapeLayout {
+  static constexpr std::size_t kNumMulTriples = /* set to your evaluator need */ 0;
+};
+
+// Helper: write one party's ReluARS key material for ONE INSTANCE.
+inline void reluars_write_instance_to_tape(
+    TapeWriter& tw,
+    u64 r_in_share,
+    u64 r_hi_share,
+    u64 r_out_share,
+    const std::vector<u8>& dcf_key_w,
+    const std::vector<u8>& dcf_key_t,
+    const std::vector<u8>& dcf_key_d,
+    const std::vector<Triple64Share>& mul_triples /* size = kNumMulTriples */) {
+  tw.write_u64(r_in_share);
+  tw.write_u64(r_hi_share);
+  tw.write_u64(r_out_share);
+
+  tw.write_bytes(dcf_key_w);
+  tw.write_bytes(dcf_key_t);
+  tw.write_bytes(dcf_key_d);
+
+  ensure(mul_triples.size() == ReluARSTapeLayout::kNumMulTriples,
+         "ReluARS: mul triple count mismatch");
+  for (const auto& t : mul_triples) tw.write_triple64(t);
+}
+
+// New: dealer API that writes both parties' instance keys directly to tapes.
+// This does NOT remove your existing key-returning API.
+template <class Backend, class Rng>
+inline void reluars_dealer_write_batch_to_tapes(
+    Backend& backend,
+    Rng& rng,
+    std::size_t N,
+    TapeWriter& t0,
+    TapeWriter& t1
+    /* plus whatever ReluARS params you already take */) {
+  for (std::size_t i = 0; i < N; i++) {
+    // 1) Produce your existing per-instance key material (in memory).
+    //    e.g.: auto [k0, k1] = reluars_dealer_gen_instance(backend, rng, ...);
+    //
+    // 2) Write to tapes with fixed order:
+    //    reluars_write_instance_to_tape(t0, k0.r_in, k0.r_hi, k0.r_out, k0.k_w, k0.k_t, k0.k_d, k0.triples);
+    //    reluars_write_instance_to_tape(t1, k1.r_in, k1.r_hi, k1.r_out, k1.k_w, k1.k_t, k1.k_d, k1.triples);
+  }
+  t0.flush();
+  t1.flush();
+}
+
 } // namespace proto
```

### What you must concretize

* `ReluARSTapeLayout::kNumMulTriples` **must equal** what your online evaluator actually consumes (the “mul count contract”).
  If your evaluator already has `static constexpr size_t kNumMulTriples`, use that instead:

```cpp
static constexpr std::size_t kNumMulTriples = ReluARSOnlineComplete::kNumMulTriples;
```

That prevents drift.

---

## 3.2 `reluars_online_complete.hpp` patch outline (tape wrapper)

We add a tiny wrapper that:

1. reads exactly one instance from tape in the fixed order,
2. builds an ephemeral key object (or passes raw pieces),
3. calls your existing evaluator path.

```diff
--- a/include/proto/reluars_online_complete.hpp
+++ b/include/proto/reluars_online_complete.hpp
@@
 #pragma once
+#include "proto/tape.hpp"
 
 namespace proto {
 
+// Wrapper: consume one ReluARS instance from tape and evaluate.
+// This ensures online and offline stay in sync without changing the old API.
+template <class Backend, class Channel>
+inline u64 reluars_eval_one_from_tape(
+    int party,
+    Backend& backend,
+    Channel& ch,
+    TapeReader& tr,
+    u64 public_hatx
+    /* plus any runtime params you already take */) {
+
+  // --- 1) Read masked-wire masks (shares) ---
+  const u64 r_in_share  = tr.read_u64();
+  const u64 r_hi_share  = tr.read_u64();
+  const u64 r_out_share = tr.read_u64();
+
+  // Derive x share from (hatx, r_in_share) if your existing evaluator expects x_b.
+  // Protocol 3.4:
+  //   P0: x0 = hatx - r_in0
+  //   P1: x1 = - r_in1
+  const u64 x_share = (party == 0) ? (public_hatx - r_in_share) : (u64)(0 - r_in_share);
+
+  // --- 2) Read PFSS/DCF keys ---
+  const std::vector<u8> k_w = tr.read_bytes();
+  const std::vector<u8> k_t = tr.read_bytes();
+  const std::vector<u8> k_d = tr.read_bytes();
+
+  // --- 3) Read Beaver triples (mul) ---
+  std::vector<Triple64Share> triples;
+  triples.reserve(ReluARSTapeLayout::kNumMulTriples);
+  for (std::size_t i = 0; i < ReluARSTapeLayout::kNumMulTriples; i++) {
+    triples.push_back(tr.read_triple64());
+  }
+
+  // --- 4) Call your existing evaluator logic ---
+  // Option A: if your old API takes a key struct, build it here:
+  //
+  // ReluARSKeyParty kp;
+  // kp.r_in_share = r_in_share; kp.r_hi_share=r_hi_share; kp.r_out_share=r_out_share;
+  // kp.k_w = k_w; kp.k_t = k_t; kp.k_d = k_d;
+  // kp.mul_triples = triples;
+  // return reluars_online_complete_eval(party, backend, ch, kp, public_hatx, x_share, ...);
+  //
+  // Option B: if your old evaluator already takes raw pieces, pass them.
+
+  (void)backend; (void)ch; (void)r_hi_share; // remove if used
+  // TODO: replace this stub call with your real one:
+  // return ReluARSOnlineComplete::eval(party, backend, ch, public_hatx, x_share, r_out_share, k_w, k_t, k_d, triples, ...);
+  fail("reluars_eval_one_from_tape: hook into your real evaluator");
+  return 0;
+}
+
 } // namespace proto
```

### Why this is the safest integration

* You do **not** change your existing evaluator logic.
* You introduce a **single canonical record order**.
* You can immediately test with `backend_clear.hpp` and your sim harness.

---

# 4) Immediate “correctness hardening” you should do in the harness

You explicitly said your harness currently “self-checks reconstructed values” rather than a true plaintext reference.

Before you proceed to Milestone 3/4, do this small change:

* Add a pure plaintext `reluars_ref(x)` using the same `(w,t,d)` definition + truncation + correction LUT.
* Then test:

    * reconstruct output `y = y0 + y1` equals `reluars_ref(x)` (mod 2^64).

This is essential once you plug in **real** DCF backends.

---

# Milestone 3/4 (next): SUF IR stabilization + mask-rewrite engine (§3.3)

You already have `include/suf/*` and `include/compiler/*`. So Milestone 3/4 in your repo is not “make SUF exist”; it’s:

## Milestone 3 — SUF reference semantics + tests (make the IR *provably correct*)

### Deliverables

1. `include/suf/ref_eval.hpp`

* `eval_poly(coeffs, x)` in `Z_2^64`
* `eval_bool_expr(expr, x)` (two’s complement sign, low-bit predicate, comparisons)
* `eval_suf(F, x) -> (arith_outs, bool_outs)`

2. `src/demo/test_suf_ref_eval.cpp`

* For small `n` (like 8 or 12) generate random miniature SUFs and **exhaustively enumerate** all inputs to validate:

    * partition selection
    * polynomial outputs
    * boolean formulas

**Definition of done**

* Exhaustive test passes at `n=8` for random SUFs (hundreds of trials).

---

## Milestone 4 — Mask rewrite (your §3.3) as executable compiler pass

Your compiler already “rotates/splits intervals” for LUTs. Milestone 4 makes predicate correctness **rigorous**.

### Deliverables

1. `include/suf/mask_rewrite.hpp`
   Implement rewrite for each primitive:

* `1[x < beta]` → formula over:

    * `1[hatx < r]`, `1[hatx < r+beta]` (wrap/no-wrap split)
* `1[x mod 2^f < gamma]` → same but in `2^f` domain
* `MSB(x+c)` → `MSB(hatx + (c - r_in))` or reduce to `1[hatx < theta]` form if you prefer

2. `src/demo/test_mask_rewrite.cpp`
   Property test:

* sample random `r_in`, random `x`
* compute `hatx = x + r_in`
* check: `eval(pred(x)) == eval(rewrite(pred,r_in)(hatx))`
  Do this for:
* raw predicates
* composite BoolExpr trees

3. Hook into compiler:

* `compiler/` should call rewrite and then collect the resulting “masked primitives set” to form your `Π_pred` descriptor.

**Definition of done**

* 100k random tests pass for each predicate class and composite expressions.
* The rewritten SUF + interval rotation yields outputs equal to “masked SUF definition” for random `x,r`.

---

## After Milestone 4, you’re ready for “real Sigma-like compilation”

Because at that point you can:

* automatically extract all masked thresholds,
* build packed-compare predicate programs,
* build interval LUT programs,
* and you’re not relying on ad-hoc mask shifts.

---

If you paste (or summarize) the exact fields in your current `ReluARSKeyParty` and the evaluator signature, I can turn the two `TODO` stubs in the patch outline into a **fully concrete diff** that compiles against your repo without renaming.
