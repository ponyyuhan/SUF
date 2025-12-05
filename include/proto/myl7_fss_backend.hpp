#pragma once

#include "proto/pfss_backend.hpp"
#include "proto/pfss_backend_batch.hpp"
#include "proto/common.hpp"
#include "proto/reference_backend.hpp"
#include <unordered_map>
#include <stdexcept>
#include <cstring>
#include <array>
#include <vector>
#include <random>
#include <mutex>

#if __has_include(<fss/dcf.h>)
  #include <fss/dcf.h>
  #define MYL7_FSS_AVAILABLE 1
#endif

namespace proto {

// Adapter skeleton for myl7/fss (bits-in / bytes-out, 2-party).
class Myl7FssBackend final : public PfssBackendBatch {
public:
  struct Params {
    int lambda_bytes = 16;   // 128-bit security
    bool bits_msb_first = false; // myl7 comparison keys expect LSB-first bits
  };

  Myl7FssBackend() : params_() {}
  explicit Myl7FssBackend(Params p) : params_(p) {}

  std::vector<u8> u64_to_bits_msb(u64 x, int in_bits) const override {
    std::vector<u8> bits(in_bits);
    bool use_msb = params_.bits_msb_first;
    for (int i = 0; i < in_bits; i++) {
      int shift = use_msb ? (in_bits - 1 - i) : i;
      bits[i] = static_cast<u8>((x >> shift) & 1u);
    }
    return bits;
  }

  BitOrder bit_order() const override {
    return params_.bits_msb_first ? BitOrder::MSB_FIRST : BitOrder::LSB_FIRST;
  }

  DcfKeyPair gen_dcf(int in_bits,
                     const std::vector<u8>& alpha_bits,
                     const std::vector<u8>& payload_bytes) override {
    return gen_dcf_impl(in_bits, alpha_bits, payload_bytes);
  }

  std::vector<u8> eval_dcf(int in_bits,
                           const FssKey& kb,
                           const std::vector<u8>& x_bits) const override {
    return eval_dcf_impl(in_bits, kb, x_bits);
  }

  // Pack a bit vector into little-endian bytes according to the configured bit order.
  std::vector<u8> pack_bits_le(const std::vector<u8>& bits, int in_bits) const {
    if (static_cast<int>(bits.size()) != in_bits) throw std::runtime_error("myl7_fss: bitlen mismatch");
    std::vector<u8> bytes(static_cast<size_t>((in_bits + 7) / 8), 0);
    bool msb = params_.bits_msb_first;
    for (int i = 0; i < in_bits; i++) {
      int pos = msb ? (in_bits - 1 - i) : i;
      if (bits[static_cast<size_t>(i)] & 1u) {
        bytes[static_cast<size_t>(pos / 8)] |= static_cast<u8>(1u << (pos % 8));
      }
    }
    return bytes;
  }

  void eval_dcf_many_u64(int in_bits,
                         size_t key_bytes,
                         const uint8_t* keys_flat,
                         const std::vector<u64>& xs_u64,
                         int out_bytes,
                         uint8_t* outs_flat) const override {
    // myl7 backend is already vectorized internally; reuse existing batch API.
    (void)out_bytes;  // payload size is implied by keys in myl7_fss.
    std::vector<FssKey> vec_keys;
    vec_keys.reserve(xs_u64.size());
    for (size_t i = 0; i < xs_u64.size(); ++i) {
      FssKey k;
      k.bytes.assign(keys_flat + i * key_bytes, keys_flat + (i + 1) * key_bytes);
      vec_keys.push_back(std::move(k));
    }
    std::vector<u64> ys(xs_u64.size(), 0);
    for (size_t i = 0; i < xs_u64.size(); ++i) {
      auto bits = u64_to_bits_msb(xs_u64[i], in_bits);
      auto out = eval_dcf(in_bits, vec_keys[i], bits);
      u64 v = 0;
      if (!out.empty()) v = static_cast<u64>(out[0] & 1u);
      ys[i] = v;
    }
    // Payload is one byte per key (XOR bit), stored in outs_flat.
    for (size_t i = 0; i < ys.size(); ++i) {
      outs_flat[i] = static_cast<uint8_t>(ys[i] & 0xffu);
    }
  }

private:
  Params params_;
#if !MYL7_FSS_AVAILABLE
  ReferenceBackend ref_;
#endif

  size_t lambda_bytes() const {
#if MYL7_FSS_AVAILABLE
    return static_cast<size_t>(kLambda);
#else
    return static_cast<size_t>(params_.lambda_bytes);
#endif
  }

#if MYL7_FSS_AVAILABLE
  void ensure_prg_seeded() const {
    static std::once_flag once;
    std::call_once(once, [this]() {
      std::random_device rd;
      // myl7 prg expects seed len >= kBlocks * kLambda bytes.
      constexpr size_t myl7_blocks = 8; // matches local myl7_core build
      std::vector<u8> state(myl7_blocks * lambda_bytes(), 0);
      for (auto& b : state) b = static_cast<u8>(rd());
      prg_init(state.data(), static_cast<int>(state.size()));
    });
  }
#endif

#if MYL7_FSS_AVAILABLE
  // Key format (per-party):
  // [in_bits:u8][num_chunks:u8][party:u8][reserved:u8][payload_len:u32][chunk0][chunk1]...
  // chunk = [seed:lambda][cws:in_bits*kDcfCwLen][cw_np1:lambda]
  DcfKeyPair gen_dcf_impl(int in_bits,
                          const std::vector<u8>& alpha_bits,
                          const std::vector<u8>& payload_bytes) {
    if (in_bits <= 0 || in_bits > 64) throw std::runtime_error("myl7_fss: in_bits out of range");
    if (static_cast<int>(alpha_bits.size()) != in_bits) throw std::runtime_error("myl7_fss: alpha_bits mismatch");
    if (params_.lambda_bytes != static_cast<int>(kLambda)) {
      throw std::runtime_error("myl7_fss: params.lambda_bytes must match kLambda");
    }
    ensure_prg_seeded();
    const size_t lambda = lambda_bytes();

    // Build alpha bytes (little-endian) respecting configured bit order.
    std::vector<u8> alpha_bytes = pack_bits_le(alpha_bits, in_bits);
    Bits alpha_bits_le{alpha_bytes.data(), in_bits};

    const size_t payload_len = payload_bytes.size();
    const size_t num_chunks = (payload_len + lambda - 1) / lambda;
    if (num_chunks > 255) throw std::runtime_error("myl7_fss: payload too large for chunk header");
    if (num_chunks == 0) throw std::runtime_error("myl7_fss: empty payload");
    const size_t cw_bytes = static_cast<size_t>(in_bits) * kDcfCwLen;
    const size_t chunk_key_bytes = lambda + cw_bytes + lambda;

    FssKey fk0, fk1;
    fk0.bytes.resize(8 + num_chunks * chunk_key_bytes);
    fk1.bytes.resize(8 + num_chunks * chunk_key_bytes);
    auto init_header = [&](FssKey& fk, int party) {
      fk.bytes[0] = static_cast<u8>(in_bits);
      fk.bytes[1] = static_cast<u8>(num_chunks);
      fk.bytes[2] = static_cast<u8>(party);
      fk.bytes[3] = 0;
      uint32_t plen = static_cast<uint32_t>(payload_len);
      fk.bytes[4] = static_cast<u8>(plen);
      fk.bytes[5] = static_cast<u8>(plen >> 8);
      fk.bytes[6] = static_cast<u8>(plen >> 16);
      fk.bytes[7] = static_cast<u8>(plen >> 24);
    };
    init_header(fk0, 0);
    init_header(fk1, 1);

    size_t payload_off = 0;
    size_t off0 = 8;
    size_t off1 = 8;
    for (size_t c = 0; c < num_chunks; c++) {
      size_t take = std::min(lambda, payload_len - payload_off);
      std::vector<u8> padded(lambda, 0);
      std::memcpy(padded.data(), payload_bytes.data() + payload_off, take);
      padded[lambda - 1] &= 0x7Fu;  // enforce MSB=0 per group spec

      std::vector<u8> cws(cw_bytes, 0), cw_np1(lambda, 0);
      Key key{cws.data(), cw_np1.data()};
      std::vector<u8> sbuf(10 * lambda, 0);
      std::random_device rd;
      std::vector<u8> seed0(lambda), seed1(lambda);
      for (size_t i = 0; i < lambda; i++) seed0[i] = static_cast<u8>(rd());
      for (size_t i = 0; i < lambda; i++) seed1[i] = static_cast<u8>(rd());
      std::memcpy(sbuf.data(), seed0.data(), lambda);
      std::memcpy(sbuf.data() + lambda, seed1.data(), lambda);

      CmpFunc cf;
      cf.point.alpha = alpha_bits_le;
      cf.point.beta = padded.data();
      cf.bound = kLtAlpha;
      dcf_gen(key, cf, sbuf.data());

      // s0 goes to party0 chunk, s1 to party1 chunk (use initial seeds per upstream tests)
      std::memcpy(fk0.bytes.data() + off0, seed0.data(), lambda);
      std::memcpy(fk1.bytes.data() + off1, seed1.data(), lambda);
      // cws and cw_np1 are shared by both parties
      std::memcpy(fk0.bytes.data() + off0 + lambda, cws.data(), cw_bytes);
      std::memcpy(fk1.bytes.data() + off1 + lambda, cws.data(), cw_bytes);
      std::memcpy(fk0.bytes.data() + off0 + lambda + cw_bytes, cw_np1.data(), lambda);
      std::memcpy(fk1.bytes.data() + off1 + lambda + cw_bytes, cw_np1.data(), lambda);

      payload_off += take;
      off0 += chunk_key_bytes;
      off1 += chunk_key_bytes;
    }

    return DcfKeyPair{fk0, fk1};
  }

  std::vector<u8> eval_dcf_impl(int in_bits,
                                const FssKey& kb,
                                const std::vector<u8>& x_bits) const {
    if (x_bits.size() != static_cast<size_t>(in_bits)) throw std::runtime_error("myl7_fss: x_bits mismatch");
    const auto& bytes = kb.bytes;
    const size_t lambda = lambda_bytes();
    if (bytes.size() < 8) throw std::runtime_error("myl7_fss: key truncated");
    int key_bits = bytes[0];
    if (key_bits != in_bits) throw std::runtime_error("myl7_fss: in_bits mismatch");
    size_t num_chunks = bytes[1];
    int party = bytes[2];
    uint32_t payload_len = static_cast<uint32_t>(bytes[4]) |
                           (static_cast<uint32_t>(bytes[5]) << 8) |
                           (static_cast<uint32_t>(bytes[6]) << 16) |
                           (static_cast<uint32_t>(bytes[7]) << 24);
    if (party != 0 && party != 1) throw std::runtime_error("myl7_fss: bad party flag");
    const size_t cw_bytes = static_cast<size_t>(in_bits) * kDcfCwLen;
    const size_t chunk_key_bytes = lambda + cw_bytes + lambda;
    if (bytes.size() < 8 + num_chunks * chunk_key_bytes) throw std::runtime_error("myl7_fss: key truncated payload");

    std::vector<u8> xbytes = pack_bits_le(x_bits, in_bits);
    Bits x{ xbytes.data(), in_bits };

    std::vector<u8> out(payload_len, 0);
    size_t off = 8;
    size_t payload_off = 0;
    for (size_t c = 0; c < num_chunks; c++) {
      const u8* chunk = bytes.data() + off;
      Key key{const_cast<uint8_t*>(chunk + lambda),
              const_cast<uint8_t*>(chunk + lambda + cw_bytes)};

      std::vector<u8> sbuf(6 * lambda, 0);
      std::memcpy(sbuf.data(), chunk, lambda);
      dcf_eval(sbuf.data(), static_cast<uint8_t>(party), key, x);

      size_t copy = std::min(lambda, out.size() - payload_off);
      std::memcpy(out.data() + payload_off, sbuf.data(), copy);
      payload_off += copy;
      off += chunk_key_bytes;
    }
    return out;
  }
#else
  // Stub fallback delegates to deterministic reference backend (correct semantics).
  DcfKeyPair gen_dcf_impl(int in_bits,
                          const std::vector<u8>& alpha_bits,
                          const std::vector<u8>& payload_bytes) {
    return ref_.gen_dcf(in_bits, alpha_bits, payload_bytes);
  }

  std::vector<u8> eval_dcf_impl(int in_bits,
                                const FssKey& kb,
                                const std::vector<u8>& x_bits) const {
    return ref_.eval_dcf(in_bits, kb, x_bits);
  }
#endif
};

}  // namespace proto
