#pragma once

#include "proto/pfss_backend.hpp"
#include "proto/common.hpp"
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
class Myl7FssBackend final : public PfssBackend {
public:
  struct Params {
    int lambda_bytes = 16;   // 128-bit security
    bool bits_msb_first = true;
  };

  Myl7FssBackend() : params_() {}
  explicit Myl7FssBackend(Params p) : params_(p) {}

  std::vector<u8> u64_to_bits_msb(u64 x, int in_bits) const override {
    std::vector<u8> bits(in_bits);
    for (int i = 0; i < in_bits; i++) {
      int shift = (in_bits - 1 - i);
      bits[i] = static_cast<u8>((x >> shift) & 1u);
    }
    return bits;
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

private:
  Params params_;

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
      std::vector<u8> state(4 * lambda_bytes(), 0);
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

    // Build alpha bytes (little-endian bits)
    u64 alpha_val = 0;
    for (int i = 0; i < in_bits; i++) {
      alpha_val = (alpha_val << 1) | (alpha_bits[static_cast<size_t>(i)] & 1u);
    }
    std::vector<u8> alpha_bytes((in_bits + 7) / 8, 0);
    for (int i = 0; i < in_bits; i++) {
      if ((alpha_val >> i) & 1ull) {
        alpha_bytes[static_cast<size_t>(i / 8)] |= u8(1u << (i % 8));
      }
    }
    Bits alpha_bits_le{alpha_bytes.data(), in_bits};

    const size_t payload_len = payload_bytes.size();
    const size_t num_chunks = (payload_len + lambda - 1) / lambda;
    if (num_chunks > 255) throw std::runtime_error("myl7_fss: payload too large for chunk header");
    if (num_chunks == 0) throw std::runtime_error("myl7_fss: empty payload");
    const size_t cw_bytes = static_cast<size_t>(in_bits) * kDcfCwLen;
    const size_t chunk_key_bytes = lambda + cw_bytes + lambda;

    auto pack_party = [&](int party) {
      if (party != 0 && party != 1) throw std::runtime_error("myl7_fss: bad party");
      FssKey fk;
      fk.bytes.resize(8 + num_chunks * chunk_key_bytes);
      fk.bytes[0] = static_cast<u8>(in_bits);
      fk.bytes[1] = static_cast<u8>(num_chunks);
      fk.bytes[2] = static_cast<u8>(party);
      fk.bytes[3] = 0;
      uint32_t plen = static_cast<uint32_t>(payload_len);
      fk.bytes[4] = static_cast<u8>(plen);
      fk.bytes[5] = static_cast<u8>(plen >> 8);
      fk.bytes[6] = static_cast<u8>(plen >> 16);
      fk.bytes[7] = static_cast<u8>(plen >> 24);

      size_t payload_off = 0;
      size_t off = 8;
      for (size_t c = 0; c < num_chunks; c++) {
        size_t take = std::min(lambda, payload_len - payload_off);
        std::vector<u8> padded(lambda, 0);
        std::memcpy(padded.data(), payload_bytes.data() + payload_off, take);
        padded[lambda - 1] &= 0x7Fu;  // enforce MSB=0 per group spec

        std::vector<u8> cws(cw_bytes, 0), cw_np1(lambda, 0);
        Key key{cws.data(), cw_np1.data()};
        std::vector<u8> sbuf(10 * lambda, 0);
        std::random_device rd;
        for (size_t i = 0; i < 2 * lambda; i++) sbuf[i] = static_cast<u8>(rd());

        CmpFunc cf;
        cf.point.alpha = alpha_bits_le;
        cf.point.beta = padded.data();
        cf.bound = kLtAlpha;
        dcf_gen(key, cf, sbuf.data());

        std::memcpy(fk.bytes.data() + off, sbuf.data() + (party == 0 ? 0 : lambda), lambda);
        std::memcpy(fk.bytes.data() + off + lambda, cws.data(), cw_bytes);
        std::memcpy(fk.bytes.data() + off + lambda + cw_bytes, cw_np1.data(), lambda);

        payload_off += take;
        off += chunk_key_bytes;
      }
      return fk;
    };

    return DcfKeyPair{pack_party(0), pack_party(1)};
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

    std::vector<u8> xbytes((in_bits + 7) / 8, 0);
    for (int i = 0; i < in_bits; i++) {
      if (x_bits[static_cast<size_t>(in_bits - 1 - i)] & 1u) {
        xbytes[static_cast<size_t>(i / 8)] |= u8(1u << (i % 8));
      }
    }
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
  // Stub fallback if fss headers are missing.
  DcfKeyPair gen_dcf_impl(int in_bits,
                          const std::vector<u8>& alpha_bits,
                          const std::vector<u8>& payload_bytes) {
    Program p;
    p.in_bits = in_bits;
    p.alpha_bits = alpha_bits;
    p.payload[0] = payload_bytes;
    p.payload[1] = std::vector<u8>(payload_bytes.size(), 0u);
    u64 id = next_id_++;
    programs_[id] = std::move(p);
    DcfKeyPair kp;
    u64 kid0 = (id << 1);
    u64 kid1 = (id << 1) | 1ull;
    kp.k0.bytes = pack_u64_le(kid0);
    kp.k1.bytes = pack_u64_le(kid1);
    return kp;
  }

  std::vector<u8> eval_dcf_impl(int in_bits,
                                const FssKey& kb,
                                const std::vector<u8>& x_bits) const {
    if (x_bits.size() != static_cast<size_t>(in_bits)) {
      throw std::runtime_error("eval_dcf: x_bits size mismatch");
    }
    auto [id, party] = decode_key(kb);
    auto it = programs_.find(id);
    if (it == programs_.end()) throw std::runtime_error("eval_dcf: unknown key id");
    const auto& p = it->second;
    if (p.in_bits != in_bits) throw std::runtime_error("eval_dcf: in_bits mismatch");

    bool lt = false;
    for (int i = 0; i < in_bits; i++) {
      u8 xb = x_bits[static_cast<size_t>(i)] & 1u;
      u8 ab = p.alpha_bits[static_cast<size_t>(i)] & 1u;
      if (xb < ab) { lt = true; break; }
      if (xb > ab) { lt = false; break; }
    }
    if (lt) return p.payload[static_cast<size_t>(party)];
    return std::vector<u8>(p.payload[static_cast<size_t>(party)].size(), 0u);
  }

  struct Program {
    int in_bits;
    std::vector<u8> alpha_bits;
    std::array<std::vector<u8>, 2> payload;
  };

  mutable std::unordered_map<u64, Program> programs_;
  u64 next_id_ = 1;

  std::pair<u64, int> decode_key(const FssKey& kb) const {
    if (kb.bytes.size() < 8) throw std::runtime_error("eval_dcf: key truncated");
    u64 kid = unpack_u64_le(kb.bytes.data());
    u64 id = kid >> 1;
    int party = static_cast<int>(kid & 1ull);
    return {id, party};
  }
#endif
};

}  // namespace proto
