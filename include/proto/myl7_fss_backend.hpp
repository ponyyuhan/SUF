#pragma once

#include "proto/pfss_backend.hpp"
#include "proto/common.hpp"
#include <unordered_map>
#include <stdexcept>
#include <array>

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

  std::vector<u8> eval_dcf(int in_bits,
                           const FssKey& kb,
                           const std::vector<u8>& x_bits) const override {
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

private:
  Params params_;

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
};

}  // namespace proto
