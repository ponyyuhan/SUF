#pragma once

#include "proto/pfss_backend.hpp"
#include "proto/common.hpp"
#include <unordered_map>
#include <stdexcept>

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
    p.payload = payload_bytes;
    u64 id = next_id_++;
    programs_[id] = std::move(p);
    DcfKeyPair kp;
    kp.k0.bytes = pack_u64_le(id);
    kp.k1.bytes = pack_u64_le(id);
    return kp;
  }

  std::vector<u8> eval_dcf(int in_bits,
                           const FssKey& kb,
                           const std::vector<u8>& x_bits) const override {
    if (x_bits.size() != static_cast<size_t>(in_bits)) {
      throw std::runtime_error("eval_dcf: x_bits size mismatch");
    }
    if (kb.bytes.size() < 8) throw std::runtime_error("eval_dcf: key truncated");
    u64 id = unpack_u64_le(kb.bytes.data());
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
    // payload if x<alpha else 0
    if (lt) return p.payload;
    return std::vector<u8>(p.payload.size(), 0u);
  }

private:
  Params params_;

  struct Program {
    int in_bits;
    std::vector<u8> alpha_bits;
    std::vector<u8> payload;
  };

  mutable std::unordered_map<u64, Program> programs_;
  u64 next_id_ = 1;
};

}  // namespace proto
