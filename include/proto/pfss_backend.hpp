#pragma once

#include "proto/common.hpp"
#include <vector>

namespace proto {

struct FssKey {
  std::vector<u8> bytes;
};

struct DcfKeyPair {
  FssKey k0;
  FssKey k1;
};

// PFSS backend interface for bits-in / bytes-out DCF.
enum class ShareSemantics : uint8_t {
  XorBytes,   // reconstruction via XOR on output bytes
  AddU64      // reconstruction via addition mod 2^64 on u64 words
};

enum class BitOrder : uint8_t { MSB_FIRST, LSB_FIRST };

struct PredKeyMeta {
  BitOrder bit_order = BitOrder::LSB_FIRST;
  ShareSemantics sem = ShareSemantics::XorBytes;
  int out_bytes = 1;  // predicate payload length
};

struct CoeffKeyMeta {
  BitOrder bit_order = BitOrder::LSB_FIRST;
  ShareSemantics sem = ShareSemantics::AddU64;
};

struct PfssBackend {
  virtual ~PfssBackend() = default;

  // Program a DCF for: f(x)=payload if x < alpha else 0.
  virtual DcfKeyPair gen_dcf(int in_bits,
                             const std::vector<u8>& alpha_bits,
                             const std::vector<u8>& payload_bytes) = 0;

  // Evaluate one party's share.
  virtual std::vector<u8> eval_dcf(int in_bits,
                                   const FssKey& kb,
                                   const std::vector<u8>& x_bits) const = 0;

  // Helpers: encode uint64->bits, bits->uint64, etc.
  virtual std::vector<u8> u64_to_bits_msb(u64 x, int in_bits) const = 0;

  // Declare bit ordering preference for alpha/x (defaults to MSB-first).
  virtual BitOrder bit_order() const { return BitOrder::MSB_FIRST; }
};

}  // namespace proto
