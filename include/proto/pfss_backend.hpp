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
};

}  // namespace proto
