#pragma once

#include "proto/pfss_backend.hpp"
#include <vector>

namespace proto {

struct PackedLtKeyPair { FssKey k0, k1; };

// Optional extension for backends that support packed multi-threshold compares.
class PackedLtBackend {
public:
  virtual ~PackedLtBackend() = default;

  virtual PackedLtKeyPair gen_packed_lt(int in_bits, const std::vector<u64>& thresholds) = 0;

  // Evaluate packed comparison keys for many inputs; outs_bitmask is [N][out_words].
  virtual void eval_packed_lt_many(size_t key_bytes,
                                   const uint8_t* keys_flat,
                                   const std::vector<u64>& xs_u64,
                                   int in_bits,
                                   int out_words,
                                   u64* outs_bitmask) const = 0;
};

}  // namespace proto
