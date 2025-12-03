#pragma once

#include "proto/pfss_backend_batch.hpp"
#include <vector>

namespace proto {

struct IntervalLutKeyPair { FssKey k0, k1; };

struct IntervalLutDesc {
  int in_bits = 64;
  int out_words = 0;                 // number of u64 words per output
  std::vector<u64> cutpoints;        // increasing in masked domain
  std::vector<u64> payload_flat;     // (#intervals * out_words) u64
};

struct PfssIntervalLutExt : public PfssBackendBatch {
  virtual IntervalLutKeyPair gen_interval_lut(const IntervalLutDesc& desc) = 0;

  // Batched evaluation: one key per x, outputs vector payload.
  virtual void eval_interval_lut_many_u64(size_t key_bytes,
                                          const uint8_t* keys_flat,     // [N][key_bytes]
                                          const std::vector<u64>& xs_u64,
                                          int out_words,
                                          u64* outs_flat               // [N][out_words]
                                          ) const = 0;
};

}  // namespace proto
