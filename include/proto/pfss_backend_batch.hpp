#pragma once

#include "proto/pfss_backend.hpp"
#include <cstring>
#include <stdexcept>

namespace proto {

// Batch-friendly extension; default implementation loops per element.
struct PfssBackendBatch : public PfssBackend {
  virtual void eval_dcf_many_u64(
      int in_bits,
      size_t key_bytes,
      const uint8_t* keys_flat,     // [N][key_bytes]
      const std::vector<u64>& xs_u64,   // [N]
      int out_bytes,
      uint8_t* outs_flat            // [N][out_bytes]
  ) const {
    for (size_t i = 0; i < xs_u64.size(); i++) {
      FssKey kb;
      kb.bytes.assign(keys_flat + i * key_bytes, keys_flat + (i + 1) * key_bytes);
      auto out = eval_dcf(in_bits, kb, u64_to_bits_msb(xs_u64[i], in_bits));
      if (out.size() != static_cast<size_t>(out_bytes)) {
        throw std::runtime_error("eval_dcf_many_u64: output size mismatch");
      }
      std::memcpy(outs_flat + i * out_bytes, out.data(), out_bytes);
    }
  }
};

}  // namespace proto
