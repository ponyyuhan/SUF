#pragma once

#include <cstddef>
#include <cstdint>

namespace cuda_pfss {

// backend-defined opaque device key handle
struct DeviceKey {
  void* ptr;
};

// Upload key bytes -> device resident key
DeviceKey upload_key(const uint8_t* key_bytes, size_t key_len);
void free_key(DeviceKey);

// Evaluate many x_hat in batch. Payload layout is backend-defined.
void eval_batch_pred(int party, DeviceKey key,
                     const uint64_t* d_xhat, uint64_t* d_out,
                     size_t count);

void eval_batch_coeff(int party, DeviceKey key,
                      const uint64_t* d_xhat, uint64_t* d_out,
                      size_t count);

}  // namespace cuda_pfss
