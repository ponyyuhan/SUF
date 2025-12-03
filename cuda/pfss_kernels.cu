#include "pfss_cuda_api.hpp"

namespace cuda_pfss {

DeviceKey upload_key(const uint8_t* /*key_bytes*/, size_t /*key_len*/) {
  return {nullptr};
}

void free_key(DeviceKey) {}

void eval_batch_pred(int /*party*/, DeviceKey /*key*/,
                     const uint64_t* /*d_xhat*/, uint64_t* /*d_out*/,
                     size_t /*count*/) {}

void eval_batch_coeff(int /*party*/, DeviceKey /*key*/,
                      const uint64_t* /*d_xhat*/, uint64_t* /*d_out*/,
                      size_t /*count*/) {}

}  // namespace cuda_pfss
