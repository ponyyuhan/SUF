#include "proto/backend_gpu.hpp"
#include "proto/backend_clear.hpp"

namespace proto {

std::unique_ptr<PfssBackendBatch> make_gpu_backend_stub() {
  return std::unique_ptr<PfssBackendBatch>(new ClearBackend());
}

}  // namespace proto
