#pragma once

#include <memory>
#include "proto/backend_clear.hpp"

namespace proto {

// Placeholder GPU backend wrapper. By default we forward to ClearBackend to
// keep tests green; when SUF_HAVE_CUDA is defined we provide a GPU-backed
// PfssBackendBatch (currently a staging stub, replace with real kernels).
std::unique_ptr<PfssBackendBatch> make_gpu_backend_stub();
#ifdef SUF_HAVE_CUDA
std::unique_ptr<PfssBackendBatch> make_real_gpu_backend();
#endif

}  // namespace proto
