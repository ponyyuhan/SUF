#pragma once

#include <memory>
#include "proto/backend_clear.hpp"

namespace proto {

// Placeholder GPU backend wrapper. By default we forward to ClearBackend to
// keep tests green; when SUF_HAVE_CUDA is defined we provide a GPU-backed
// PfssBackendBatch (currently a staging stub, replace with real kernels).
std::unique_ptr<PfssBackendBatch> make_gpu_backend_stub();

// Optional staged-eval interface for GPU backends that can consume device
// pointers directly (hatx already staged). Implemented by the CUDA backend.
struct PfssGpuStagedEval {
  virtual ~PfssGpuStagedEval() = default;
  virtual void eval_dcf_many_u64_device(int in_bits,
                                        size_t key_bytes,
                                        const uint8_t* keys_flat,
                                        const uint64_t* xs_device,
                                        size_t N,
                                        int out_bytes,
                                        uint8_t* outs_flat) const = 0;
  virtual void eval_packed_lt_many_device(size_t key_bytes,
                                          const uint8_t* keys_flat,
                                          const uint64_t* xs_device,
                                          size_t N,
                                          int in_bits,
                                          int out_words,
                                          uint64_t* outs_bitmask) const = 0;
  virtual void eval_interval_lut_many_device(size_t key_bytes,
                                             const uint8_t* keys_flat,
                                             const uint64_t* xs_device,
                                             size_t N,
                                             int out_words,
                                             uint64_t* outs_flat) const = 0;
  // Expose underlying compute stream handle for overlap (opaque to callers).
  virtual void* device_stream() const = 0;
};

#ifdef SUF_HAVE_CUDA
std::unique_ptr<PfssBackendBatch> make_real_gpu_backend();
#endif

}  // namespace proto
