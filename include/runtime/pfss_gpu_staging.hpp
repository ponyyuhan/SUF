#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <cstring>

namespace runtime {

// Lightweight descriptors for host/device buffers. In the CPU stub path these
// simply alias host memory; GPU implementations can carry device pointers.
struct DeviceBufferRef {
  void* ptr = nullptr;
  size_t bytes = 0;
};

struct ConstDeviceBufferRef {
  const void* ptr = nullptr;
  size_t bytes = 0;
};

struct HostBufferRef {
  const void* ptr = nullptr;
  size_t bytes = 0;
};

// Abstract staging interface: planners/superbatches can request device storage
// for hatx/payloads/keys and schedule copies. Default stub is CPU passthrough.
class PfssGpuStager {
 public:
  virtual ~PfssGpuStager() = default;

  virtual DeviceBufferRef alloc_bytes(size_t bytes) = 0;
  virtual void free_bytes(DeviceBufferRef buf) = 0;

  virtual DeviceBufferRef stage_to_device(const HostBufferRef& host) = 0;
  virtual void stage_to_host(const ConstDeviceBufferRef& dev, void* host_out, size_t bytes) = 0;

  // Optional: expose underlying stream handle (e.g., for overlap). Default is null.
  virtual void* stream() const { return nullptr; }
};

// CPU fallback: no-op staging, returns host pointers directly.
class CpuPassthroughStager final : public PfssGpuStager {
 public:
  DeviceBufferRef alloc_bytes(size_t bytes) override {
    storage_.resize(bytes);
    return DeviceBufferRef{storage_.data(), bytes};
  }
  void free_bytes(DeviceBufferRef) override {}

  DeviceBufferRef stage_to_device(const HostBufferRef& host) override {
    DeviceBufferRef out = alloc_bytes(host.bytes);
    if (host.ptr && host.bytes) {
      std::memcpy(out.ptr, host.ptr, host.bytes);
    }
    return out;
  }

  void stage_to_host(const ConstDeviceBufferRef& dev, void* host_out, size_t bytes) override {
    if (dev.ptr && host_out && bytes) {
      std::memcpy(host_out, dev.ptr, bytes);
    }
  }

 private:
  std::vector<uint8_t> storage_;
};

#ifdef SUF_HAVE_CUDA
// Simple CUDA stager: allocates device buffers and copies on a dedicated stream.
 class CudaPfssStager final : public PfssGpuStager {
 public:
  // If an existing CUDA stream is supplied (opaque pointer), it will be reused
  // without taking ownership; otherwise a new non-blocking stream is created.
  explicit CudaPfssStager(void* stream = nullptr);
  ~CudaPfssStager() override;

  DeviceBufferRef alloc_bytes(size_t bytes) override;
  void free_bytes(DeviceBufferRef buf) override;
  DeviceBufferRef stage_to_device(const HostBufferRef& host) override;
  void stage_to_host(const ConstDeviceBufferRef& dev, void* host_out, size_t bytes) override;

  void* stream() const override { return stream_; }

 private:
  struct CachedBuf {
    void* ptr = nullptr;
    size_t cap = 0;
    void* ready = nullptr;  // cudaEvent_t (opaque), recorded on stream_ at free time
  };

  struct CachedPinnedHost {
    void* ptr = nullptr;    // cudaMallocHost buffer
    size_t cap = 0;
    void* ready = nullptr;  // cudaEvent_t (opaque), recorded after H2D copy
  };

  void* stream_ = nullptr;
  bool own_stream_ = false;
  bool cache_enabled_ = false;
  size_t max_cached_bytes_ = 0;
  size_t max_single_cached_bytes_ = 0;
  size_t cached_bytes_ = 0;
  std::vector<CachedBuf> cache_;

  // Optional pinned-host staging cache to avoid pageable-host blocking in large H2D copies.
  bool pinned_host_enabled_ = false;
  size_t pinned_host_min_bytes_ = 0;
  size_t max_pinned_host_bytes_ = 0;
  size_t max_single_pinned_host_bytes_ = 0;
  size_t pinned_host_bytes_ = 0;
  std::vector<CachedPinnedHost> pinned_host_cache_;
};
#endif

}  // namespace runtime
