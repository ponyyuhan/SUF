#include "runtime/pfss_gpu_staging.hpp"

#ifdef SUF_HAVE_CUDA
#include <cuda_runtime.h>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace runtime {

namespace {
inline void check(cudaError_t st, const char* what) {
  if (st != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(st));
  }
}

inline size_t env_mb(const char* name, size_t def_mb) {
  const char* v = std::getenv(name);
  if (!v || !*v) return def_mb;
  long long mb = std::atoll(v);
  if (mb <= 0) return 0;
  return static_cast<size_t>(mb);
}
}  // namespace

CudaPfssStager::CudaPfssStager(void* stream) {
  if (stream) {
    stream_ = stream;
    own_stream_ = false;
  } else {
    cudaStream_t s = nullptr;
    check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "cudaStreamCreate");
    stream_ = s;
    own_stream_ = true;
  }

  // Enable a small cache by default when reusing an external stream (stream-ordered
  // pipeline); disable by default for internally-owned streams where callers may
  // use the staged buffers from other streams.
  const bool external = !own_stream_;
  max_cached_bytes_ = env_mb("SUF_CUDA_STAGER_CACHE_MB", external ? 64 : 0) * 1024 * 1024;
  max_single_cached_bytes_ = env_mb("SUF_CUDA_STAGER_CACHE_MAX_SINGLE_MB", 16) * 1024 * 1024;
  cache_enabled_ = (max_cached_bytes_ > 0);
}

CudaPfssStager::~CudaPfssStager() {
  // Best-effort cleanup: ensure pending work on this stream is complete before
  // freeing cached buffers.
  if (stream_) {
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_));
  }
  for (auto& e : cache_) {
    if (e.ready) cudaEventDestroy(reinterpret_cast<cudaEvent_t>(e.ready));
    if (e.ptr) cudaFree(e.ptr);
  }
  cache_.clear();
  cached_bytes_ = 0;
  if (own_stream_ && stream_) {
    cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream_));
  }
}

DeviceBufferRef CudaPfssStager::alloc_bytes(size_t bytes) {
  if (bytes == 0) return DeviceBufferRef{nullptr, 0};
  if (cache_enabled_) {
    size_t best = static_cast<size_t>(-1);
    size_t best_cap = static_cast<size_t>(-1);
    for (size_t i = 0; i < cache_.size(); ++i) {
      auto& e = cache_[i];
      if (!e.ptr || e.cap < bytes) continue;
      if (!e.ready) continue;
      auto st = cudaEventQuery(reinterpret_cast<cudaEvent_t>(e.ready));
      if (st == cudaErrorNotReady) continue;
      check(st, "cudaEventQuery (alloc)");
      if (e.cap < best_cap) {
        best = i;
        best_cap = e.cap;
      }
    }
    if (best != static_cast<size_t>(-1)) {
      CachedBuf e = cache_[best];
      cache_.erase(cache_.begin() + static_cast<std::ptrdiff_t>(best));
      cached_bytes_ -= e.cap;
      cudaEventDestroy(reinterpret_cast<cudaEvent_t>(e.ready));
      e.ready = nullptr;
      return DeviceBufferRef{e.ptr, e.cap};
    }
  }
  void* ptr = nullptr;
  check(cudaMalloc(&ptr, bytes), "cudaMalloc");
  return DeviceBufferRef{ptr, bytes};
}

void CudaPfssStager::free_bytes(DeviceBufferRef buf) {
  if (!buf.ptr) return;
  if (!cache_enabled_ || buf.bytes == 0 || buf.bytes > max_single_cached_bytes_) {
    cudaFree(buf.ptr);
    return;
  }
  if (cached_bytes_ + buf.bytes > max_cached_bytes_) {
    cudaFree(buf.ptr);
    return;
  }
  cudaEvent_t ev = nullptr;
  check(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate");
  check(cudaEventRecord(ev, reinterpret_cast<cudaStream_t>(stream_)), "cudaEventRecord");
  cache_.push_back(CachedBuf{buf.ptr, buf.bytes, reinterpret_cast<void*>(ev)});
  cached_bytes_ += buf.bytes;
}

DeviceBufferRef CudaPfssStager::stage_to_device(const HostBufferRef& host) {
  auto dev = alloc_bytes(host.bytes);
  if (host.ptr && host.bytes) {
    check(cudaMemcpyAsync(dev.ptr, host.ptr, host.bytes, cudaMemcpyHostToDevice,
                          reinterpret_cast<cudaStream_t>(stream_)),
          "cudaMemcpyAsync H2D");
    // If we reuse an externally-owned stream (e.g., PFSS backend compute stream),
    // the caller typically enqueues downstream work on the same stream, so an
    // immediate sync would only add overhead and destroy overlap.
    if (own_stream_) {
      check(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_)), "stream sync h2d");
    }
  }
  return dev;
}

void CudaPfssStager::stage_to_host(const ConstDeviceBufferRef& dev, void* host_out, size_t bytes) {
  if (!dev.ptr || !host_out || bytes == 0) return;
  check(cudaMemcpyAsync(host_out, dev.ptr, bytes, cudaMemcpyDeviceToHost,
                        reinterpret_cast<cudaStream_t>(stream_)),
        "cudaMemcpyAsync D2H");
  check(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_)), "stream sync d2h");
}

}  // namespace runtime

#endif  // SUF_HAVE_CUDA
