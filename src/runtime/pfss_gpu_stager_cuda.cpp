#include "runtime/pfss_gpu_staging.hpp"

#ifdef SUF_HAVE_CUDA
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace runtime {

namespace {
inline void check(cudaError_t st, const char* what) {
  if (st != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(st));
  }
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
}

CudaPfssStager::~CudaPfssStager() {
  if (own_stream_ && stream_) {
    cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream_));
  }
}

DeviceBufferRef CudaPfssStager::alloc_bytes(size_t bytes) {
  void* ptr = nullptr;
  if (bytes > 0) check(cudaMalloc(&ptr, bytes), "cudaMalloc");
  return DeviceBufferRef{ptr, bytes};
}

void CudaPfssStager::free_bytes(DeviceBufferRef buf) {
  if (buf.ptr) cudaFree(buf.ptr);
}

DeviceBufferRef CudaPfssStager::stage_to_device(const HostBufferRef& host) {
  auto dev = alloc_bytes(host.bytes);
  if (host.ptr && host.bytes) {
    check(cudaMemcpyAsync(dev.ptr, host.ptr, host.bytes, cudaMemcpyHostToDevice,
                          reinterpret_cast<cudaStream_t>(stream_)),
          "cudaMemcpyAsync H2D");
    check(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_)), "stream sync h2d");
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
