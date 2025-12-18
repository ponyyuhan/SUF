#include "nn/matmul_gpu.hpp"

#ifdef SUF_HAVE_CUDA
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "runtime/bench_online_profile.hpp"

namespace {

// Matmul kernels implement exact mod 2^64 arithmetic using a 32-bit limb
// decomposition (low/high words) to avoid slow 64-bit integer ops in inner loops.

template<int BM, int BN, int BK, int COLS_PER_THREAD>
__global__ void matmul_publicW_tiled(const uint64_t* __restrict__ X,
                                     const int64_t* __restrict__ W,
                                     const int64_t* __restrict__ bias,
                                     uint64_t* __restrict__ Y,
                                     size_t batch,
                                     size_t M,
                                     size_t K,
                                     size_t N,
                                     bool w_transposed) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const size_t block_m = static_cast<size_t>(blockIdx.y) * BM;
  const size_t block_n = static_cast<size_t>(blockIdx.x) * BN;
  const size_t m = block_m + ty;
  const size_t col_base = block_n + static_cast<size_t>(tx * COLS_PER_THREAD);
  const bool active_row = (m < M);

  extern __shared__ uint8_t smem[];
  uint64_t* sX = reinterpret_cast<uint64_t*>(smem);
  int64_t* sW = reinterpret_cast<int64_t*>(sX + BM * BK);

  for (size_t b = 0; b < batch; ++b) {
    uint32_t acc_lo[COLS_PER_THREAD];
    uint32_t acc_hi[COLS_PER_THREAD];
    #pragma unroll
    for (int c = 0; c < COLS_PER_THREAD; ++c) {
      acc_lo[c] = 0;
      acc_hi[c] = 0;
    }
    const uint64_t* Xb = X + b * M * K;
    const size_t tiles = (K + BK - 1) / BK;

    for (size_t tile = 0; tile < tiles; ++tile) {
      const size_t k_base = tile * BK;
      // Load X tile (one row per thread.y, COLS_PER_THREAD columns per thread.x)
      #pragma unroll
      for (int c = 0; c < COLS_PER_THREAD; ++c) {
        const size_t xc = k_base + static_cast<size_t>(tx * COLS_PER_THREAD + c);
        const size_t x_idx = m * K + xc;
        uint64_t v = 0;
        if (active_row && xc < K) v = Xb[x_idx];
        sX[ty * BK + tx * COLS_PER_THREAD + c] = v;
      }
      // Load W tile (cover all BK rows). Each thread.y cooperatively loads rows
      // kk = ty, ty+BM, ty+2BM, ... < BK.
      for (int kk = ty; kk < BK; kk += BM) {
        const int wrow = static_cast<int>(k_base) + kk;
        #pragma unroll
        for (int c = 0; c < COLS_PER_THREAD; ++c) {
          const size_t wc = col_base + static_cast<size_t>(c);
          int64_t wv = 0;
          if (wrow < static_cast<int>(K) && wc < N) {
            const size_t widx = w_transposed ? (wc * K + static_cast<size_t>(wrow))
                                             : (static_cast<size_t>(wrow) * N + wc);
            wv = W[widx];
          }
          sW[kk * BN + tx * COLS_PER_THREAD + c] = wv;
        }
      }
      __syncthreads();

      #pragma unroll
      for (int k = 0; k < BK; ++k) {
        const uint64_t xv = sX[ty * BK + k];
        const uint32_t x0 = static_cast<uint32_t>(xv);
        const uint32_t x1 = static_cast<uint32_t>(xv >> 32);
        #pragma unroll
        for (int c = 0; c < COLS_PER_THREAD; ++c) {
          const uint64_t wv64 = static_cast<uint64_t>(sW[k * BN + tx * COLS_PER_THREAD + c]);
          const uint32_t w0 = static_cast<uint32_t>(wv64);
          const uint32_t w1 = static_cast<uint32_t>(wv64 >> 32);
          const uint32_t p_lo = static_cast<uint32_t>(x0 * w0);
          const uint32_t p_mid = __umulhi(x0, w0);
          const uint32_t cross = static_cast<uint32_t>(x1 * w0 + x0 * w1);
          const uint32_t p_hi = static_cast<uint32_t>(p_mid + cross);
          const uint32_t new_lo = static_cast<uint32_t>(acc_lo[c] + p_lo);
          const uint32_t carry = (new_lo < acc_lo[c]) ? 1u : 0u;
          acc_lo[c] = new_lo;
          acc_hi[c] = static_cast<uint32_t>(acc_hi[c] + p_hi + carry);
        }
      }
      __syncthreads();
    }

    #pragma unroll
    for (int c = 0; c < COLS_PER_THREAD; ++c) {
      const size_t n = col_base + static_cast<size_t>(c);
      if (!active_row || n >= N) continue;
      uint64_t acc64 = static_cast<uint64_t>(acc_lo[c]) | (static_cast<uint64_t>(acc_hi[c]) << 32);
      if (bias) acc64 += static_cast<uint64_t>(bias[n]);
      Y[(b * M + m) * N + n] = acc64;
    }
  }
}

template<int BM, int BN, int BK, int COLS_PER_THREAD>
__global__ void matmul_publicW_tiled_w32(const uint64_t* __restrict__ X,
                                        const int32_t* __restrict__ W,
                                        const int64_t* __restrict__ bias,
                                        uint64_t* __restrict__ Y,
                                        size_t batch,
                                        size_t M,
                                        size_t K,
                                        size_t N,
                                        bool w_transposed) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const size_t block_m = static_cast<size_t>(blockIdx.y) * BM;
  const size_t block_n = static_cast<size_t>(blockIdx.x) * BN;
  const size_t m = block_m + ty;
  const size_t col_base = block_n + static_cast<size_t>(tx * COLS_PER_THREAD);
  const bool active_row = (m < M);

  extern __shared__ uint8_t smem[];
  uint64_t* sX = reinterpret_cast<uint64_t*>(smem);
  int32_t* sW = reinterpret_cast<int32_t*>(sX + BM * BK);

  for (size_t b = 0; b < batch; ++b) {
    uint32_t acc_lo[COLS_PER_THREAD];
    uint32_t acc_hi[COLS_PER_THREAD];
    #pragma unroll
    for (int c = 0; c < COLS_PER_THREAD; ++c) {
      acc_lo[c] = 0;
      acc_hi[c] = 0;
    }
    const uint64_t* Xb = X + b * M * K;
    const size_t tiles = (K + BK - 1) / BK;

    for (size_t tile = 0; tile < tiles; ++tile) {
      const size_t k_base = tile * BK;
      // Load X tile
      #pragma unroll
      for (int c = 0; c < COLS_PER_THREAD; ++c) {
        const size_t xc = k_base + static_cast<size_t>(tx * COLS_PER_THREAD + c);
        const size_t x_idx = m * K + xc;
        uint64_t v = 0;
        if (active_row && xc < K) v = Xb[x_idx];
        sX[ty * BK + tx * COLS_PER_THREAD + c] = v;
      }
      // Load W tile (cover all BK rows)
      for (int kk = ty; kk < BK; kk += BM) {
        const int wrow = static_cast<int>(k_base) + kk;
        #pragma unroll
        for (int c = 0; c < COLS_PER_THREAD; ++c) {
          const size_t wc = col_base + static_cast<size_t>(c);
          int32_t wv = 0;
          if (wrow < static_cast<int>(K) && wc < N) {
            const size_t widx = w_transposed ? (wc * K + static_cast<size_t>(wrow))
                                             : (static_cast<size_t>(wrow) * N + wc);
            wv = W[widx];
          }
          sW[kk * BN + tx * COLS_PER_THREAD + c] = wv;
        }
      }
      __syncthreads();

      #pragma unroll
      for (int k = 0; k < BK; ++k) {
        const uint64_t xv = sX[ty * BK + k];
        const uint32_t x0 = static_cast<uint32_t>(xv);
        const uint32_t x1 = static_cast<uint32_t>(xv >> 32);
        #pragma unroll
        for (int c = 0; c < COLS_PER_THREAD; ++c) {
          const int32_t wv = sW[k * BN + tx * COLS_PER_THREAD + c];
          const uint32_t w0 = static_cast<uint32_t>(wv);
          const uint32_t p_lo = static_cast<uint32_t>(x0 * w0);
          const uint32_t p_mid = __umulhi(x0, w0);
          uint32_t cross = static_cast<uint32_t>(x1 * w0);
          if (wv < 0) cross = static_cast<uint32_t>(cross - x0);
          const uint32_t p_hi = static_cast<uint32_t>(p_mid + cross);
          const uint32_t new_lo = static_cast<uint32_t>(acc_lo[c] + p_lo);
          const uint32_t carry = (new_lo < acc_lo[c]) ? 1u : 0u;
          acc_lo[c] = new_lo;
          acc_hi[c] = static_cast<uint32_t>(acc_hi[c] + p_hi + carry);
        }
      }
      __syncthreads();
    }

    #pragma unroll
    for (int c = 0; c < COLS_PER_THREAD; ++c) {
      const size_t n = col_base + static_cast<size_t>(c);
      if (!active_row || n >= N) continue;
      uint64_t acc64 = static_cast<uint64_t>(acc_lo[c]) | (static_cast<uint64_t>(acc_hi[c]) << 32);
      if (bias) acc64 += static_cast<uint64_t>(bias[n]);
      Y[(b * M + m) * N + n] = acc64;
    }
  }
}

struct MatmulScratch {
  struct HostBufKey {
    const void* host = nullptr;
    size_t bytes = 0;
    bool operator==(const HostBufKey& o) const { return host == o.host && bytes == o.bytes; }
  };
  struct HostBufKeyHash {
    size_t operator()(const HostBufKey& k) const noexcept {
      size_t h = 1469598103934665603ull;
      auto mix = [&](size_t v) {
        h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
      };
      mix(reinterpret_cast<size_t>(k.host));
      mix(static_cast<size_t>(k.bytes));
      return h;
    }
  };
  struct DevBuf {
    void* d = nullptr;
    size_t bytes = 0;
  };

  uint64_t* dX = nullptr;
  int64_t* dW = nullptr;
  int32_t* dW32 = nullptr;
  int64_t* dB = nullptr;
  uint64_t* dY = nullptr;
  size_t x_cap = 0;
  size_t w_cap = 0;
  size_t w32_cap = 0;
  size_t b_cap = 0;
  size_t y_cap = 0;
  const void* last_x_host = nullptr;
  size_t last_x_bytes = 0;
  cudaEvent_t ready = nullptr;  // signals when scratch buffers are safe to reuse
  bool ready_recorded = false;
  std::mutex mu;

  std::unordered_map<HostBufKey, DevBuf, HostBufKeyHash> w_cache;
  std::unordered_map<HostBufKey, DevBuf, HostBufKeyHash> w32_cache;
  std::unordered_map<HostBufKey, DevBuf, HostBufKeyHash> b_cache;
  size_t w_cache_bytes = 0;
  size_t w32_cache_bytes = 0;
  size_t b_cache_bytes = 0;
  std::unordered_map<HostBufKey, bool, HostBufKeyHash> w_fit32;

  bool ensure_alloc(size_t bytes, void** ptr, size_t& cap) {
    if (bytes <= cap) return true;
    if (*ptr) cudaFree(*ptr);
    cudaError_t st = cudaMalloc(ptr, bytes);
    if (st != cudaSuccess) {
      *ptr = nullptr;
      cap = 0;
      return false;
    }
    cap = bytes;
    return true;
  }

  template <typename MapT>
  void clear_cache(MapT& m, size_t& total_bytes) {
    for (auto& kv : m) {
      if (kv.second.d) cudaFree(kv.second.d);
    }
    m.clear();
    total_bytes = 0;
  }

  void clear_all_caches() {
    clear_cache(w_cache, w_cache_bytes);
    clear_cache(w32_cache, w32_cache_bytes);
    clear_cache(b_cache, b_cache_bytes);
  }

  void release() {
    if (dX) cudaFree(dX);
    if (dW) cudaFree(dW);
    if (dW32) cudaFree(dW32);
    if (dB) cudaFree(dB);
    if (dY) cudaFree(dY);
    dX = nullptr; dW = nullptr; dW32 = nullptr; dB = nullptr; dY = nullptr;
    x_cap = w_cap = w32_cap = b_cap = y_cap = 0;
    last_x_host = nullptr;
    last_x_bytes = 0;
    clear_all_caches();
    w_fit32.clear();
    if (ready) {
      cudaEventDestroy(ready);
      ready = nullptr;
    }
    ready_recorded = false;
  }
};

MatmulScratch& scratch() {
  static MatmulScratch s;
  return s;
}

cudaStream_t get_default_stream() {
  static cudaStream_t stream = nullptr;
  static std::once_flag once;
  std::call_once(once, [] {
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  });
  return stream;
}

inline bool check(cudaError_t st) { return st == cudaSuccess; }

template<int BM, int BN, int BK, int COLS_PER_THREAD>
bool launch_matmul_kernel(const uint64_t* dX,
                          const int64_t* dW,
                          const int64_t* dB,
                          uint64_t* dY,
                          size_t batch,
                          size_t M,
                          size_t K,
                          size_t N,
                          bool w_transposed,
                          cudaStream_t stream) {
  static_assert(BN % COLS_PER_THREAD == 0, "BN must be divisible by COLS_PER_THREAD");
  dim3 threads(BN / COLS_PER_THREAD, BM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  size_t smem = sizeof(uint64_t) * BM * BK + sizeof(int64_t) * BK * BN;
  matmul_publicW_tiled<BM, BN, BK, COLS_PER_THREAD><<<grid, threads, smem, stream>>>(
      dX, dW, dB, dY, batch, M, K, N, w_transposed);
  return check(cudaGetLastError());
}

template<int BM, int BN, int BK, int COLS_PER_THREAD>
bool launch_matmul_kernel_w32(const uint64_t* dX,
                              const int32_t* dW,
                              const int64_t* dB,
                              uint64_t* dY,
                              size_t batch,
                              size_t M,
                              size_t K,
                              size_t N,
                              bool w_transposed,
                              cudaStream_t stream) {
  static_assert(BN % COLS_PER_THREAD == 0, "BN must be divisible by COLS_PER_THREAD");
  dim3 threads(BN / COLS_PER_THREAD, BM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  size_t smem = sizeof(uint64_t) * BM * BK + sizeof(int32_t) * BK * BN;
  matmul_publicW_tiled_w32<BM, BN, BK, COLS_PER_THREAD><<<grid, threads, smem, stream>>>(
      dX, dW, dB, dY, batch, M, K, N, w_transposed);
  return check(cudaGetLastError());
}

enum class TileMode { Auto, Narrow, Wide };

TileMode tile_mode_from_env() {
  static TileMode mode = [] {
    const char* env = std::getenv("SUF_MATMUL_GPU_TILE");
    if (!env) return TileMode::Auto;
    if (std::strcmp(env, "wide") == 0 || std::strcmp(env, "WIDE") == 0) return TileMode::Wide;
    if (std::strcmp(env, "narrow") == 0 || std::strcmp(env, "NARROW") == 0) return TileMode::Narrow;
    return TileMode::Auto;
  }();
  return mode;
}

}  // namespace
#endif  // SUF_HAVE_CUDA

namespace nn {

void* matmul_default_stream() {
#ifndef SUF_HAVE_CUDA
  return nullptr;
#else
  return reinterpret_cast<void*>(get_default_stream());
#endif
}

bool matmul_publicW_gpu(const TensorView<uint64_t>& X_share,
                        const TensorView<int64_t>& W_public,
                        TensorView<uint64_t> Y_share,
                        const MatmulParams& params) {
#ifndef SUF_HAVE_CUDA
  (void)X_share;
  (void)W_public;
  (void)Y_share;
  (void)params;
  return false;
#else
  if (params.local_rescale) {
    throw std::runtime_error("matmul_publicW_gpu: local_rescale is unsupported; insert explicit Rescale");
  }
  if (X_share.dims != 2 && X_share.dims != 3) return false;
  size_t batch = (X_share.dims == 2) ? 1 : X_share.shape[0];
  size_t M = (X_share.dims == 2) ? X_share.shape[0] : X_share.shape[1];
  size_t K = (X_share.dims == 2) ? X_share.shape[1] : X_share.shape[2];
  size_t N = params.w_transposed ? W_public.shape[0] : W_public.shape[1];
  size_t total_out = batch * M * N;

  const bool prof = runtime::bench::online_profiling_enabled();
  bool prof_events = false;
  cudaEvent_t ev_start = nullptr;
  cudaEvent_t ev_after_upload = nullptr;
  cudaEvent_t ev_after_kernel = nullptr;
  cudaEvent_t ev_after_download = nullptr;
  auto destroy_events = [&]() {
    if (ev_start) cudaEventDestroy(ev_start);
    if (ev_after_upload) cudaEventDestroy(ev_after_upload);
    if (ev_after_kernel) cudaEventDestroy(ev_after_kernel);
    if (ev_after_download) cudaEventDestroy(ev_after_download);
    ev_start = ev_after_upload = ev_after_kernel = ev_after_download = nullptr;
  };

  cudaStream_t stream = params.overlap_stream ? reinterpret_cast<cudaStream_t>(params.overlap_stream)
                                              : get_default_stream();
  auto& sc = scratch();
  std::unique_lock<std::mutex> lock(sc.mu);

  // Ensure any prior async use of the scratch buffers has completed before
  // reusing/freeing them. Use a lightweight event dependency rather than a
  // global device sync so different streams stay overlap-friendly.
  if (!sc.ready) {
    cudaEventCreateWithFlags(&sc.ready, cudaEventDisableTiming);
  }
  if (sc.ready && sc.ready_recorded) {
    cudaStreamWaitEvent(stream, sc.ready, 0);
  }

  if (prof && !params.device_only) {
    if (cudaEventCreate(&ev_start) == cudaSuccess &&
        cudaEventCreate(&ev_after_upload) == cudaSuccess &&
        cudaEventCreate(&ev_after_kernel) == cudaSuccess &&
        cudaEventCreate(&ev_after_download) == cudaSuccess) {
      prof_events = true;
      cudaEventRecord(ev_start, stream);
    } else {
      destroy_events();
    }
  }

  bool ok = true;
  ok &= sc.ensure_alloc(sizeof(uint64_t) * batch * M * K, reinterpret_cast<void**>(&sc.dX), sc.x_cap);
  ok &= sc.ensure_alloc(sizeof(uint64_t) * total_out, reinterpret_cast<void**>(&sc.dY), sc.y_cap);
  if (!ok) {
    if (prof_events) destroy_events();
    sc.release();
    return false;
  }
  size_t X_bytes = sizeof(uint64_t) * batch * M * K;
  const bool cache_x = params.cache_input || (std::getenv("SUF_MATMUL_GPU_CACHE_X") != nullptr);
  const bool cache_w = params.cache_weights || (std::getenv("SUF_MATMUL_GPU_CACHE_W") != nullptr);
  const bool cache_b = params.cache_bias || (std::getenv("SUF_MATMUL_GPU_CACHE_B") != nullptr);
  if (!cache_x || X_share.data != sc.last_x_host || X_bytes != sc.last_x_bytes) {
    ok &= check(cudaMemcpyAsync(sc.dX, X_share.data, X_bytes, cudaMemcpyHostToDevice, stream));
    sc.last_x_host = X_share.data;
    sc.last_x_bytes = X_bytes;
  }
  const size_t W_bytes = sizeof(int64_t) * W_public.shape[0] * W_public.shape[1];
  const size_t W_elems = W_public.shape[0] * W_public.shape[1];
  const char* env_w32 = std::getenv("SUF_MATMUL_GPU_W32");
  const bool allow_w32 = (!env_w32 || std::strcmp(env_w32, "0") != 0);
  auto cache_max_bytes = [&]() -> size_t {
    const char* env = std::getenv("SUF_MATMUL_GPU_CACHE_MAX_MB");
    if (!env) return size_t(2048ull) * 1024ull * 1024ull;
    char* endp = nullptr;
    unsigned long long mb = std::strtoull(env, &endp, 10);
    if (!endp || endp == env) return size_t(2048ull) * 1024ull * 1024ull;
    return static_cast<size_t>(mb) * 1024ull * 1024ull;
  }();

  auto cached_upload = [&](std::unordered_map<MatmulScratch::HostBufKey, MatmulScratch::DevBuf, MatmulScratch::HostBufKeyHash>& m,
                           size_t& total,
                           const void* host,
                           size_t bytes,
                           cudaStream_t st) -> void* {
    if (bytes == 0 || host == nullptr) return nullptr;
    MatmulScratch::HostBufKey key{host, bytes};
    auto it = m.find(key);
    if (it != m.end()) return it->second.d;
    if (cache_max_bytes > 0 && bytes > cache_max_bytes) return nullptr;
    if (cache_max_bytes > 0 && total + bytes > cache_max_bytes) {
      sc.clear_all_caches();
    }
    void* d = nullptr;
    if (cudaMalloc(&d, bytes) != cudaSuccess) return nullptr;
    if (!check(cudaMemcpyAsync(d, host, bytes, cudaMemcpyHostToDevice, st))) {
      cudaFree(d);
      return nullptr;
    }
    m.emplace(key, MatmulScratch::DevBuf{d, bytes});
    total += bytes;
    return d;
  };

  bool w_fit32 = false;
  if (allow_w32 && W_public.data && W_elems > 0) {
    MatmulScratch::HostBufKey key{W_public.data, W_bytes};
    auto it = sc.w_fit32.find(key);
    if (it != sc.w_fit32.end()) {
      w_fit32 = it->second;
    } else {
      bool ok32 = true;
      for (size_t i = 0; i < W_elems; ++i) {
        int64_t v = W_public.data[i];
        if (v < static_cast<int64_t>(std::numeric_limits<int32_t>::min()) ||
            v > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
          ok32 = false;
          break;
        }
      }
      sc.w_fit32.emplace(key, ok32);
      w_fit32 = ok32;
    }
  }

  const int64_t* dW = nullptr;
  const int32_t* dW32 = nullptr;
  if (w_fit32) {
    auto upload_w32_into = [&](int32_t* d_dst) -> bool {
      if (!d_dst || !W_public.data || W_elems == 0) return false;
      const size_t chunk_elems = 1ull << 20;
      std::vector<int32_t> tmp;
      tmp.reserve(std::min(W_elems, chunk_elems));
      for (size_t off = 0; off < W_elems; off += chunk_elems) {
        size_t n = std::min(chunk_elems, W_elems - off);
        tmp.assign(n, 0);
        for (size_t i = 0; i < n; ++i) tmp[i] = static_cast<int32_t>(W_public.data[off + i]);
        cudaError_t st = cudaMemcpy(reinterpret_cast<uint8_t*>(d_dst) + off * sizeof(int32_t),
                                    tmp.data(),
                                    n * sizeof(int32_t),
                                    cudaMemcpyHostToDevice);
        if (st != cudaSuccess) return false;
      }
      return true;
    };

    const size_t W32_bytes = sizeof(int32_t) * W_elems;
    if (cache_w) {
      MatmulScratch::HostBufKey key{W_public.data, W_bytes};
      auto it = sc.w32_cache.find(key);
      if (it != sc.w32_cache.end()) {
        dW32 = reinterpret_cast<const int32_t*>(it->second.d);
      } else {
        if (cache_max_bytes == 0 || W32_bytes <= cache_max_bytes) {
          if (cache_max_bytes > 0 && sc.w32_cache_bytes + W32_bytes > cache_max_bytes) {
            sc.clear_all_caches();
          }
          void* d = nullptr;
          if (cudaMalloc(&d, W32_bytes) == cudaSuccess) {
            if (upload_w32_into(reinterpret_cast<int32_t*>(d))) {
              sc.w32_cache.emplace(key, MatmulScratch::DevBuf{d, W32_bytes});
              sc.w32_cache_bytes += W32_bytes;
              dW32 = reinterpret_cast<const int32_t*>(d);
            } else {
              cudaFree(d);
            }
          }
        }
      }
    }
    if (!dW32) {
      ok &= sc.ensure_alloc(W32_bytes, reinterpret_cast<void**>(&sc.dW32), sc.w32_cap);
      if (!ok || !sc.dW32 || !upload_w32_into(sc.dW32)) {
        w_fit32 = false;  // fall back below
      } else {
        dW32 = sc.dW32;
      }
    }
  }

  if (!w_fit32) {
    if (cache_w) {
      dW = reinterpret_cast<const int64_t*>(
          cached_upload(sc.w_cache, sc.w_cache_bytes, W_public.data, W_bytes, stream));
    }
    if (!dW) {
      ok &= sc.ensure_alloc(W_bytes, reinterpret_cast<void**>(&sc.dW), sc.w_cap);
      ok &= check(cudaMemcpyAsync(sc.dW, W_public.data, W_bytes, cudaMemcpyHostToDevice, stream));
      dW = sc.dW;
    }
  }

  const int64_t* dB = nullptr;
  if (params.bias) {
    const size_t B_bytes = sizeof(int64_t) * params.bias->size();
    if (cache_b) {
      dB = reinterpret_cast<const int64_t*>(
          cached_upload(sc.b_cache, sc.b_cache_bytes, params.bias->data(), B_bytes, stream));
    }
    if (!dB) {
      ok &= sc.ensure_alloc(B_bytes, reinterpret_cast<void**>(&sc.dB), sc.b_cap);
      ok &= check(cudaMemcpyAsync(sc.dB, params.bias->data(), B_bytes,
                                  cudaMemcpyHostToDevice, stream));
      dB = sc.dB;
    }
  }
  if (!ok) {
    if (prof_events) destroy_events();
    sc.release();
    return false;
  }

  if (prof_events) {
    cudaEventRecord(ev_after_upload, stream);
  }

  int64_t* bias_ptr = params.bias ? const_cast<int64_t*>(dB) : nullptr;
  bool launched = false;
  TileMode mode = tile_mode_from_env();
  const bool prefer_narrow = (N >= 4096 || K >= 1536);
  auto try_wide = [&]() {
    if (w_fit32 && dW32) {
      return launch_matmul_kernel_w32<16, 64, 64, 4>(
          sc.dX, dW32, bias_ptr, sc.dY, batch, M, K, N, params.w_transposed, stream);
    }
    return launch_matmul_kernel<16, 64, 64, 4>(
        sc.dX, dW, bias_ptr, sc.dY, batch, M, K, N, params.w_transposed, stream);
  };
  // Medium BK reduces shared memory; often helps very large K/N.
  auto try_medium = [&]() {
    if (w_fit32 && dW32) {
      return launch_matmul_kernel_w32<16, 64, 32, 4>(
          sc.dX, dW32, bias_ptr, sc.dY, batch, M, K, N, params.w_transposed, stream);
    }
    return launch_matmul_kernel<16, 64, 32, 4>(
        sc.dX, dW, bias_ptr, sc.dY, batch, M, K, N, params.w_transposed, stream);
  };
  auto try_narrow = [&]() {
    if (w_fit32 && dW32) {
      return launch_matmul_kernel_w32<16, 32, 64, 2>(
          sc.dX, dW32, bias_ptr, sc.dY, batch, M, K, N, params.w_transposed, stream);
    }
    return launch_matmul_kernel<16, 32, 64, 2>(
        sc.dX, dW, bias_ptr, sc.dY, batch, M, K, N, params.w_transposed, stream);
  };
  auto try_fallback = [&]() {
    if (w_fit32 && dW32) {
      return launch_matmul_kernel_w32<16, 32, 32, 2>(
          sc.dX, dW32, bias_ptr, sc.dY, batch, M, K, N, params.w_transposed, stream);
    }
    return launch_matmul_kernel<16, 32, 32, 2>(
        sc.dX, dW, bias_ptr, sc.dY, batch, M, K, N, params.w_transposed, stream);
  };

  if (mode == TileMode::Narrow) {
    launched = try_narrow();
    if (!launched) launched = try_fallback();
  } else if (mode == TileMode::Wide) {
    launched = try_wide();
    if (!launched) launched = try_medium();
    if (!launched) launched = try_narrow();
    if (!launched) launched = try_fallback();
  } else {
    // Auto: try an order based on shape, then fall back.
    if (prefer_narrow) {
      launched = try_narrow();
      if (!launched) launched = try_medium();
      if (!launched) launched = try_wide();
    } else {
      launched = try_wide();
      if (!launched) launched = try_medium();
      if (!launched) launched = try_narrow();
    }
    if (!launched) launched = try_fallback();
  }
  if (!launched) {
    if (prof_events) destroy_events();
    sc.release();
    return false;
  }
  if (prof_events) {
    cudaEventRecord(ev_after_kernel, stream);
  }
  if (!params.device_only) {
    ok &= check(cudaMemcpyAsync(Y_share.data, sc.dY, sizeof(uint64_t) * total_out,
                                cudaMemcpyDeviceToHost, stream));
    if (prof_events) {
      cudaEventRecord(ev_after_download, stream);
    }
  }
  if (sc.ready) {
    cudaEventRecord(sc.ready, stream);
    sc.ready_recorded = true;
  }
  if (!params.device_only) {
    ok &= check(cudaStreamSynchronize(stream));
  }
  if (prof_events) {
    float ms_upload = 0.0f;
    float ms_kernel = 0.0f;
    float ms_download = 0.0f;
    float ms_total = 0.0f;
    cudaEventElapsedTime(&ms_upload, ev_start, ev_after_upload);
    cudaEventElapsedTime(&ms_kernel, ev_after_upload, ev_after_kernel);
    cudaEventElapsedTime(&ms_download, ev_after_kernel, ev_after_download);
    cudaEventElapsedTime(&ms_total, ev_start, ev_after_download);
    const uint64_t ns_upload = static_cast<uint64_t>(ms_upload * 1e6);
    const uint64_t ns_kernel = static_cast<uint64_t>(ms_kernel * 1e6);
    const uint64_t ns_download = static_cast<uint64_t>(ms_download * 1e6);
    const uint64_t ns_total = static_cast<uint64_t>(ms_total * 1e6);
    runtime::bench::add_online_ns(runtime::bench::OnlineTimeKind::MatmulUpload, ns_upload);
    runtime::bench::add_online_ns(runtime::bench::OnlineTimeKind::MatmulKernel, ns_kernel);
    runtime::bench::add_online_ns(runtime::bench::OnlineTimeKind::MatmulDownload, ns_download);
    runtime::bench::add_online_ns(runtime::bench::OnlineTimeKind::MatmulTotal, ns_total);
    destroy_events();
  }
  // Keep device buffers for reuse; scratch access is serialized via mutex.
  lock.unlock();
  return ok;
#endif
}

}  // namespace nn
