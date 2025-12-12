#include "nn/matmul_gpu.hpp"

#ifdef SUF_HAVE_CUDA
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>

namespace {

// Split 64-bit multiply into 32-bit halves to keep mod 2^64 semantics while
// mapping to faster 32-bit mul instructions.
__device__ __forceinline__ uint64_t mul_mod64(uint64_t a, uint64_t b) {
  uint64_t alo = static_cast<uint32_t>(a);
  uint64_t ahi = a >> 32;
  uint64_t blo = static_cast<uint32_t>(b);
  uint64_t bhi = b >> 32;
  uint64_t cross = alo * bhi + ahi * blo;
  uint64_t low = alo * blo;
  return low + (cross << 32);
}

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
    uint64_t acc[COLS_PER_THREAD];
    #pragma unroll
    for (int c = 0; c < COLS_PER_THREAD; ++c) acc[c] = 0;
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
      // Load W tile two rows at a time to cover BK rows.
      const int wrow0 = static_cast<int>(k_base) + ty;
      const int wrow1 = wrow0 + BM;
      #pragma unroll
      for (int pass = 0; pass < 2; ++pass) {
        const int wrow = (pass == 0) ? wrow0 : wrow1;
        if (wrow < static_cast<int>(K) && wrow - static_cast<int>(k_base) < BK) {
          #pragma unroll
          for (int c = 0; c < COLS_PER_THREAD; ++c) {
            const size_t wc = col_base + static_cast<size_t>(c);
            int64_t wv = 0;
            if (wc < N) {
              const size_t widx = w_transposed ? (wc * K + static_cast<size_t>(wrow))
                                               : (static_cast<size_t>(wrow) * N + wc);
              wv = W[widx];
            }
            sW[(wrow - static_cast<int>(k_base)) * BN + tx * COLS_PER_THREAD + c] = wv;
          }
        } else if (wrow - static_cast<int>(k_base) < BK) {
          #pragma unroll
          for (int c = 0; c < COLS_PER_THREAD; ++c) {
            sW[(wrow - static_cast<int>(k_base)) * BN + tx * COLS_PER_THREAD + c] = 0;
          }
        }
      }
      __syncthreads();

      #pragma unroll
      for (int k = 0; k < BK; ++k) {
        const uint64_t xv = sX[ty * BK + k];
        #pragma unroll
        for (int c = 0; c < COLS_PER_THREAD; ++c) {
          const uint64_t wv = static_cast<uint64_t>(sW[k * BN + tx * COLS_PER_THREAD + c]);
          acc[c] += mul_mod64(xv, wv);
        }
      }
      __syncthreads();
    }

    #pragma unroll
    for (int c = 0; c < COLS_PER_THREAD; ++c) {
      const size_t n = col_base + static_cast<size_t>(c);
      if (!active_row || n >= N) continue;
      if (bias) acc[c] += static_cast<uint64_t>(bias[n]);
      Y[(b * M + m) * N + n] = acc[c];
    }
  }
}

struct MatmulScratch {
  uint64_t* dX = nullptr;
  int64_t* dW = nullptr;
  int64_t* dB = nullptr;
  uint64_t* dY = nullptr;
  size_t x_cap = 0;
  size_t w_cap = 0;
  size_t b_cap = 0;
  size_t y_cap = 0;
  const void* last_x_host = nullptr;
  size_t last_x_bytes = 0;
  const void* last_w_host = nullptr;
  size_t last_w_bytes = 0;
  const void* last_b_host = nullptr;
  size_t last_b_bytes = 0;
  cudaEvent_t ready = nullptr;  // signals when scratch buffers are safe to reuse
  bool ready_recorded = false;
  std::mutex mu;

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

  void release() {
    if (dX) cudaFree(dX);
    if (dW) cudaFree(dW);
    if (dB) cudaFree(dB);
    if (dY) cudaFree(dY);
    dX = nullptr; dW = nullptr; dB = nullptr; dY = nullptr;
    x_cap = w_cap = b_cap = y_cap = 0;
    last_x_host = last_w_host = last_b_host = nullptr;
    last_x_bytes = last_w_bytes = last_b_bytes = 0;
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

  bool ok = true;
  ok &= sc.ensure_alloc(sizeof(uint64_t) * batch * M * K, reinterpret_cast<void**>(&sc.dX), sc.x_cap);
  ok &= sc.ensure_alloc(sizeof(int64_t) * W_public.shape[0] * W_public.shape[1], reinterpret_cast<void**>(&sc.dW), sc.w_cap);
  if (params.bias) {
    ok &= sc.ensure_alloc(sizeof(int64_t) * params.bias->size(), reinterpret_cast<void**>(&sc.dB), sc.b_cap);
  }
  ok &= sc.ensure_alloc(sizeof(uint64_t) * total_out, reinterpret_cast<void**>(&sc.dY), sc.y_cap);
  if (!ok) {
    sc.release();
    return false;
  }
  size_t X_bytes = sizeof(uint64_t) * batch * M * K;
  size_t W_bytes = sizeof(int64_t) * W_public.shape[0] * W_public.shape[1];
  const bool cache_x = params.cache_input || (std::getenv("SUF_MATMUL_GPU_CACHE_X") != nullptr);
  const bool cache_w = params.cache_weights || (std::getenv("SUF_MATMUL_GPU_CACHE_W") != nullptr);
  const bool cache_b = params.cache_bias || (std::getenv("SUF_MATMUL_GPU_CACHE_B") != nullptr);
  if (!cache_x || X_share.data != sc.last_x_host || X_bytes != sc.last_x_bytes) {
    ok &= check(cudaMemcpyAsync(sc.dX, X_share.data, X_bytes, cudaMemcpyHostToDevice, stream));
    sc.last_x_host = X_share.data;
    sc.last_x_bytes = X_bytes;
  }
  if (!cache_w || W_public.data != sc.last_w_host || W_bytes != sc.last_w_bytes) {
    ok &= check(cudaMemcpyAsync(sc.dW, W_public.data, W_bytes, cudaMemcpyHostToDevice, stream));
    sc.last_w_host = W_public.data;
    sc.last_w_bytes = W_bytes;
  }
  if (params.bias) {
    size_t B_bytes = sizeof(int64_t) * params.bias->size();
    if (!cache_b || params.bias->data() != sc.last_b_host || B_bytes != sc.last_b_bytes) {
      ok &= check(cudaMemcpyAsync(sc.dB, params.bias->data(), B_bytes,
                                  cudaMemcpyHostToDevice, stream));
      sc.last_b_host = params.bias->data();
      sc.last_b_bytes = B_bytes;
    }
  } else {
    sc.dB = nullptr;
    sc.last_b_host = nullptr;
    sc.last_b_bytes = 0;
  }
  if (!ok) {
    sc.release();
    return false;
  }
  int64_t* bias_ptr = params.bias ? sc.dB : nullptr;
  bool launched = false;
  TileMode mode = tile_mode_from_env();
  // Heuristic: prefer wider tiles for large N/K, allow env override; fall back to narrower.
  if ((mode == TileMode::Wide) || (mode == TileMode::Auto && N >= 128 && K >= 128)) {
    launched = launch_matmul_kernel<16, 64, 64, 4>(
        sc.dX, sc.dW, bias_ptr, sc.dY, batch, M, K, N, params.w_transposed, stream);
  }
  if (!launched) {
    launched = launch_matmul_kernel<16, 32, 64, 2>(
        sc.dX, sc.dW, bias_ptr, sc.dY, batch, M, K, N, params.w_transposed, stream);
  }
  if (!launched) {
    launched = launch_matmul_kernel<16, 32, 32, 2>(
        sc.dX, sc.dW, bias_ptr, sc.dY, batch, M, K, N, params.w_transposed, stream);
  }
  if (!launched) {
    sc.release();
    return false;
  }
  if (!params.device_only) {
    ok &= check(cudaMemcpyAsync(Y_share.data, sc.dY, sizeof(uint64_t) * total_out,
                                cudaMemcpyDeviceToHost, stream));
  }
  if (sc.ready) {
    cudaEventRecord(sc.ready, stream);
    sc.ready_recorded = true;
  }
  if (!params.device_only) {
    ok &= check(cudaStreamSynchronize(stream));
  }
  // Keep device buffers for reuse; scratch access is serialized via mutex.
  lock.unlock();
  return ok;
#endif
}

}  // namespace nn
