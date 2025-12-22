#include "nn/matmul_publicW.hpp"

#include <cassert>
#include <chrono>
#include <cmath>
#include <stdexcept>

#include "proto/common.hpp"
#include "runtime/bench_online_profile.hpp"

namespace nn {

static inline int64_t to_signed(uint64_t v) { return proto::to_signed(v); }
static inline uint64_t to_ring(int64_t v) { return proto::from_signed(v); }

static void matmul2d(const uint64_t* X,
                     const int64_t* W,
                     uint64_t* Y,
                     size_t M,
                     size_t K,
                     size_t N,
                     const MatmulParams& params) {
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      __int128 acc = 0;
      for (size_t k = 0; k < K; ++k) {
        size_t widx = params.w_transposed ? (n * K + k) : (k * N + n);
        __int128 xv = static_cast<__int128>(to_signed(X[m * K + k]));
        __int128 wv = static_cast<__int128>(W[widx]);
        acc += xv * wv;
      }
      if (params.bias && n < params.bias->size()) acc += static_cast<__int128>((*params.bias)[n]);
      Y[m * N + n] = to_ring(static_cast<int64_t>(acc));
    }
  }
}

void matmul_publicW(const TensorView<uint64_t>& X_share,
                    const TensorView<int64_t>& W_public,
                    TensorView<uint64_t> Y_share,
                    const MatmulParams& params) {
  const bool prof = runtime::bench::online_profiling_enabled();
  const auto t0 = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  if (params.local_rescale) {
    throw std::runtime_error("matmul_publicW: local_rescale is unsupported; insert explicit Rescale");
  }
  if (X_share.dims == 2) {
    size_t M = X_share.shape[0];
    size_t K = X_share.shape[1];
    size_t N = params.w_transposed ? W_public.shape[0] : W_public.shape[1];
    matmul2d(X_share.data, W_public.data, Y_share.data, M, K, N, params);
    if (prof) {
      const auto t1 = std::chrono::steady_clock::now();
      runtime::bench::add_online_ns(
          runtime::bench::OnlineTimeKind::MatmulTotal,
          static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()));
    }
    return;
  }
  size_t B = X_share.shape[0];
  size_t M = X_share.shape[1];
  size_t K = X_share.shape[2];
  size_t N = params.w_transposed ? W_public.shape[0] : W_public.shape[1];
  for (size_t b = 0; b < B; ++b) {
    const uint64_t* Xb = X_share.data + b * M * K;
    uint64_t* Yb = Y_share.data + b * M * N;
    matmul2d(Xb, W_public.data, Yb, M, K, N, params);
  }
  if (prof) {
    const auto t1 = std::chrono::steady_clock::now();
    runtime::bench::add_online_ns(
        runtime::bench::OnlineTimeKind::MatmulTotal,
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()));
  }
}

}  // namespace nn
