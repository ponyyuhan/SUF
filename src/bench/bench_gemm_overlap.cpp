#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>
#include <thread>

#include "nn/matmul_publicW.hpp"
#include "nn/matmul_gpu.hpp"
#include "proto/backend_factory.hpp"
#include "proto/packed_backend.hpp"
#include "proto/backend_gpu.hpp"

#ifdef SUF_HAVE_CUDA
#include <cuda_runtime.h>
#endif

using namespace nn;

int main() {
#ifndef SUF_HAVE_CUDA
  std::cout << "SUF_HAVE_CUDA not defined; skipping overlap benchmark.\n";
  return 0;
#else
  std::cout << "Starting GEMM/PFSS overlap bench...\n";
  int devs = 0;
  if (cudaGetDeviceCount(&devs) != cudaSuccess || devs == 0) {
    std::cout << "No CUDA device; skipping overlap benchmark.\n";
    return 0;
  }
  std::cout << "CUDA devices found: " << devs << "\n";
  cudaError_t set_dev = cudaSetDevice(0);
  if (set_dev != cudaSuccess) {
    std::cout << "cudaSetDevice failed: " << cudaGetErrorString(set_dev) << "\n";
    return 0;
  }
  auto backend = proto::make_real_gpu_backend();
  auto* packed = dynamic_cast<proto::PackedLtBackend*>(backend.get());
  auto* staged = dynamic_cast<proto::PfssGpuStagedEval*>(backend.get());
  if (!packed || !staged) {
    std::cout << "GPU backend missing packed/staged eval; skipping.\n";
    return 0;
  }

  // Matmul dims
  size_t batch = 1, M = 256, K = 256, N = 256;
  size_t total_out = batch * M * N;
  std::mt19937_64 rng(123);
  std::vector<uint64_t> X(batch * M * K);
  std::vector<int64_t> W(K * N);
  for (auto& v : X) v = rng();
  for (auto& w : W) w = static_cast<int64_t>(rng() % 16);
  std::vector<uint64_t> Y(total_out, 0);

  MatmulParams mp;
  mp.frac_bits = 0;
  mp.w_transposed = false;
  mp.local_rescale = false;
  mp.allow_legacy_shift = false;

  // PFSS workload: one packed compare batch on GPU
  size_t num_thr = 256;
  std::vector<uint64_t> thr(num_thr);
  for (size_t i = 0; i < num_thr; ++i) thr[i] = rng();
  auto kp = packed->gen_packed_lt(32, thr);
  size_t key_bytes = kp.k0.bytes.size();
  size_t Npfss = 4096;
  std::vector<uint8_t> keys_flat(Npfss * key_bytes);
  for (size_t i = 0; i < Npfss; ++i) {
    std::memcpy(keys_flat.data() + i * key_bytes, kp.k0.bytes.data(), key_bytes);
  }
  std::vector<uint64_t> xs(Npfss);
  for (auto& x : xs) x = rng();
  int out_words = static_cast<int>((num_thr + 63) / 64);
  std::vector<uint64_t> masks(Npfss * static_cast<size_t>(out_words), 0);

  // Create separate stream for GEMM to allow overlap.
  cudaStream_t gemm_stream;
  cudaStreamCreateWithFlags(&gemm_stream, cudaStreamNonBlocking);
  mp.overlap_stream = gemm_stream;

  const int iters = 5;

  auto bench_once = [&](bool run_pfss, bool run_gemm) -> float {
    std::exception_ptr thr_exc;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    std::thread pfss_thr;
    if (run_pfss) {
      pfss_thr = std::thread([&] {
        try {
          cudaSetDevice(0);
          for (int i = 0; i < iters; ++i) {
            packed->eval_packed_lt_many(key_bytes, keys_flat.data(), xs, 32, out_words, masks.data());
          }
        } catch (...) {
          thr_exc = std::current_exception();
        }
      });
    }
    if (run_gemm) {
      for (int i = 0; i < iters; ++i) {
        matmul_publicW_gpu(view2(X.data(), batch * M, K),
                           view2(W.data(), K, N),
                           view2(Y.data(), batch * M, N),
                           mp);
      }
    }
    if (pfss_thr.joinable()) pfss_thr.join();
    if (thr_exc) std::rethrow_exception(thr_exc);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / static_cast<float>(iters);
  };

  try {
    // Warm-up
    std::cerr << "Warmup...\n";
    bench_once(true, true);
    std::cerr << "Timing PFSS only...\n";
    float t_pfss = bench_once(true, false);
    std::cerr << "Timing GEMM only...\n";
    float t_gemm = bench_once(false, true);
    std::cerr << "Timing overlap...\n";
    float t_overlap = bench_once(true, true);
    cudaStreamDestroy(gemm_stream);

    std::cout << "PFSS_only_ms=" << t_pfss
              << " GEMM_only_ms=" << t_gemm
              << " overlap_ms=" << t_overlap << "\n";
  } catch (const std::exception& e) {
    cudaStreamDestroy(gemm_stream);
    std::cerr << "bench_gemm_overlap failed: " << e.what() << "\n";
    return 1;
  }
  return 0;
#endif
}
