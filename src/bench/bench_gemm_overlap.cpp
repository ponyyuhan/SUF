#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <random>
#include <vector>
#include <thread>
#include <cstring>

#include "nn/matmul_publicW.hpp"
#include "nn/matmul_gpu.hpp"
#include "proto/backend_factory.hpp"
#include "proto/packed_backend.hpp"
#include "proto/backend_gpu.hpp"

#ifdef SUF_HAVE_CUDA
#include <cuda_runtime.h>
#endif

using namespace nn;

struct BenchTimings {
  float total_ms = 0.0f;
  float pfss_ms = 0.0f;
  float gemm_ms = 0.0f;
};

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
  // Hint the backend to cache keys across calls.
  setenv("SUF_PFSS_CACHE_KEYS", "1", 1);

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
  mp.device_only = true;     // bench device-side overlap
  mp.cache_input = true;     // X is stable across iters
  mp.cache_weights = true;   // W is stable across iters
  mp.cache_bias = true;

  // PFSS workload: one packed compare batch on GPU
  size_t num_thr = 256;
  std::vector<uint64_t> thr(num_thr);
  // Use sorted thresholds to exercise the fast-path (binary-search) packed kernel,
  // which matches how compiled predicate buckets are typically generated.
  const uint64_t thr_mask = (32 == 64) ? ~0ull : ((uint64_t(1) << 32) - 1ull);
  for (size_t i = 0; i < num_thr; ++i) thr[i] = rng() & thr_mask;
  std::sort(thr.begin(), thr.end());
  auto kp = packed->gen_packed_lt(32, thr);
  size_t key_bytes = kp.k0.bytes.size();
  size_t Npfss = 4096;
  std::vector<uint64_t> xs_host(Npfss);
  for (auto& x : xs_host) x = rng();
  int out_words = static_cast<int>((num_thr + 63) / 64);
  // Device buffers for PFSS to avoid host copies per iter.
  uint64_t* xs_dev = nullptr;
  uint64_t* masks_dev = nullptr;
  cudaMalloc(&xs_dev, Npfss * sizeof(uint64_t));
  cudaMalloc(&masks_dev, Npfss * static_cast<size_t>(out_words) * sizeof(uint64_t));
  cudaMemcpy(xs_dev, xs_host.data(), Npfss * sizeof(uint64_t), cudaMemcpyHostToDevice);

  // Create separate stream for GEMM to allow overlap.
  cudaStream_t gemm_stream;
  cudaStreamCreateWithFlags(&gemm_stream, cudaStreamNonBlocking);
  mp.overlap_stream = gemm_stream;

  const int iters = 5;

  auto bench_once = [&](bool run_pfss, bool run_gemm) -> BenchTimings {
    BenchTimings t{};
    std::exception_ptr thr_exc;
    cudaStream_t pfss_stream = nullptr;
    if (auto s = staged->device_stream()) {
      pfss_stream = reinterpret_cast<cudaStream_t>(s);
    }
    if (run_pfss && !pfss_stream) {
      // Ensure the backend has created its streams before we start timing.
      staged->eval_packed_lt_many_device_broadcast(key_bytes,
                                         kp.k0.bytes.data(),
                                         xs_dev,
                                         Npfss,
                                         32,
                                         out_words,
                                         nullptr);
      if (auto s = staged->device_stream()) {
        pfss_stream = reinterpret_cast<cudaStream_t>(s);
      }
    }
    cudaEvent_t start, stop, pfss_start, pfss_stop, gemm_start, gemm_stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&pfss_start);
    cudaEventCreate(&pfss_stop);
    cudaEventCreate(&gemm_start);
    cudaEventCreate(&gemm_stop);
    cudaEventRecord(start, 0);
    std::thread pfss_thr;
    if (run_pfss) {
      pfss_thr = std::thread([&] {
        try {
          cudaSetDevice(0);
          if (pfss_stream) cudaEventRecord(pfss_start, pfss_stream);
          for (int i = 0; i < iters; ++i) {
            staged->eval_packed_lt_many_device_broadcast(key_bytes,
                                               kp.k0.bytes.data(),
                                               xs_dev,
                                               Npfss,
                                               32,
                                               out_words,
                                               nullptr /*device-only*/);
          }
          if (pfss_stream) cudaEventRecord(pfss_stop, pfss_stream);
        } catch (...) {
          thr_exc = std::current_exception();
        }
      });
    }
    if (run_gemm) {
      cudaEventRecord(gemm_start, gemm_stream);
      for (int i = 0; i < iters; ++i) {
        matmul_publicW_gpu(view2(X.data(), batch * M, K),
                           view2(W.data(), K, N),
                           view2(Y.data(), batch * M, N),
                           mp);
      }
      cudaEventRecord(gemm_stop, gemm_stream);
    }
    if (pfss_thr.joinable()) pfss_thr.join();
    if (thr_exc) std::rethrow_exception(thr_exc);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float total = 0.0f;
    cudaEventElapsedTime(&total, start, stop);
    if (run_pfss && pfss_stream) {
      cudaEventSynchronize(pfss_stop);
      cudaEventElapsedTime(&t.pfss_ms, pfss_start, pfss_stop);
      t.pfss_ms /= static_cast<float>(iters);
    }
    if (run_gemm) {
      cudaEventSynchronize(gemm_stop);
      cudaEventElapsedTime(&t.gemm_ms, gemm_start, gemm_stop);
      t.gemm_ms /= static_cast<float>(iters);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(pfss_start);
    cudaEventDestroy(pfss_stop);
    cudaEventDestroy(gemm_start);
    cudaEventDestroy(gemm_stop);
    t.total_ms = total / static_cast<float>(iters);
    return t;
  };

  try {
    // Warm-up
    std::cerr << "Warmup...\n";
    bench_once(true, true);
    auto run_suite = [&](const char* label) {
      std::cerr << "Timing PFSS only (" << label << ")...\n";
      BenchTimings t_pfss = bench_once(true, false);
      std::cerr << "Timing GEMM only (" << label << ")...\n";
      BenchTimings t_gemm = bench_once(false, true);
      std::cerr << "Timing overlap (" << label << ")...\n";
      BenchTimings t_overlap = bench_once(true, true);
      std::cout << "[" << label << "] PFSS_only_ms=" << t_pfss.total_ms
                << " (pfss=" << t_pfss.pfss_ms << ")"
                << " GEMM_only_ms=" << t_gemm.total_ms
                << " (gemm=" << t_gemm.gemm_ms << ")"
                << " overlap_ms=" << t_overlap.total_ms
                << " (pfss=" << t_overlap.pfss_ms
                << ", gemm=" << t_overlap.gemm_ms << ")\n";
    };
    const char* sweep_env = std::getenv("SUF_BENCH_TILE_SWEEP");
    if (sweep_env && std::atoi(sweep_env) != 0) {
      // Sweep both narrow/wide tiles for quick tuning.
      setenv("SUF_MATMUL_GPU_TILE", "narrow", 1);
      run_suite("narrow");
      setenv("SUF_MATMUL_GPU_TILE", "wide", 1);
      run_suite("wide");
    } else {
      run_suite("auto");
    }
    cudaStreamDestroy(gemm_stream);
    cudaFree(xs_dev);
    cudaFree(masks_dev);
  } catch (const std::exception& e) {
    cudaStreamDestroy(gemm_stream);
    cudaFree(xs_dev);
    cudaFree(masks_dev);
    std::cerr << "bench_gemm_overlap failed: " << e.what() << "\n";
    return 1;
  }
  return 0;
#endif
}
