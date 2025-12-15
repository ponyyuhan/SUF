#include <vector>
#include <random>
#include <iostream>
#include <cuda_runtime.h>
#include "runtime/cuda_primitives.hpp"

int main() {
#ifndef SUF_HAVE_CUDA
  std::cout << "SUF_HAVE_CUDA not defined; skipping CUDA Horner test.\n";
  return 0;
#else
  int dev_count = 0;
  cudaError_t dev_err = cudaGetDeviceCount(&dev_count);
  if (dev_err != cudaSuccess || dev_count <= 0) {
    std::cout << "skip: no CUDA-capable device is detected\n";
    return 0;
  }
  const size_t N = 1024;
  std::mt19937_64 rng(2024);
  std::vector<uint64_t> x(N), c0(N), c1(N), c2(N), c3(N);
  for (size_t i = 0; i < N; ++i) {
    x[i] = rng();
    c0[i] = rng();
    c1[i] = rng();
    c2[i] = rng();
    c3[i] = rng();
  }
  size_t bytes = N * sizeof(uint64_t);
  uint64_t *d_x, *d_c0, *d_c1, *d_c2, *d_c3, *d_out;
  cudaMalloc(&d_x, bytes);
  cudaMalloc(&d_c0, bytes);
  cudaMalloc(&d_c1, bytes);
  cudaMalloc(&d_c2, bytes);
  cudaMalloc(&d_c3, bytes);
  cudaMalloc(&d_out, bytes);
  cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c0, c0.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c1, c1.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c2, c2.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c3, c3.data(), bytes, cudaMemcpyHostToDevice);

  launch_horner_cubic_kernel(d_x, d_c0, d_c1, d_c2, d_c3, d_out, N, nullptr);
  cudaDeviceSynchronize();
  std::vector<uint64_t> gpu(N);
  cudaMemcpy(gpu.data(), d_out, bytes, cudaMemcpyDeviceToHost);

  bool ok = true;
  for (size_t i = 0; i < N; ++i) {
    unsigned __int128 y = c3[i];
    y = y * x[i] + c2[i];
    y = y * x[i] + c1[i];
    y = y * x[i] + c0[i];
    uint64_t ref = static_cast<uint64_t>(y);
    if (gpu[i] != ref) {
      std::cerr << "Mismatch at " << i << " got " << gpu[i] << " expected " << ref << "\n";
      ok = false;
      break;
    }
  }

  cudaFree(d_x); cudaFree(d_c0); cudaFree(d_c1); cudaFree(d_c2); cudaFree(d_c3); cudaFree(d_out);
  if (!ok) return 1;
  std::cout << "CUDA Horner cubic test passed\n";
  return 0;
#endif
}
