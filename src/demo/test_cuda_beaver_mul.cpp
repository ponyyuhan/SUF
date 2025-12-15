#include <vector>
#include <random>
#include <iostream>
#include <cuda_runtime.h>
#include "runtime/cuda_primitives.hpp"

int main() {
#ifndef SUF_HAVE_CUDA
  std::cout << "SUF_HAVE_CUDA not defined; skipping CUDA beaver mul test.\n";
  return 0;
#else
  int dev_count = 0;
  cudaError_t dev_err = cudaGetDeviceCount(&dev_count);
  if (dev_err != cudaSuccess || dev_count <= 0) {
    std::cout << "skip: no CUDA-capable device is detected\n";
    return 0;
  }
  const size_t N = 1024;
  std::mt19937_64 rng(1234);
  std::vector<uint64_t> x(N), y(N), a(N), b(N), c(N), d_open(N), e_open(N);
  for (size_t i = 0; i < N; ++i) {
    x[i] = rng();
    y[i] = rng();
    a[i] = rng();
    b[i] = rng();
    c[i] = rng();
    d_open[i] = x[i] - a[i];
    e_open[i] = y[i] - b[i];
  }
  size_t bytes = N * sizeof(uint64_t);
  uint64_t *d_x, *d_y, *d_a, *d_b, *d_c, *d_d, *d_e, *d_out;
  cudaMalloc(&d_x, bytes);
  cudaMalloc(&d_y, bytes);
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);
  cudaMalloc(&d_d, bytes);
  cudaMalloc(&d_e, bytes);
  cudaMalloc(&d_out, bytes);
  cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_d, d_open.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_e, e_open.data(), bytes, cudaMemcpyHostToDevice);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  launch_beaver_mul_kernel(/*party=*/0, d_x, d_y, d_a, d_b, d_c, d_d, d_e, d_out, N, stream);
  cudaStreamSynchronize(stream);
  std::vector<uint64_t> gpu(N);
  cudaMemcpy(gpu.data(), d_out, bytes, cudaMemcpyDeviceToHost);

  // CPU reference
  bool ok = true;
  for (size_t i = 0; i < N; ++i) {
    unsigned __int128 z = static_cast<unsigned __int128>(c[i]);
    z += static_cast<unsigned __int128>(d_open[i]) * static_cast<unsigned __int128>(b[i]);
    z += static_cast<unsigned __int128>(e_open[i]) * static_cast<unsigned __int128>(a[i]);
    z += static_cast<unsigned __int128>(d_open[i]) * static_cast<unsigned __int128>(e_open[i]);
    uint64_t ref = static_cast<uint64_t>(z);
    if (gpu[i] != ref) {
      std::cerr << "Mismatch at " << i << " got " << gpu[i] << " expected " << ref << "\n";
      ok = false;
      break;
    }
  }

  cudaFree(d_x); cudaFree(d_y); cudaFree(d_a); cudaFree(d_b);
  cudaFree(d_c); cudaFree(d_d); cudaFree(d_e); cudaFree(d_out);
  cudaStreamDestroy(stream);

  if (!ok) return 1;
  std::cout << "CUDA beaver mul test passed\n";
  return 0;
#endif
}
