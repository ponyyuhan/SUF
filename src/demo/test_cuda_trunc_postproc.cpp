#include <vector>
#include <random>
#include <iostream>
#include <cuda_runtime.h>
#include "runtime/cuda_primitives.hpp"

int main() {
#ifndef SUF_HAVE_CUDA
  std::cout << "SUF_HAVE_CUDA not defined; skipping CUDA trunc test.\n";
  return 0;
#else
  const size_t N = 2048;
  const int frac_bits = 8;
  std::mt19937_64 rng(4321);
  std::vector<uint64_t> x(N);
  for (auto& v : x) v = rng();

  size_t bytes = N * sizeof(uint64_t);
  uint64_t *d_in, *d_out;
  cudaMalloc(&d_in, bytes);
  cudaMalloc(&d_out, bytes);
  cudaMemcpy(d_in, x.data(), bytes, cudaMemcpyHostToDevice);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  // Simple shift kernel
  launch_trunc_shift_kernel(d_in, d_out, frac_bits, N, stream);
  cudaStreamSynchronize(stream);
  std::vector<uint64_t> gpu_shift(N);
  cudaMemcpy(gpu_shift.data(), d_out, bytes, cudaMemcpyDeviceToHost);

  bool ok = true;
  for (size_t i = 0; i < N; ++i) {
    uint64_t ref = (frac_bits >= 64) ? 0ull : (x[i] >> frac_bits);
    if (gpu_shift[i] != ref) {
      std::cerr << "Shift mismatch at " << i << " got " << gpu_shift[i] << " expected " << ref << "\n";
      ok = false;
      break;
    }
  }

  // Faithful trunc postproc kernel (carry/sign zero)
  // Faithful trunc postproc kernel (carry/sign zero, base=0 so output = x>>f)
  uint64_t* d_zero = nullptr;
  cudaMalloc(&d_zero, bytes);
  cudaMemset(d_zero, 0, bytes);
  launch_trunc_postproc_kernel(/*party=*/0,
                               /*kind_gapars=*/0,
                               frac_bits,
                               /*r_hi_share=*/0,
                               /*r_in=*/0,
                               d_in,
                               d_zero,
                               /*arith_stride=*/1,
                               /*arith_idx=*/0,
                               /*d_bools=*/nullptr,
                               /*bool_stride=*/0,
                               /*carry_idx=*/-1,
                               /*sign_idx=*/-1,
                               d_out,
                               N,
                               stream);
  cudaStreamSynchronize(stream);
  std::vector<uint64_t> gpu_post(N);
  cudaMemcpy(gpu_post.data(), d_out, bytes, cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < N && ok; ++i) {
    uint64_t ref = (frac_bits >= 64) ? 0ull : (x[i] >> frac_bits);
    if (gpu_post[i] != ref) {
      std::cerr << "Postproc mismatch at " << i << " got " << gpu_post[i] << " expected " << ref << "\n";
      ok = false;
      break;
    }
  }

  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_zero);
  cudaStreamDestroy(stream);

  if (!ok) return 1;
  std::cout << "CUDA trunc postproc test passed\n";
  return 0;
#endif
}
