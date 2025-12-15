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
  int dev_count = 0;
  cudaError_t dev_err = cudaGetDeviceCount(&dev_count);
  if (dev_err != cudaSuccess || dev_count <= 0) {
    std::cout << "skip: no CUDA-capable device is detected\n";
    return 0;
  }
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
                               /*m_share=*/0,
                               d_in,
                               d_zero,
                               /*arith_stride=*/1,
                               /*arith_idx=*/0,
                               /*d_bools=*/nullptr,
                               /*bool_stride=*/0,
                               /*carry_idx=*/-1,
                               /*sign_idx=*/-1,
                               /*wrap_idx=*/-1,
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

  // GapARS postproc kernel sanity: with r_out/base=0 and consistent (r_in, carry),
  // party0 output should equal arithmetic shift of v by frac_bits.
  std::vector<int64_t> v_plain(N, 0);
  std::vector<uint64_t> hatx_gap(N, 0);
  std::vector<uint64_t> carry_gap(N, 0);
  uint64_t r_in = (rng() | (uint64_t(1) << 63));  // force MSB(r_in)=1 so m_share=modulus
  uint64_t r_hi = (frac_bits >= 64) ? 0ull : (r_in >> frac_bits);
  uint64_t r_low = (frac_bits >= 64) ? r_in : (frac_bits <= 0 ? 0ull : (r_in & ((uint64_t(1) << frac_bits) - 1)));
  for (size_t i = 0; i < N; ++i) {
    uint64_t mag = rng() & ((uint64_t(1) << 61) - 1ull);
    int64_t s = static_cast<int64_t>(mag);
    if (rng() & 1) s = -s;
    v_plain[i] = s;
    uint64_t v_ring = static_cast<uint64_t>(s);
    hatx_gap[i] = v_ring + r_in;
    if (frac_bits > 0 && frac_bits < 64) {
      uint64_t x_low = v_ring & ((uint64_t(1) << frac_bits) - 1ull);
      uint64_t sum = x_low + r_low;
      carry_gap[i] = (sum >> frac_bits) & 1ull;
    } else {
      carry_gap[i] = 0;
    }
  }
  cudaMemcpy(d_in, hatx_gap.data(), bytes, cudaMemcpyHostToDevice);
  uint64_t* d_carry = nullptr;
  cudaMalloc(&d_carry, bytes);
  cudaMemcpy(d_carry, carry_gap.data(), bytes, cudaMemcpyHostToDevice);
  uint64_t m_share = (frac_bits <= 0 || frac_bits >= 64) ? 0ull : (uint64_t(1) << (64 - frac_bits));
  launch_trunc_postproc_kernel(/*party=*/0,
                               /*kind_gapars=*/1,
                               frac_bits,
                               /*r_hi_share=*/r_hi,
                               /*m_share=*/m_share,
                               d_in,
                               d_zero,
                               /*arith_stride=*/1,
                               /*arith_idx=*/0,
                               d_carry,
                               /*bool_stride=*/1,
                               /*carry_idx=*/0,
                               /*sign_idx=*/-1,
                               /*wrap_idx=*/-1,
                               d_out,
                               N,
                               stream);
  cudaStreamSynchronize(stream);
  std::vector<uint64_t> gpu_gap(N);
  cudaMemcpy(gpu_gap.data(), d_out, bytes, cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < N && ok; ++i) {
    int64_t ref_s = (frac_bits >= 64) ? 0ll : (v_plain[i] >> frac_bits);
    uint64_t ref = static_cast<uint64_t>(ref_s);
    if (gpu_gap[i] != ref) {
      std::cerr << "GapARS postproc mismatch at " << i
                << " got " << gpu_gap[i] << " expected " << ref
                << " v=" << v_plain[i] << "\n";
      ok = false;
      break;
    }
  }
  cudaFree(d_carry);

  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_zero);
  cudaStreamDestroy(stream);

  if (!ok) return 1;
  std::cout << "CUDA trunc postproc test passed\n";
  return 0;
#endif
}
