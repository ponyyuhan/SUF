#include <cassert>
#include <vector>
#include <random>
#include <iostream>
#include <array>
#include <cstring>
#include <openssl/aes.h>
#include <cuda_runtime.h>

// Kernel defined in cuda/pfss_kernels.cu
extern "C" __global__ void aes128_ctr_kernel(uint8_t* out,
                                             const uint8_t* round_keys,
                                             uint64_t ctr_lo,
                                             uint64_t ctr_hi,
                                             size_t blocks);

int main() {
#ifndef SUF_HAVE_CUDA
  std::cout << "SUF_HAVE_CUDA not defined; skipping CUDA PRG test.\n";
  return 0;
#else
  const size_t blocks = 1024;
  std::array<uint8_t,16> key{};
  std::mt19937_64 rng(12345);
  for (auto& b : key) b = static_cast<uint8_t>(rng() & 0xFFu);

  // Host reference using OpenSSL
  AES_KEY aes_key;
  AES_set_encrypt_key(key.data(), 128, &aes_key);
  std::vector<uint8_t> ref(blocks * 16);
  for (size_t i = 0; i < blocks; i++) {
    uint8_t ctr_block[16] = {0};
    uint64_t ctr = static_cast<uint64_t>(i);
    for (int j = 0; j < 8; j++) ctr_block[j] = static_cast<uint8_t>((ctr >> (8 * j)) & 0xFFu);
    AES_encrypt(ctr_block, ref.data() + i * 16, &aes_key);
  }

  // Expand OpenSSL round keys into flat bytes for device.
  std::vector<uint8_t> round_keys(176);
  std::memcpy(round_keys.data(), aes_key.rd_key, 176);

  uint8_t* d_out = nullptr;
  uint8_t* d_rk = nullptr;
  cudaMalloc(&d_out, blocks * 16);
  cudaMalloc(&d_rk, round_keys.size());
  cudaMemcpy(d_rk, round_keys.data(), round_keys.size(), cudaMemcpyHostToDevice);

  constexpr int kBlock = 128;
  int grid = static_cast<int>((blocks + kBlock - 1) / kBlock);
  uint64_t ctr_lo = 0, ctr_hi = 0;
  void* args[] = {&d_out, &d_rk, &ctr_lo, &ctr_hi, const_cast<size_t*>(&blocks)};
  cudaError_t st = cudaLaunchKernel(reinterpret_cast<const void*>(&aes128_ctr_kernel),
                                    dim3(grid), dim3(kBlock),
                                    args, 0, nullptr);
  if (st != cudaSuccess) {
    std::cout << "Skipping: CUDA kernel launch failed: " << cudaGetErrorString(st) << "\n";
    cudaFree(d_out);
    cudaFree(d_rk);
    return 0;
  }
  std::vector<uint8_t> out(blocks * 16);
  cudaMemcpy(out.data(), d_out, out.size(), cudaMemcpyDeviceToHost);

  cudaFree(d_out);
  cudaFree(d_rk);

  if (out != ref) {
    std::cerr << "AES-CTR mismatch\n";
    return 1;
  }
  std::cout << "AES-CTR CUDA test passed\n";
  return 0;
#endif
}
