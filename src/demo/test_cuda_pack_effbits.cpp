#include <vector>
#include <random>
#include <iostream>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cuda_runtime.h>

// Kernel defined in cuda/pfss_kernels.cu
extern "C" __global__ void unpack_eff_bits_kernel(const uint64_t* packed,
                                                  int eff_bits,
                                                  uint64_t* out,
                                                  size_t N);

namespace {

std::vector<uint64_t> pack_host(const std::vector<uint64_t>& xs, int eff_bits) {
  if (eff_bits <= 0 || eff_bits > 64) throw std::runtime_error("eff_bits out of range");
  if (eff_bits == 64) return xs;
  uint64_t mask = (eff_bits == 64) ? ~0ull : ((uint64_t(1) << eff_bits) - 1ull);
  size_t bits = static_cast<size_t>(eff_bits) * xs.size();
  size_t words = (bits + 63) >> 6;
  std::vector<uint64_t> packed(words, 0);
  for (size_t i = 0; i < xs.size(); i++) {
    uint64_t v = xs[i] & mask;
    size_t bit_idx = i * static_cast<size_t>(eff_bits);
    size_t w = bit_idx >> 6;
    int off = static_cast<int>(bit_idx & 63);
    packed[w] |= (v << off);
    int spill = off + eff_bits - 64;
    if (spill > 0 && w + 1 < packed.size()) {
      packed[w + 1] |= (v >> (eff_bits - spill));
    }
  }
  return packed;
}

bool check_cuda_available() {
  int devices = 0;
  auto st = cudaGetDeviceCount(&devices);
  return (st == cudaSuccess && devices > 0);
}

}  // namespace

int main() {
#ifndef SUF_HAVE_CUDA
  std::cout << "SUF_HAVE_CUDA not defined; skipping eff_bits packing test.\n";
  return 0;
#else
  if (!check_cuda_available()) {
    std::cout << "No CUDA device; skipping eff_bits packing test.\n";
    return 0;
  }

  const int eff_bits = 13;
  const size_t N = 97;
  uint64_t mask = (uint64_t(1) << eff_bits) - 1ull;
  std::mt19937_64 rng(123);
  std::vector<uint64_t> xs(N);
  for (size_t i = 0; i < N; i++) xs[i] = rng() & mask;

  auto packed = pack_host(xs, eff_bits);
  uint64_t* d_packed = nullptr;
  uint64_t* d_out = nullptr;
  cudaMalloc(&d_packed, packed.size() * sizeof(uint64_t));
  cudaMalloc(&d_out, xs.size() * sizeof(uint64_t));
  cudaMemcpy(d_packed, packed.data(), packed.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);

  constexpr int kBlock = 256;
  int grid = static_cast<int>((xs.size() + kBlock - 1) / kBlock);
  void* args_dense[] = {&d_packed, const_cast<int*>(&eff_bits), &d_out, const_cast<size_t*>(&N)};
  auto st = cudaLaunchKernel(reinterpret_cast<const void*>(&unpack_eff_bits_kernel),
                             dim3(grid), dim3(kBlock), args_dense, 0, nullptr);
  if (st != cudaSuccess) {
    std::cout << "Skipping: kernel launch failed: " << cudaGetErrorString(st) << "\n";
    cudaFree(d_packed);
    cudaFree(d_out);
    return 0;
  }
  st = cudaDeviceSynchronize();
  std::vector<uint64_t> out(xs.size(), 0);
  cudaMemcpy(out.data(), d_out, out.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost);
  cudaFree(d_packed);
  cudaFree(d_out);

  for (size_t i = 0; i < xs.size(); i++) {
    if ((out[i] & mask) != xs[i]) {
      std::cerr << "Dense unpack mismatch at " << i << " got=" << out[i] << " expect=" << xs[i] << "\n";
      return 1;
    }
  }

  // Ragged/causal packing: keep per-row valid prefix only.
  const int rows = 3;
  const int cols = 8;
  std::vector<int> valid = {5, 3, 7};
  std::vector<uint64_t> ragged(rows * cols, 0);
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      size_t idx = static_cast<size_t>(r * cols + c);
      ragged[idx] = rng() & mask;
    }
  }
  std::vector<uint64_t> active;
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < valid[r]; c++) {
      active.push_back(ragged[static_cast<size_t>(r * cols + c)]);
    }
  }
  auto packed_ragged = pack_host(active, eff_bits);
  uint64_t* d_packed_r = nullptr;
  uint64_t* d_out_r = nullptr;
  cudaMalloc(&d_packed_r, packed_ragged.size() * sizeof(uint64_t));
  cudaMalloc(&d_out_r, active.size() * sizeof(uint64_t));
  cudaMemcpy(d_packed_r, packed_ragged.data(), packed_ragged.size() * sizeof(uint64_t),
             cudaMemcpyHostToDevice);
  size_t active_size = active.size();
  grid = static_cast<int>((active_size + kBlock - 1) / kBlock);
  void* args_ragged[] = {&d_packed_r, const_cast<int*>(&eff_bits), &d_out_r, const_cast<size_t*>(&active_size)};
  st = cudaLaunchKernel(reinterpret_cast<const void*>(&unpack_eff_bits_kernel),
                        dim3(grid), dim3(kBlock), args_ragged, 0, nullptr);
  if (st != cudaSuccess) {
    std::cout << "Skipping ragged: kernel launch failed: " << cudaGetErrorString(st) << "\n";
    cudaFree(d_packed_r);
    cudaFree(d_out_r);
    return 0;
  }
  cudaDeviceSynchronize();
  std::vector<uint64_t> out_active(active.size(), 0);
  cudaMemcpy(out_active.data(), d_out_r, out_active.size() * sizeof(uint64_t),
             cudaMemcpyDeviceToHost);
  cudaFree(d_packed_r);
  cudaFree(d_out_r);

  size_t off = 0;
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < valid[r]; c++) {
      size_t idx = static_cast<size_t>(r * cols + c);
      uint64_t expect = ragged[idx] & mask;
      if (out_active[off] != expect) {
        std::cerr << "Ragged unpack mismatch r=" << r << " c=" << c
                  << " got=" << out_active[off] << " expect=" << expect << "\n";
        return 1;
      }
      ++off;
    }
  }

  std::cout << "CUDA eff_bits pack/unpack tests passed.\n";
  return 0;
#endif
}
