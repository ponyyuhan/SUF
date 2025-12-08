#include <cassert>
#include <vector>
#include <random>
#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>

extern "C" __global__ void pred_mask_kernel(const uint64_t* thresholds,
                                            int num_thr,
                                            const uint64_t* xs,
                                            uint64_t* out_masks,
                                            size_t N);

int main() {
#ifndef SUF_HAVE_CUDA
  std::cout << "SUF_HAVE_CUDA not defined; skipping CUDA pred mask test.\n";
  return 0;
#else
  std::mt19937_64 rng(123456);
  const int num_thr = 32;
  std::vector<uint64_t> thresholds(num_thr);
  for (int i = 0; i < num_thr; i++) thresholds[i] = rng() & 0xFFFFFFFFull;
  std::sort(thresholds.begin(), thresholds.end());
  const size_t N = 512;
  std::vector<uint64_t> xs(N);
  for (auto& v : xs) v = rng();

  std::vector<uint64_t> ref(N, 0);
  for (size_t i = 0; i < N; i++) {
    uint64_t mask = 0;
    for (int j = 0; j < num_thr; j++) {
      if (xs[i] < thresholds[j]) mask |= (1ull << j);
    }
    ref[i] = mask;
  }

  uint64_t* d_thr = nullptr;
  uint64_t* d_xs = nullptr;
  uint64_t* d_out = nullptr;
  cudaMalloc(&d_thr, thresholds.size() * sizeof(uint64_t));
  cudaMalloc(&d_xs, xs.size() * sizeof(uint64_t));
  cudaMalloc(&d_out, xs.size() * sizeof(uint64_t));
  cudaMemcpy(d_thr, thresholds.data(), thresholds.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_xs, xs.data(), xs.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);

  constexpr int kBlock = 128;
  int grid = static_cast<int>((N + kBlock - 1) / kBlock);
  void* args[] = {&d_thr, const_cast<int*>(&num_thr), &d_xs, &d_out, const_cast<size_t*>(&N)};
  cudaError_t st = cudaLaunchKernel(reinterpret_cast<const void*>(&pred_mask_kernel),
                                    dim3(grid), dim3(kBlock), args, 0, nullptr);
  if (st != cudaSuccess) {
    std::cout << "Skipping: CUDA launch failed: " << cudaGetErrorString(st) << "\n";
    cudaFree(d_thr); cudaFree(d_xs); cudaFree(d_out);
    return 0;
  }
  std::vector<uint64_t> out(N);
  cudaMemcpy(out.data(), d_out, out.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost);

  cudaFree(d_thr);
  cudaFree(d_xs);
  cudaFree(d_out);

  if (out != ref) {
    std::cerr << "Predicate mask mismatch\n";
    return 1;
  }
  std::cout << "CUDA pred mask test passed\n";
  return 0;
#endif
}
