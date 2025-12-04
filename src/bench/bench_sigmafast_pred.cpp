#include <chrono>
#include <iostream>
#include <random>
#include "proto/sigma_fast_backend_ext.hpp"

int main() {
  try {
  proto::SigmaFastBackend be;
  std::mt19937_64 rng(12345);
  const int in_bits = 64;
  std::vector<uint64_t> thrs;
  for (int i = 0; i < 128; i++) thrs.push_back(rng());
  auto kp = be.gen_packed_lt(in_bits, thrs);
  const size_t N = 1 << 16;
  std::vector<uint64_t> xs(N);
  for (auto& v : xs) v = rng();
  std::vector<uint8_t> keys_flat(N * kp.k0.bytes.size());
  for (size_t i = 0; i < N; i++) {
    const auto& src = kp.k0.bytes;
    std::memcpy(keys_flat.data() + i * src.size(), src.data(), src.size());
  }
  int out_words = static_cast<int>((thrs.size() + 63) / 64);
  std::vector<uint64_t> outs(N * static_cast<size_t>(out_words));
  auto t0 = std::chrono::high_resolution_clock::now();
  be.eval_packed_lt_many(kp.k0.bytes.size(), keys_flat.data(), xs, in_bits, out_words, outs.data());
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "SigmaFast packed pred: N=" << N << " thresholds=" << thrs.size()
            << " time_ms=" << ms << " ns/elem=" << (ms * 1e6 / N) << "\n";
  return 0;
  } catch (const std::exception& e) {
    std::cerr << "bench_sigmafast_pred error: " << e.what() << "\n";
    return 1;
  }
}
