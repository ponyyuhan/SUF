#include <chrono>
#include <iostream>
#include <random>
#include "proto/sigma_fast_backend_ext.hpp"

int main() {
  try {
  proto::SigmaFastBackend be;
  std::mt19937_64 rng(2024);
  const int in_bits = 64;
  const int out_words = 8;
  proto::IntervalLutDesc desc;
  desc.in_bits = in_bits;
  desc.out_words = out_words;
  // 8 intervals
  for (int i = 0; i <= 8; i++) desc.cutpoints.push_back(static_cast<uint64_t>(i) << 40);
  desc.payload_flat.resize(static_cast<size_t>(out_words) * (desc.cutpoints.size() - 1));
  for (auto& v : desc.payload_flat) v = rng();
  auto kp = be.gen_interval_lut(desc);
  const size_t N = 1 << 15;
  std::vector<uint64_t> xs(N);
  for (auto& v : xs) v = rng();
  std::vector<uint8_t> keys_flat(N * kp.k0.bytes.size());
  for (size_t i = 0; i < N; i++) std::memcpy(keys_flat.data() + i * kp.k0.bytes.size(), kp.k0.bytes.data(), kp.k0.bytes.size());
  std::vector<uint64_t> outs(N * static_cast<size_t>(out_words));
  auto t0 = std::chrono::high_resolution_clock::now();
  be.eval_interval_lut_many_u64(kp.k0.bytes.size(), keys_flat.data(), xs, out_words, outs.data());
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "SigmaFast interval LUT: N=" << N << " out_words=" << out_words
            << " time_ms=" << ms << " ns/elem=" << (ms * 1e6 / N) << "\n";
  return 0;
  } catch (const std::exception& e) {
    std::cerr << "bench_sigmafast_coeff error: " << e.what() << "\n";
    return 1;
  }
}
