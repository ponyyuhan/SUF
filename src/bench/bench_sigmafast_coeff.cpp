#include <chrono>
#include <iostream>
#include <limits>
#include <random>
#include "proto/sigma_fast_backend_ext.hpp"

int main() {
  try {
  proto::SigmaFastBackend be;
  std::mt19937_64 rng(2024);
  const int in_bits = 64;
  const size_t N = 1 << 15;
  for (int out_words : {8, 16, 32}) {
    for (int intervals : {8, 12}) {
      proto::IntervalLutDesc desc;
      desc.in_bits = in_bits;
      desc.out_words = out_words;
      desc.cutpoints.clear();
      desc.cutpoints.push_back(0);
      uint64_t step = std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(intervals);
      for (int i = 1; i <= intervals; i++) desc.cutpoints.push_back(step * static_cast<uint64_t>(i));
      desc.payload_flat.resize(static_cast<size_t>(out_words) * (desc.cutpoints.size() - 1));
      for (auto& v : desc.payload_flat) v = rng();
      auto kp = be.gen_interval_lut(desc);
      std::vector<uint64_t> xs(N);
      for (auto& v : xs) v = rng();
      std::vector<uint8_t> keys_flat(N * kp.k0.bytes.size());
      for (size_t i = 0; i < N; i++) std::memcpy(keys_flat.data() + i * kp.k0.bytes.size(), kp.k0.bytes.data(), kp.k0.bytes.size());
      std::vector<uint64_t> outs(N * static_cast<size_t>(out_words));
      auto t0 = std::chrono::high_resolution_clock::now();
      be.eval_interval_lut_many_u64(kp.k0.bytes.size(), keys_flat.data(), xs, out_words, outs.data());
      auto t1 = std::chrono::high_resolution_clock::now();
      double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      std::cout << "SigmaFast interval LUT: intervals=" << intervals
                << " out_words=" << out_words << " time_ms=" << ms
                << " ns/elem=" << (ms * 1e6 / N) << "\n";
    }
  }
  // SIGMA-style large run: N=1e6, intervals=8, out_words=8
  {
    const size_t bigN = 1000000;
    proto::IntervalLutDesc desc;
    desc.in_bits = in_bits;
    desc.out_words = 8;
    desc.cutpoints.clear();
    for (int i = 0; i <= 8; i++) desc.cutpoints.push_back(static_cast<uint64_t>(i) << 61);
    desc.payload_flat.resize(static_cast<size_t>(desc.out_words) * (desc.cutpoints.size() - 1));
    for (auto& v : desc.payload_flat) v = rng();
    auto kp = be.gen_interval_lut(desc);
    std::vector<uint64_t> xs(bigN);
    for (auto& v : xs) v = rng();
    std::vector<uint8_t> keys_flat(bigN * kp.k0.bytes.size());
    for (size_t i = 0; i < bigN; i++) std::memcpy(keys_flat.data() + i * kp.k0.bytes.size(), kp.k0.bytes.data(), kp.k0.bytes.size());
    std::vector<uint64_t> outs(bigN * static_cast<size_t>(desc.out_words));
    auto t0 = std::chrono::high_resolution_clock::now();
    be.eval_interval_lut_many_u64(kp.k0.bytes.size(), keys_flat.data(), xs, desc.out_words, outs.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "SigmaFast interval LUT (SIGMA-style): intervals=8 out_words=8 N=1e6"
              << " time_ms=" << ms << " ns/elem=" << (ms * 1e6 / bigN) << "\n";
  }
  return 0;
  } catch (const std::exception& e) {
    std::cerr << "bench_sigmafast_coeff error: " << e.what() << "\n";
    return 1;
  }
}
