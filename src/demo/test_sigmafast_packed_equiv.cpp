#include <random>
#include <vector>
#include <iostream>
#include <cassert>
#include "proto/sigma_fast_backend_ext.hpp"
#include "proto/backend_clear.hpp"

// Compare SigmaFast packed outputs vs clear backend for packed predicates and interval LUT.
static bool check_packed_pred(int in_bits, size_t num_thr, std::mt19937_64& rng) {
  proto::SigmaFastBackend sb;
  proto::ClearBackend cb;
  std::vector<uint64_t> thr(num_thr);
  uint64_t mask = (in_bits == 64) ? ~0ull : ((uint64_t(1) << in_bits) - 1);
  for (size_t i = 0; i < num_thr; i++) thr[i] = rng() & mask;
  auto kp = sb.gen_packed_lt(in_bits, thr);
  size_t out_words = (num_thr + 63) / 64;
  const size_t N = 256;
  std::vector<uint64_t> xs(N);
  for (auto& x : xs) x = rng() & mask;
  size_t key_bytes = kp.k0.bytes.size();
  std::vector<uint8_t> k0(N * key_bytes), k1(N * key_bytes);
  for (size_t i = 0; i < N; i++) {
    std::memcpy(k0.data() + i * key_bytes, kp.k0.bytes.data(), key_bytes);
    std::memcpy(k1.data() + i * key_bytes, kp.k1.bytes.data(), key_bytes);
  }
  std::vector<uint64_t> out0(N * out_words), out1(N * out_words);
  sb.eval_packed_lt_many(key_bytes, k0.data(), xs, in_bits, static_cast<int>(out_words), out0.data());
  sb.eval_packed_lt_many(key_bytes, k1.data(), xs, in_bits, static_cast<int>(out_words), out1.data());
  // Reconstruct and compare to clear.
  for (size_t i = 0; i < N; i++) {
    for (size_t w = 0; w < out_words; w++) {
      uint64_t recon = out0[i * out_words + w] ^ out1[i * out_words + w];
      uint64_t expect = 0;
      for (size_t b = 0; b < 64; b++) {
        size_t idx = w * 64 + b;
        if (idx >= thr.size()) break;
        bool bit = (xs[i] & mask) < (thr[idx] & mask);
        if (bit) expect |= (uint64_t(1) << b);
      }
      if (recon != expect) return false;
    }
  }
  return true;
}

static bool check_interval_lut(std::mt19937_64& rng) {
  proto::SigmaFastBackend sb;
  proto::IntervalLutDesc desc;
  desc.in_bits = 64;
  desc.out_words = 4;
  desc.cutpoints = {0ull, (1ull << 16), (1ull << 32), ~0ull};
  desc.payload_flat.resize((desc.cutpoints.size() - 1) * desc.out_words);
  for (auto& v : desc.payload_flat) v = rng();
  auto kp = sb.gen_interval_lut(desc);
  size_t key_bytes = kp.k0.bytes.size();
  const size_t N = 256;
  std::vector<uint64_t> xs(N);
  for (auto& x : xs) x = rng();
  std::vector<uint8_t> k0(N * key_bytes), k1(N * key_bytes);
  for (size_t i = 0; i < N; i++) {
    std::memcpy(k0.data() + i * key_bytes, kp.k0.bytes.data(), key_bytes);
    std::memcpy(k1.data() + i * key_bytes, kp.k1.bytes.data(), key_bytes);
  }
  std::vector<uint64_t> out0(N * desc.out_words), out1(N * desc.out_words);
  sb.eval_interval_lut_many_u64(key_bytes, k0.data(), xs, desc.out_words, out0.data());
  sb.eval_interval_lut_many_u64(key_bytes, k1.data(), xs, desc.out_words, out1.data());
  for (size_t i = 0; i < N; i++) {
    size_t idx = desc.cutpoints.size() - 2;
    for (size_t j = 0; j + 1 < desc.cutpoints.size(); j++) {
      if (xs[i] >= desc.cutpoints[j] && xs[i] < desc.cutpoints[j + 1]) { idx = j; break; }
    }
    for (int w = 0; w < desc.out_words; w++) {
      uint64_t recon = out0[i * desc.out_words + w] + out1[i * desc.out_words + w];
      uint64_t expect = desc.payload_flat[idx * desc.out_words + static_cast<size_t>(w)];
      if (recon != expect) return false;
    }
  }
  return true;
}

int main() {
  std::mt19937_64 rng(7);
  assert(check_packed_pred(64, 128, rng));
  assert(check_packed_pred(12, 96, rng));
  assert(check_interval_lut(rng));
  std::cout << "SigmaFast packed-mode correctness tests passed\n";
  return 0;
}
