#include <cassert>
#include <random>
#include <algorithm>
#include <vector>
#include <iostream>
#include "proto/sigma_fast_backend_ext.hpp"

using proto::SigmaFastBackend;
using proto::IntervalLutDesc;

static bool test_packed_lt_once(SigmaFastBackend& backend, int in_bits, size_t num_thr, std::mt19937_64& rng) {
  std::vector<uint64_t> thr(num_thr);
  uint64_t mask = (in_bits == 64) ? ~0ull : ((uint64_t(1) << in_bits) - 1);
  for (size_t i = 0; i < num_thr; i++) thr[i] = rng() & mask;
  std::sort(thr.begin(), thr.end());

  auto kp = backend.gen_packed_lt(in_bits, thr);
  size_t out_words = (num_thr + 63) / 64;
  const size_t N = 64;
  std::vector<uint64_t> xs(N);
  for (size_t i = 0; i < N; i++) xs[i] = rng();

  size_t key_bytes = kp.k0.bytes.size();
  std::vector<uint8_t> keys0(N * key_bytes), keys1(N * key_bytes);
  for (size_t i = 0; i < N; i++) {
    std::memcpy(keys0.data() + i * key_bytes, kp.k0.bytes.data(), key_bytes);
    std::memcpy(keys1.data() + i * key_bytes, kp.k1.bytes.data(), key_bytes);
  }
  std::vector<uint64_t> out0(N * out_words), out1(N * out_words);
  backend.eval_packed_lt_many(key_bytes, keys0.data(), xs, in_bits, static_cast<int>(out_words), out0.data());
  backend.eval_packed_lt_many(key_bytes, keys1.data(), xs, in_bits, static_cast<int>(out_words), out1.data());

  for (size_t i = 0; i < N; i++) {
    for (size_t w = 0; w < out_words; w++) {
      uint64_t recon = out0[i * out_words + w] ^ out1[i * out_words + w];
      uint64_t expect = 0;
      for (size_t t = 0; t < 64; t++) {
        size_t idx = w * 64 + t;
        if (idx >= thr.size()) break;
        bool bit = (xs[i] & mask) < (thr[idx] & mask);
        if (bit) expect |= (uint64_t(1) << t);
      }
      if (recon != expect) return false;
    }
  }
  return true;
}

static bool test_interval_lut_once(SigmaFastBackend& backend, std::mt19937_64& rng) {
  IntervalLutDesc desc;
  desc.in_bits = 64;
  desc.out_words = 4;
  desc.cutpoints = {0ull, (1ull << 20), (1ull << 40), ~0ull};
  desc.payload_flat.resize((desc.cutpoints.size() - 1) * desc.out_words);
  for (size_t i = 0; i < desc.payload_flat.size(); i++) desc.payload_flat[i] = rng();

  auto kp = backend.gen_interval_lut(desc);
  size_t key_bytes = kp.k0.bytes.size();
  const size_t N = 64;
  std::vector<uint64_t> xs(N);
  for (size_t i = 0; i < N; i++) xs[i] = rng();

  std::vector<uint8_t> keys0(N * key_bytes), keys1(N * key_bytes);
  for (size_t i = 0; i < N; i++) {
    std::memcpy(keys0.data() + i * key_bytes, kp.k0.bytes.data(), key_bytes);
    std::memcpy(keys1.data() + i * key_bytes, kp.k1.bytes.data(), key_bytes);
  }

  std::vector<uint64_t> out0(N * desc.out_words), out1(N * desc.out_words);
  backend.eval_interval_lut_many_u64(key_bytes, keys0.data(), xs, desc.out_words, out0.data());
  backend.eval_interval_lut_many_u64(key_bytes, keys1.data(), xs, desc.out_words, out1.data());

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
  SigmaFastBackend backend;
  std::mt19937_64 rng(2026);

  assert(test_packed_lt_once(backend, 64, 80, rng));
  assert(test_packed_lt_once(backend, 12, 100, rng));
  assert(test_interval_lut_once(backend, rng));

  std::cout << "SigmaFastBackend predicate/interval tests passed\n";
  return 0;
}
