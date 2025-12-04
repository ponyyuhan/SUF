#include <random>
#include <vector>
#include <cassert>
#include <iostream>
#include "proto/backend_clear.hpp"
#include "proto/myl7_fss_backend.hpp"
#include "proto/pfss_utils.hpp"

static bool recon_bit_xor(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
  uint8_t v = 0;
  if (!a.empty()) v ^= (a[0] & 1u);
  if (!b.empty()) v ^= (b[0] & 1u);
  return v & 1u;
}

template<typename Backend>
void run_backend_tests(const char* name) {
  Backend backend;
  std::mt19937_64 rng(12345);
  for (int trial = 0; trial < 200; trial++) {
    int bits = 8 + (trial % 8);
    uint64_t alpha = rng() & ((bits == 64) ? ~0ull : ((1ull << bits) - 1));
    auto alpha_bits = backend.u64_to_bits_msb(alpha, bits);
    std::vector<uint8_t> payload{1u};
    auto kp = backend.gen_dcf(bits, alpha_bits, payload);
    for (int t = 0; t < 50; t++) {
      uint64_t x = rng() & ((bits == 64) ? ~0ull : ((1ull << bits) - 1));
      auto xb = backend.u64_to_bits_msb(x, bits);
      auto o0 = backend.eval_dcf(bits, kp.k0, xb);
      auto o1 = backend.eval_dcf(bits, kp.k1, xb);
      bool recon = recon_bit_xor(o0, o1);
      bool plain = (x < alpha);
      if (recon != plain) {
        std::cerr << "Backend " << name << " mismatch bits=" << bits << " alpha=" << alpha << " x=" << x << "\n";
        std::exit(1);
      }
    }
  }
}

int main() {
  run_backend_tests<proto::ClearBackend>("ClearBackend");
  run_backend_tests<proto::Myl7FssBackend>("Myl7FssBackend");
  std::cout << "Predicate semantics XOR test passed\n";
  return 0;
}
