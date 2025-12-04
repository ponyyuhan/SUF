#include <random>
#include <iostream>
#include "proto/myl7_fss_backend.hpp"

int main() {
#if defined(MYL7_FSS_AVAILABLE)
  // Try four combinations: (alpha_bits order, x_bits order) = {LSB,MSB}^2
  auto run_probe = [](bool alpha_msb, bool x_msb)->bool {
    proto::Myl7FssBackend::Params p;
    p.bits_msb_first = alpha_msb; // use for alpha packing
    proto::Myl7FssBackend backend(p);
    std::mt19937_64 rng(7);
    for (int trial = 0; trial < 20; trial++) {
      int bits = 16;
      uint64_t alpha = 1 + (rng() & 0xFFFF);
      auto alpha_bits = backend.u64_to_bits_msb(alpha, bits);
      std::vector<uint8_t> payload{1u};
      auto kp = backend.gen_dcf(bits, alpha_bits, payload);
      for (uint64_t x = alpha > 2 ? alpha - 2 : 0; x < alpha + 2; x++) {
        // Temporarily flip the x bit order
        proto::Myl7FssBackend::Params p_eval = p;
        p_eval.bits_msb_first = x_msb;
        proto::Myl7FssBackend backend_eval(p_eval);
        auto xb = backend_eval.u64_to_bits_msb(x, bits);
        auto o0 = backend_eval.eval_dcf(bits, kp.k0, xb);
        auto o1 = backend_eval.eval_dcf(bits, kp.k1, xb);
        uint8_t bit = 0;
        if (!o0.empty()) bit ^= (o0[0] & 1u);
        if (!o1.empty()) bit ^= (o1[0] & 1u);
        bool plain = x < alpha;
        if ((bit & 1u) != static_cast<uint8_t>(plain)) {
          return false;
        }
      }
    }
    return true;
  };
  if (run_probe(false, false)) { std::cout << "myl7 bit-order probe passed (alpha LSB, x LSB)\n"; return 0; }
  if (run_probe(true, false))  { std::cout << "myl7 bit-order probe passed (alpha MSB, x LSB)\n"; return 0; }
  if (run_probe(false, true))  { std::cout << "myl7 bit-order probe passed (alpha LSB, x MSB)\n"; return 0; }
  if (run_probe(true, true))   { std::cout << "myl7 bit-order probe passed (alpha MSB, x MSB)\n"; return 0; }
  std::cerr << "myl7 bit-order probe failed for all combinations\n";
  return 1;
#else
  std::cout << "myl7 headers not available; skipping bit-order probe\n";
#endif
  return 0;
}
