#include <iostream>
#include <random>
#include <vector>

#include "proto/beaver.hpp"
#include "proto/myl7_fss_backend.hpp"
#include "proto/reluars_dealer.hpp"

using namespace proto;

int main() {
  // Parameters
  ReluARSParams params;
  params.f = 4;
  // Simple correction LUT placeholder: all zeros
  params.delta.fill(0);

  Myl7FssBackend backend;
  BeaverDealer dealer;

  // Dealer keygen
  auto keys = ReluARSDealer::keygen(params, backend, dealer);
  u64 r_in = add_mod(keys.k0.r_in_share, keys.k1.r_in_share);
  std::cout << "ReluARS demo with r_in=" << r_in << " f=" << params.f << "\n";

  // Test a few inputs
  std::vector<int64_t> xs = { -20, -1, 0, 5, 33 };
  for (auto x_signed : xs) {
    u64 x = static_cast<u64>(x_signed);
    u64 x_hat = add_mod(x, r_in);
    std::cout << "x=" << x_signed << " hat=" << x_hat << " -> ";

    // Parties evaluate predicate DCFs (shares)
    auto xb = backend.u64_to_bits_msb(x_hat, 64);
    auto xb_low = backend.u64_to_bits_msb(mask_low(x_hat, params.f), params.f);

    auto c1_p0 = backend.eval_dcf(64, keys.k0.dcf_hat_lt_r, xb);
    auto c1_p1 = backend.eval_dcf(64, keys.k1.dcf_hat_lt_r, xb);
    u64 c1 = (c1_p0.empty() ? 0 : c1_p0[0]) + (c1_p1.empty() ? 0 : c1_p1[0]);

    auto c2_p0 = backend.eval_dcf(64, keys.k0.dcf_hat_lt_r_plus_2p63, xb);
    auto c2_p1 = backend.eval_dcf(64, keys.k1.dcf_hat_lt_r_plus_2p63, xb);
    u64 c2 = (c2_p0.empty() ? 0 : c2_p0[0]) + (c2_p1.empty() ? 0 : c2_p1[0]);

    auto t0 = backend.eval_dcf(params.f, keys.k0.dcf_low_lt_r_low, xb_low);
    auto t1 = backend.eval_dcf(params.f, keys.k1.dcf_low_lt_r_low, xb_low);
    u64 t = (t0.empty() ? 0 : t0[0]) + (t1.empty() ? 0 : t1[0]);

    auto d0 = backend.eval_dcf(params.f, keys.k0.dcf_low_lt_r_low_plus1, xb_low);
    auto d1 = backend.eval_dcf(params.f, keys.k1.dcf_low_lt_r_low_plus1, xb_low);
    u64 d = (d0.empty() ? 0 : d0[0]) + (d1.empty() ? 0 : d1[0]);

    // For demo: reconstruct w as sign of x, t and d as obtained
    u64 w = (static_cast<int64_t>(x) >= 0) ? 1 : 0;
    u64 z = add_mod(x, (params.f == 0 ? 0ull : (1ull << (params.f - 1))));
    u64 y_trunc = static_cast<u64>(static_cast<int64_t>(z) >> params.f);
    u64 y = (w == 0) ? 0 : y_trunc;

    std::cout << "c1=" << c1 << " c2=" << c2 << " t=" << t << " d=" << d
              << " -> y=" << static_cast<int64_t>(y) << "\n";
  }

  return 0;
}
