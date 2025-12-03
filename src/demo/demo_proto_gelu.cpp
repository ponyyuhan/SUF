#include <iostream>
#include <vector>

#include "proto/beaver.hpp"
#include "proto/gelu_spline_dealer.hpp"
#include "proto/myl7_fss_backend.hpp"

using namespace proto;

int main() {
  GeluSplineParams params;
  params.f = 4;
  params.d = 1;    // linear delta for demo
  params.T = 32;   // clip bound (scaled)
  params.a = { -static_cast<int64_t>(params.T), 0, static_cast<int64_t>(params.T) };
  // Piece 0: delta=0 (left tail), piece1: small slope, piece2: tail handled by zero vector in dealer
  params.coeffs = {
      {0, 0},           // [-T,0)
      {0, 1},           // [0,T)
  };

  Myl7FssBackend backend;
  BeaverDealer dealer;
  auto keys = GeluSplineDealer::keygen(params, backend, dealer);
  u64 r_in = add_mod(keys.k0.r_in_share, keys.k1.r_in_share);
  std::cout << "GeLU-spline demo r_in=" << r_in << " T=" << params.T << " f=" << params.f << "\n";

  std::vector<int64_t> xs = { -40, -10, 0, 8, 40 };
  for (auto x_signed : xs) {
    u64 x = static_cast<u64>(x_signed);
    u64 x_hat = add_mod(x, r_in);
    u64 x_hat_bias = add_mod(x_hat, (u64(1) << 63));
    auto xb = backend.u64_to_bits_msb(x_hat_bias, 64);

    // Reconstruct coeffs via step cuts
    std::vector<u64> coeff = keys.k0.cuts.empty() ? std::vector<u64>(params.d + 1, 0)
                                                  : keys.k0.cuts.front().delta;  // placeholder
    coeff.assign(params.d + 1, 0);

    // base v0 is zero (dealer uses zero vec for tails), so sum deltas where x >= start.
    for (size_t i = 0; i < keys.k0.cuts.size(); i++) {
      auto bytes0 = backend.eval_dcf(64, keys.k0.cuts[i].party0.dcf_key, xb);
      auto bytes1 = backend.eval_dcf(64, keys.k1.cuts[i].party1.dcf_key, xb);
      auto share0 = unpack_u64_vec_le(bytes0);
      auto share1 = unpack_u64_vec_le(bytes1);
      if (share0.size() != coeff.size()) coeff.resize(share0.size());
      for (size_t j = 0; j < coeff.size() && j < share0.size(); j++) {
        coeff[j] = add_mod(coeff[j], add_mod(share0[j], share1[j]));
      }
    }

    // Evaluate delta(x) = c0 + c1*x (since d=1)
    u64 delta = coeff[0] + mul_mod(coeff[1], x);
    // x_plus = max(x,0)
    u64 x_plus = (x_signed >= 0) ? x : 0;
    u64 y = add_mod(x_plus, delta);

    std::cout << "x=" << x_signed << " hat=" << x_hat << " coeff[0]=" << coeff[0]
              << " coeff[1]=" << coeff[1] << " -> y=" << static_cast<int64_t>(y) << "\n";
  }
  return 0;
}
