#pragma once

#include <cstdint>
#include <random>
#include <vector>
#include "compiler/utils.hpp"
#include "core/ring.hpp"
#include "mpc/net.hpp"
#include "mpc/shares.hpp"

namespace gates {

// Faithful arithmetic right shift with masking (open-and-correct style).
// y = floor((x + r) / 2^f) - floor(r / 2^f) + out_mask
struct ArsParams {
  int frac_bits = 0;
};

struct ArsKey {
  ArsParams params;
  std::vector<uint64_t> r_value;
  std::vector<uint64_t> r_share;
  std::vector<uint64_t> r_high_share;
  std::vector<uint64_t> out_mask;
};

struct ArsKeys {
  ArsKey party0;
  ArsKey party1;
};

inline ArsKeys dealer_make_ars_keys(const ArsParams& params,
                                    size_t n,
                                    std::mt19937_64& rng) {
  ArsKeys keys;
  keys.party0.params = params;
  keys.party1.params = params;
  keys.party0.r_value.resize(n);
  keys.party1.r_value.resize(n);
  keys.party0.r_share.resize(n);
  keys.party1.r_share.resize(n);
  keys.party0.r_high_share.resize(n);
  keys.party1.r_high_share.resize(n);
  keys.party0.out_mask.resize(n);
  keys.party1.out_mask.resize(n);

  std::uniform_int_distribution<uint64_t> dist;
  for (size_t i = 0; i < n; ++i) {
    uint64_t r = dist(rng);
    uint64_t r0 = dist(rng);
    uint64_t r1 = r - r0;
    keys.party0.r_value[i] = r;
    keys.party1.r_value[i] = r;
    keys.party0.r_share[i] = r0;
    keys.party1.r_share[i] = r1;

    uint64_t r_hi = r >> params.frac_bits;
    uint64_t rhi0 = dist(rng);
    uint64_t rhi1 = r_hi - rhi0;
    keys.party0.r_high_share[i] = rhi0;
    keys.party1.r_high_share[i] = rhi1;

    uint64_t m = dist(rng);
    auto [m0, m1] = compiler::split_u64(rng, m);
    keys.party0.out_mask[i] = m0;
    keys.party1.out_mask[i] = m1;
  }
  return keys;
}

inline std::vector<mpc::AddShare<core::Z2n<64>>> eval_ars_faithful(
    const ArsKey& k,
    int party,
    net::Chan& ch,
    const std::vector<mpc::AddShare<core::Z2n<64>>>& xs) {
  std::vector<mpc::AddShare<core::Z2n<64>>> out(xs.size());
  const int f = k.params.frac_bits;
  for (size_t i = 0; i < xs.size(); ++i) {
    uint64_t masked = xs[i].s.v + k.r_share[i];
    uint64_t other = 0;
    if (party == 0) {
      ch.send_u64(masked);
      other = ch.recv_u64();
    } else {
      other = ch.recv_u64();
      ch.send_u64(masked);
    }
    uint64_t x_hat = masked + other;
    uint64_t r = k.r_value[i];
    uint64_t x_unmasked = x_hat - r;
    int64_t x_signed = static_cast<int64_t>(x_unmasked);
    int64_t y_signed = x_signed >> f;

    uint64_t y = (party == 0) ? static_cast<uint64_t>(y_signed) : 0ull;
    y += k.out_mask[i];
    out[i] = {core::Z2n<64>(y)};
  }
  return out;
}

}  // namespace gates
