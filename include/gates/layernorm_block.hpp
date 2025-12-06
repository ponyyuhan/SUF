#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>
#include "core/ring.hpp"
#include "gates/rsqrt_gate.hpp"
#include "pfss/backend_cleartext.hpp"
#include "pfss/pfss.hpp"

namespace gates {

struct LayerNormParams {
  int frac_bits = 16;
  size_t L = 128;
  double eps = 1.0 / 1024.0;
  int nr_iters = 1;
};

struct LayerNormKey {
  LayerNormParams params;
  PiecewisePolySpec rsqrt_init_spec;
  int rsqrt_iters = 1;
  int64_t eps_fixed = 0;
  std::vector<mpc::AddShare<core::Z2n<64>>> out_masks;
};

struct LayerNormKeys {
  LayerNormKey party0;
  LayerNormKey party1;
};

inline LayerNormKeys dealer_make_layernorm_keys(const LayerNormParams& params,
                                                std::mt19937_64& rng) {
  LayerNormKeys out;
  RsqrtParams rparams;
  rparams.frac_bits = params.frac_bits;
  rparams.eps = params.eps;
  rparams.vmax = 16.0;
  rparams.nr_iters = params.nr_iters;
  out.party0.rsqrt_init_spec = make_rsqrt_affine_init_spec(rparams.frac_bits, rparams.eps, rparams.vmax);
  out.party1.rsqrt_init_spec = out.party0.rsqrt_init_spec;
  out.party0.rsqrt_iters = rparams.nr_iters;
  out.party1.rsqrt_iters = rparams.nr_iters;
  out.party0.params = params;
  out.party1.params = params;
  out.party0.eps_fixed = static_cast<int64_t>(std::llround(params.eps * std::ldexp(1.0, params.frac_bits)));
  out.party1.eps_fixed = out.party0.eps_fixed;

  out.party0.out_masks.resize(params.L);
  out.party1.out_masks.resize(params.L);
  for (size_t i = 0; i < params.L; ++i) {
    uint64_t r = rng();
    auto [a, b] = compiler::split_u64(rng, r);
    out.party0.out_masks[i] = {core::Z2n<64>(a)};
    out.party1.out_masks[i] = {core::Z2n<64>(b)};
  }
  return out;
}

inline std::vector<uint64_t> exchange_layer_shares(int party,
                                                   net::Chan& ch,
                                                   const std::vector<mpc::AddShare<core::Z2n<64>>>& mine) {
  std::vector<uint64_t> other(mine.size(), 0);
  if (party == 0) {
    for (auto& s : mine) ch.send_u64(s.s.v);
    for (size_t i = 0; i < mine.size(); ++i) other[i] = ch.recv_u64();
  } else {
    for (size_t i = 0; i < mine.size(); ++i) other[i] = ch.recv_u64();
    for (auto& s : mine) ch.send_u64(s.s.v);
  }
  return other;
}

inline void eval_layernorm_block(const LayerNormKey& k,
                                 int party,
                                 net::Chan& ch,
                                 const std::vector<mpc::AddShare<core::Z2n<64>>>& x_shares,
                                 const std::vector<mpc::AddShare<core::Z2n<64>>>* gamma_shares,
                                 const std::vector<mpc::AddShare<core::Z2n<64>>>* beta_shares,
                                 std::vector<mpc::AddShare<core::Z2n<64>>>& out_shares) {
  out_shares.resize(x_shares.size());
  auto other_x = exchange_layer_shares(party, ch, x_shares);
  std::vector<int64_t> x_plain(x_shares.size(), 0);
  for (size_t i = 0; i < x_shares.size(); ++i) {
    x_plain[i] = static_cast<int64_t>(x_shares[i].s.v + other_x[i]);
  }

  int64_t sum = 0;
  for (auto v : x_plain) sum += v;
  int64_t mu = x_plain.empty() ? 0 : (sum / static_cast<int64_t>(x_plain.size()));

  int64_t var_acc = 0;
  for (auto v : x_plain) {
    int64_t d = v - mu;
    __int128 sq = static_cast<__int128>(d) * static_cast<__int128>(d);
    var_acc += static_cast<int64_t>(sq >> k.params.frac_bits);
  }
  int64_t var = x_plain.empty() ? 0 : (var_acc / static_cast<int64_t>(x_plain.size()));
  int64_t var_eps = var + k.eps_fixed;
  int64_t r = ref_rsqrt_fixed(k.rsqrt_init_spec, var_eps, k.params.frac_bits, k.rsqrt_iters);

  std::vector<int64_t> gamma_plain, beta_plain;
  if (gamma_shares) {
    auto other_g = exchange_layer_shares(party, ch, *gamma_shares);
    gamma_plain.resize(gamma_shares->size());
    for (size_t i = 0; i < gamma_shares->size(); ++i) {
      gamma_plain[i] = static_cast<int64_t>((*gamma_shares)[i].s.v + other_g[i]);
    }
  }
  if (beta_shares) {
    auto other_b = exchange_layer_shares(party, ch, *beta_shares);
    beta_plain.resize(beta_shares->size());
    for (size_t i = 0; i < beta_shares->size(); ++i) {
      beta_plain[i] = static_cast<int64_t>((*beta_shares)[i].s.v + other_b[i]);
    }
  }

  int64_t one_fixed = static_cast<int64_t>(uint64_t(1) << k.params.frac_bits);
  for (size_t i = 0; i < x_plain.size(); ++i) {
    int64_t d = x_plain[i] - mu;
    __int128 prod = static_cast<__int128>(d) * static_cast<__int128>(r);
    int64_t z = static_cast<int64_t>(prod >> k.params.frac_bits);
    int64_t gamma = (gamma_shares && i < gamma_plain.size()) ? gamma_plain[i] : one_fixed;
    int64_t beta = (beta_shares && i < beta_plain.size()) ? beta_plain[i] : 0;
    __int128 zg = static_cast<__int128>(z) * static_cast<__int128>(gamma);
    int64_t y = static_cast<int64_t>(zg >> k.params.frac_bits);
    y += beta;
    uint64_t contrib = (party == 0) ? static_cast<uint64_t>(y) : 0ull;
    out_shares[i] = {core::Z2n<64>(contrib + k.out_masks[i].s.v)};
  }
}

}  // namespace gates
