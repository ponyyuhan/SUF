#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>
#include "core/ring.hpp"
#include "gates/nexp_gate.hpp"
#include "gates/reciprocal_gate.hpp"
#include "pfss/backend_cleartext.hpp"
#include "pfss/pfss.hpp"

namespace gates {

struct SoftmaxBlockParams {
  int frac_bits = 16;
  size_t L = 128;
  int nr_iters_recip = 1;
};

struct SoftmaxBlockKey {
  SoftmaxBlockParams params;
  PiecewisePolySpec nexp_spec;
  PiecewisePolySpec recip_init_spec;
  int recip_iters = 1;
  std::vector<mpc::AddShare<core::Z2n<64>>> out_masks;  // size L
};

struct SoftmaxBlockKeys {
  SoftmaxBlockKey party0;
  SoftmaxBlockKey party1;
};

inline SoftmaxBlockKeys dealer_make_softmax_keys(
    const SoftmaxBlockParams& params,
    std::mt19937_64& rng) {
  SoftmaxBlockKeys out;
  NExpGateParams nexp_params;
  nexp_params.frac_bits = params.frac_bits;
  nexp_params.segments = 16;
  out.party0.nexp_spec = make_nexp_spec(nexp_params);
  out.party1.nexp_spec = out.party0.nexp_spec;

  ReciprocalParams rparams;
  rparams.frac_bits = params.frac_bits;
  rparams.nr_iters = params.nr_iters_recip;
  rparams.nmax = static_cast<double>(params.L);
  out.party0.recip_init_spec = make_recip_affine_init_spec(rparams.frac_bits, rparams.nmax);
  out.party1.recip_init_spec = out.party0.recip_init_spec;
  out.party0.recip_iters = rparams.nr_iters;
  out.party1.recip_iters = rparams.nr_iters;

  out.party0.params = params;
  out.party1.params = params;

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

inline std::vector<uint64_t> exchange_shares(
    int party,
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

inline void eval_softmax_block(
    const SoftmaxBlockKey& k,
    int party,
    net::Chan& ch,
    const std::vector<mpc::AddShare<core::Z2n<64>>>& x_shares,
    std::vector<mpc::AddShare<core::Z2n<64>>>& out_shares) {
  out_shares.resize(x_shares.size());
  auto other = exchange_shares(party, ch, x_shares);
  std::vector<int64_t> plain(x_shares.size(), 0);
  for (size_t i = 0; i < x_shares.size(); ++i) {
    plain[i] = static_cast<int64_t>(x_shares[i].s.v + other[i]);
  }

  int64_t max_x = plain.empty() ? 0 : plain[0];
  for (size_t i = 1; i < plain.size(); ++i) {
    if (plain[i] > max_x) max_x = plain[i];
  }

  std::vector<int64_t> exp_vals(plain.size(), 0);
  int64_t sum = 0;
  for (size_t i = 0; i < plain.size(); ++i) {
    int64_t diff = max_x - plain[i];
    exp_vals[i] = ref_nexp_fixed(k.nexp_spec, diff);
    sum += exp_vals[i];
  }

  if (sum == 0) sum = 1;  // avoid divide by zero in degenerate case
  int64_t inv = ref_reciprocal_fixed(k.recip_init_spec, sum, k.params.frac_bits, k.recip_iters);

  for (size_t i = 0; i < plain.size(); ++i) {
    __int128 prod = static_cast<__int128>(exp_vals[i]) * static_cast<__int128>(inv);
    int64_t y = static_cast<int64_t>(prod >> k.params.frac_bits);
    uint64_t contrib = (party == 0) ? static_cast<uint64_t>(y) : 0ull;
    core::Z2n<64> masked = core::Z2n<64>(contrib + k.out_masks[i].s.v);
    out_shares[i] = {masked};
  }
}

}  // namespace gates
