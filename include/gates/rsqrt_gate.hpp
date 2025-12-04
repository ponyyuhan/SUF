#pragma once

#include <random>
#include "gates/piecewise_poly.hpp"
#include "gates/tables/rsqrt_piecewise_affine_init.hpp"
#include "pfss/backend_cleartext.hpp"
#include "pfss/pfss.hpp"

namespace gates {

struct RsqrtParams {
  int frac_bits = 16;
  int nr_iters = 1;
  double eps = 1.0 / 1024.0;
  double vmax = 16.0;
};

struct RsqrtKey {
  PiecewisePolyKey init_key;
  int frac_bits = 16;
  int nr_iters = 1;
  mpc::AddShare<core::Z2n<64>> out_mask;
  std::vector<mpc::BeaverTripleA<core::Z2n<64>>> nr_triples;  // 3 per iter
};

struct RsqrtKeys {
  RsqrtKey party0;
  RsqrtKey party1;
};

template<typename CoeffBackend>
inline RsqrtKeys dealer_make_rsqrt_keys(
    CoeffBackend& backend,
    const pfss::PublicParams& pp_coeff,
    const RsqrtParams& params,
    std::mt19937_64& rng) {
  RsqrtKeys out;
  auto spec = make_rsqrt_affine_init_spec(params.frac_bits, params.eps, params.vmax);
  auto pw_keys = dealer_make_piecewise_poly_keys(backend, pp_coeff, spec, rng);

  int need = params.nr_iters * 3;
  for (int i = 0; i < need; ++i) {
    auto tri_pair = mpc::dealer_make_tripleA<core::Z2n<64>>(rng);
    out.party0.nr_triples.push_back(tri_pair.first);
    out.party1.nr_triples.push_back(tri_pair.second);
  }

  uint64_t r_out = rng();
  auto [o0, o1] = compiler::split_u64(rng, r_out);
  out.party0.out_mask = {core::Z2n<64>(o0)};
  out.party1.out_mask = {core::Z2n<64>(o1)};

  out.party0.init_key = pw_keys.party0;
  out.party1.init_key = pw_keys.party1;
  out.party0.init_key.r_out_share = {core::Z2n<64>(0)};
  out.party1.init_key.r_out_share = {core::Z2n<64>(0)};
  out.party0.frac_bits = params.frac_bits;
  out.party1.frac_bits = params.frac_bits;
  out.party0.nr_iters = params.nr_iters;
  out.party1.nr_iters = params.nr_iters;
  return out;
}

inline mpc::AddShare<core::Z2n<64>> eval_rsqrt_gate(
    const RsqrtKey& k,
    int party,
    net::Chan& ch,
    const pfss::Backend<pfss::CoeffPayload>& backend,
    const pfss::PublicParams& pp,
    uint64_t x_hat_public) {
  mpc::AddShare<core::Z2n<64>> x_share =
      (party == 0) ? mpc::AddShare<core::Z2n<64>>{core::Z2n<64>(x_hat_public - k.init_key.r_in_share.s.v)}
                   : mpc::AddShare<core::Z2n<64>>{core::Z2n<64>(0ull - k.init_key.r_in_share.s.v)};

  auto y_share = eval_piecewise_poly(k.init_key, party, ch, backend, pp, x_hat_public);

  int fb = k.frac_bits;
  core::Z2n<64> one_const((party == 0) ? (uint64_t(1) << fb) : 0ull);
  core::Z2n<64> half_const((party == 0) ? (uint64_t(1) << (fb - 1)) : 0ull);  // 0.5
  core::Z2n<64> one_pt5_const{core::Z2n<64>(one_const.v + half_const.v)};

  size_t tri_pos = 0;
  for (int iter = 0; iter < k.nr_iters; ++iter) {
    if (tri_pos + 2 >= k.nr_triples.size()) break;
    auto y2 = mul_rescale_share(party, ch, y_share, y_share, k.nr_triples[tri_pos], fb);
    ++tri_pos;
    auto xy2 = mul_rescale_share(party, ch, x_share, y2, k.nr_triples[tri_pos], fb);
    ++tri_pos;
    mpc::AddShare<core::Z2n<64>> half_xy2{core::Z2n<64>(arith_shift_signed(static_cast<int64_t>(xy2.s.v), 1))};
    mpc::AddShare<core::Z2n<64>> inner{core::Z2n<64>(one_pt5_const.v - half_xy2.s.v)};
    auto update = mul_rescale_share(party, ch, y_share, inner, k.nr_triples[tri_pos], fb);
    ++tri_pos;
    y_share = update;
  }

  y_share.s += k.out_mask.s;
  return y_share;
}

inline int64_t ref_rsqrt_fixed(
    const PiecewisePolySpec& init_spec,
    int64_t x_fixed,
    int frac_bits,
    int nr_iters) {
  int64_t y = eval_piecewise_poly_ref(init_spec, x_fixed);
  int64_t one = static_cast<int64_t>(uint64_t(1) << frac_bits);
  int64_t half = static_cast<int64_t>(uint64_t(1) << (frac_bits - 1));
  int64_t onept5 = one + half;
  for (int i = 0; i < nr_iters; ++i) {
    int64_t y2 = mul_rescale_ring(y, y, frac_bits);
    int64_t xy2 = mul_rescale_ring(x_fixed, y2, frac_bits);
    int64_t half_xy2 = arith_shift_signed(xy2, 1);
    int64_t inner = onept5 - half_xy2;
    y = mul_rescale_ring(y, inner, frac_bits);
  }
  return y;
}

}  // namespace gates
