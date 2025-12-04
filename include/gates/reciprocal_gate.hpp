#pragma once

#include <random>
#include "gates/piecewise_poly.hpp"
#include "gates/tables/recip_piecewise_affine_init.hpp"
#include "pfss/backend_cleartext.hpp"
#include "pfss/pfss.hpp"

namespace gates {

struct ReciprocalParams {
  int frac_bits = 16;
  int nr_iters = 1;
  double nmax = 1024.0;
};

struct ReciprocalKey {
  PiecewisePolyKey init_key;
  int frac_bits = 16;
  int nr_iters = 1;
  mpc::AddShare<core::Z2n<64>> out_mask;
  std::vector<mpc::BeaverTripleA<core::Z2n<64>>> nr_triples;  // 2 per iter
};

struct ReciprocalKeys {
  ReciprocalKey party0;
  ReciprocalKey party1;
};

template<typename CoeffBackend>
inline ReciprocalKeys dealer_make_recip_keys(
    CoeffBackend& backend,
    const pfss::PublicParams& pp_coeff,
    const ReciprocalParams& params,
    std::mt19937_64& rng) {
  ReciprocalKeys out;
  auto spec = make_recip_affine_init_spec(params.frac_bits, params.nmax);
  auto pw_keys = dealer_make_piecewise_poly_keys(backend, pp_coeff, spec, rng);

  // NR triples
  int need = params.nr_iters * 2;
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
  // zero out init mask so NR math uses raw share
  out.party0.init_key.r_out_share = {core::Z2n<64>(0)};
  out.party1.init_key.r_out_share = {core::Z2n<64>(0)};
  out.party0.frac_bits = params.frac_bits;
  out.party1.frac_bits = params.frac_bits;
  out.party0.nr_iters = params.nr_iters;
  out.party1.nr_iters = params.nr_iters;
  return out;
}

inline mpc::AddShare<core::Z2n<64>> eval_reciprocal_gate(
    const ReciprocalKey& k,
    int party,
    net::Chan& ch,
    const pfss::Backend<pfss::CoeffPayload>& backend,
    const pfss::PublicParams& pp,
    uint64_t x_hat_public) {
  // x share from mask
  mpc::AddShare<core::Z2n<64>> x_share =
      (party == 0) ? mpc::AddShare<core::Z2n<64>>{core::Z2n<64>(x_hat_public - k.init_key.r_in_share.s.v)}
                   : mpc::AddShare<core::Z2n<64>>{core::Z2n<64>(0ull - k.init_key.r_in_share.s.v)};

  auto y_share = eval_piecewise_poly(k.init_key, party, ch, backend, pp, x_hat_public);

  int fb = k.frac_bits;
  size_t tri_pos = 0;
  for (int iter = 0; iter < k.nr_iters; ++iter) {
    if (tri_pos + 1 >= k.nr_triples.size()) break;
    auto xy = mul_rescale_share(party, ch, x_share, y_share, k.nr_triples[tri_pos], fb);
    ++tri_pos;
    core::Z2n<64> two_const((party == 0) ? (uint64_t(2) << fb) : 0ull);
    mpc::AddShare<core::Z2n<64>> two_minus_xy{core::Z2n<64>(two_const.v - xy.s.v)};
    auto update = mul_rescale_share(party, ch, y_share, two_minus_xy, k.nr_triples[tri_pos], fb);
    ++tri_pos;
    y_share = update;
  }

  y_share.s += k.out_mask.s;
  return y_share;
}

inline int64_t ref_reciprocal_fixed(
    const PiecewisePolySpec& init_spec,
    int64_t x_fixed,
    int frac_bits,
    int nr_iters) {
  int64_t y = eval_piecewise_poly_ref(init_spec, x_fixed);
  for (int i = 0; i < nr_iters; ++i) {
    int64_t xy = mul_rescale_ring(y, x_fixed, frac_bits);
    int64_t two = static_cast<int64_t>(uint64_t(2) << frac_bits);
    int64_t two_minus_xy = two - xy;
    y = mul_rescale_ring(y, two_minus_xy, frac_bits);
  }
  return y;
}

}  // namespace gates
