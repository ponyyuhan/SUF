#pragma once

#include <random>
#include "gates/piecewise_poly.hpp"
#include "gates/tables/silu_spline_table.hpp"
#include "pfss/backend_cleartext.hpp"
#include "pfss/pfss.hpp"

namespace gates {

struct SiLUGateParams {
  int frac_bits = 16;
  int segments = 16;
};

inline PiecewisePolySpec make_silu_spec(const SiLUGateParams& p) {
  return make_silu_spline_spec(p.frac_bits, p.segments);
}

template<typename CoeffBackend>
inline PiecewisePolyKeys dealer_make_silu_keys(
    CoeffBackend& backend,
    const pfss::PublicParams& pp_coeff,
    const SiLUGateParams& params,
    std::mt19937_64& rng) {
  auto spec = make_silu_spec(params);
  return dealer_make_piecewise_poly_keys(backend, pp_coeff, spec, rng);
}

inline mpc::AddShare<core::Z2n<64>> eval_silu_gate(
    const PiecewisePolyKey& k,
    int party,
    net::Chan& ch,
    const pfss::Backend<pfss::CoeffPayload>& backend,
    const pfss::PublicParams& pp,
    uint64_t x_hat) {
  return eval_piecewise_poly(k, party, ch, backend, pp, x_hat);
}

inline int64_t ref_silu_fixed(const PiecewisePolySpec& spec, int64_t x_fixed) {
  return eval_piecewise_poly_ref(spec, x_fixed);
}

}  // namespace gates
