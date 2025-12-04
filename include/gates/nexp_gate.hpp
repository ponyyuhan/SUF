#pragma once

#include <random>
#include "gates/piecewise_poly.hpp"
#include "gates/tables/nexp_piecewise_table.hpp"
#include "pfss/backend_cleartext.hpp"
#include "pfss/pfss.hpp"

namespace gates {

struct NExpGateParams {
  int frac_bits = 16;
  int segments = 16;
};

inline PiecewisePolySpec make_nexp_spec(const NExpGateParams& p) {
  return make_nexp_piecewise_spec(p.frac_bits, p.segments);
}

template<typename CoeffBackend>
inline PiecewisePolyKeys dealer_make_nexp_keys(
    CoeffBackend& backend,
    const pfss::PublicParams& pp_coeff,
    const NExpGateParams& params,
    std::mt19937_64& rng) {
  auto spec = make_nexp_spec(params);
  return dealer_make_piecewise_poly_keys(backend, pp_coeff, spec, rng);
}

inline mpc::AddShare<core::Z2n<64>> eval_nexp_gate(
    const PiecewisePolyKey& k,
    int party,
    net::Chan& ch,
    const pfss::Backend<pfss::CoeffPayload>& backend,
    const pfss::PublicParams& pp,
    uint64_t x_hat) {
  return eval_piecewise_poly(k, party, ch, backend, pp, x_hat);
}

inline int64_t ref_nexp_fixed(const PiecewisePolySpec& spec, int64_t x_fixed) {
  return eval_piecewise_poly_ref(spec, x_fixed);
}

}  // namespace gates
