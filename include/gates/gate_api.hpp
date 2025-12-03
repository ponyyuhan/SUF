#pragma once

#include <vector>
#include "compiler/suf_to_pfss.hpp"
#include "mpc/masked_wire.hpp"
#include "mpc/net.hpp"
#include "suf/polynomial.hpp"

namespace gates {

template<typename RingT>
struct GateEvalResult {
  std::vector<mpc::AddShare<RingT>> y_hat_shares;  // one per arithmetic output
};

// Generic “compiled SUF gate evaluation”: 2 PFSS evals + Horner + add r_out.
// triples should contain at least r_out * degree Beaver triples (degree per output).
template<typename RingT, typename PredPayloadT, typename CoeffPayloadT>
inline GateEvalResult<RingT> eval_compiled_suf_gate(
    int party,
    net::Chan& ch,
    const pfss::Backend<PredPayloadT>& pred_backend,
    const pfss::Backend<CoeffPayloadT>& coeff_backend,
    const pfss::PublicParams& pp_pred,
    const pfss::PublicParams& pp_coeff,
    const compiler::CompiledSUFKeys& k,
    uint64_t x_hat_public,
    mpc::AddShare<RingT> x_share,  // if not available, derive from masked->shares using r_in_share
    const std::vector<mpc::BeaverTripleA<RingT>>& triples) {
  const pfss::Key& pred_key = (party == 0) ? k.pred_key0 : k.pred_key1;
  const pfss::Key& coeff_key = (party == 0) ? k.coeff_key0 : k.coeff_key1;

  PredPayloadT pred_payload = pred_backend.eval(party, pp_pred, pred_key, x_hat_public);
  (void)pred_payload;  // placeholder: predicates are available for helper bits if needed.
  CoeffPayloadT coeff_payload = coeff_backend.eval(party, pp_coeff, coeff_key, x_hat_public);

  int coeffs_per_poly = k.degree + 1;
  size_t expected = static_cast<size_t>(k.r_out) * static_cast<size_t>(coeffs_per_poly);
  if (coeff_payload.size() < expected) {
    return {{}};  // malformed payload
  }

  GateEvalResult<RingT> res;
  res.y_hat_shares.resize(static_cast<size_t>(k.r_out));

  for (int j = 0; j < k.r_out; ++j) {
    suf::Poly<RingT> poly;
    poly.coeffs.resize(static_cast<size_t>(coeffs_per_poly));
    size_t base = static_cast<size_t>(j * coeffs_per_poly);
    for (int i = 0; i < coeffs_per_poly; ++i) {
      poly.coeffs[static_cast<size_t>(i)] = RingT(coeff_payload[base + static_cast<size_t>(i)]);
    }

    size_t t_base = static_cast<size_t>(j * k.degree);
    if (k.degree > 0 && triples.size() < t_base + static_cast<size_t>(k.degree)) {
      return {{}};
    }
    std::vector<mpc::BeaverTripleA<RingT>> triples_slice;
    triples_slice.reserve(static_cast<size_t>(k.degree));
    for (int t = 0; t < k.degree; ++t) {
      triples_slice.push_back(triples[t_base + static_cast<size_t>(t)]);
    }

    auto y_share = (k.degree == 0)
        ? mpc::AddShare<RingT>{poly.coeffs[0]}
        : suf::eval_poly_horner_shared(party, ch, poly, x_share, triples_slice);

    uint64_t r_out_b = (party == 0) ? k.r_out_share0[static_cast<size_t>(j)]
                                    : k.r_out_share1[static_cast<size_t>(j)];
    y_share.s += RingT(r_out_b);
    res.y_hat_shares[static_cast<size_t>(j)] = y_share;
  }

  return res;
}

}  // namespace gates
