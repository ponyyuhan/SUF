#pragma once

#include "gates/ars_faithful_gate.hpp"

namespace gates {

// GapARS placeholder: uses the faithful ARS path until range analysis proves
// the gap certificate and a cheaper evaluator is wired in.
using GapArsParams = ArsParams;
using GapArsKey = ArsKey;
using GapArsKeys = ArsKeys;

inline GapArsKeys dealer_make_gapars_keys(const GapArsParams& params,
                                          size_t n,
                                          std::mt19937_64& rng) {
  return dealer_make_ars_keys(params, n, rng);
}

inline std::vector<mpc::AddShare<core::Z2n<64>>> eval_gapars(
    const GapArsKey& k,
    int party,
    net::Chan& ch,
    const std::vector<mpc::AddShare<core::Z2n<64>>>& xs) {
  return eval_ars_faithful(k, party, ch, xs);
}

}  // namespace gates

