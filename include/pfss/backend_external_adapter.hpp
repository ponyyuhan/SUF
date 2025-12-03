#pragma once

#include "pfss/pfss.hpp"

// This is the only file that should mention a concrete library (Grotto/SIGMA-style).
// The rest of the framework is backend-agnostic.

namespace pfss {

template<typename PayloadT>
struct ExternalBackendAdapter final : Backend<PayloadT> {
  PublicParams setup(int lambda_bits) override {
    return {"EXTERNAL_BACKEND", lambda_bits};
  }

  std::pair<Key, Key> prog_gen(const PublicParams&, const ProgramDesc&) override {
    // TODO: hook up to concrete DPF/DCF library.
    return {};
  }

  PayloadT eval(int,
                const PublicParams&,
                const Key&,
                uint64_t) const override {
    // TODO: hook up to concrete DPF/DCF library.
    return {};
  }

  void eval_batch(int party,
                  const PublicParams& pp,
                  const Key& key,
                  const uint64_t* x_hat,
                  PayloadT* out,
                  size_t count) const override {
    for (size_t i = 0; i < count; i++) out[i] = eval(party, pp, key, x_hat[i]);
  }
};

}  // namespace pfss
