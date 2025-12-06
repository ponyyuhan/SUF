#include "gates/postproc_hooks.hpp"

#include "compiler/truncation_lowering.hpp"
#include "proto/pfss_backend_batch.hpp"
#include "proto/pfss_utils.hpp"
#include "runtime/pfss_superbatch.hpp"

namespace gates {

HornerCubicHook::~HornerCubicHook() {
  delete trunc_bundle;
  trunc_bundle = nullptr;
}

const compiler::TruncationLoweringResult& HornerCubicHook::ensure_trunc_bundle() const {
  if (!backend) {
    throw std::runtime_error("HornerCubicHook: backend not set");
  }
  if (!trunc_bundle) {
    compiler::GateParams p;
    p.kind = compiler::GateKind::FaithfulARS;
    p.frac_bits = trunc_frac_bits > 0 ? trunc_frac_bits : frac_bits;
    trunc_bundle = new compiler::TruncationLoweringResult(
        compiler::lower_truncation_gate(*backend, rng, p));
    auto fill = [&](std::vector<proto::BeaverTriple64Share>& dst0,
                    std::vector<proto::BeaverTriple64Share>& dst1,
                    size_t need) {
      while (dst0.size() < need || dst1.size() < need) {
        uint64_t a = rng();
        uint64_t b = rng();
        uint64_t c = proto::mul_mod(a, b);
        uint64_t a0 = rng();
        uint64_t a1 = a - a0;
        uint64_t b0 = rng();
        uint64_t b1 = b - b0;
        uint64_t c0 = rng();
        uint64_t c1 = c - c0;
        dst0.push_back({a0, b0, c0});
        dst1.push_back({a1, b1, c1});
      }
    };
    // Provide a generous stash so hook-only paths don't exhaust triples.
    constexpr size_t kDefaultTriples = 4096;
    fill(trunc_bundle->keys.k0.triples, trunc_bundle->keys.k1.triples, kDefaultTriples);
  }
  return *trunc_bundle;
}

void HornerCubicHook::run_batch(int,
                                proto::IChannel&,
                                proto::BeaverMul64&,
                                const uint64_t*,
                                const uint64_t*,
                                size_t,
                                const uint64_t*,
                                size_t,
                                size_t,
                                uint64_t*) const {
  throw std::runtime_error("HornerCubicHook is disabled; use task-based cubic evaluation with truncation");
}

}  // namespace gates
