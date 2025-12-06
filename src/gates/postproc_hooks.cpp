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

void HornerCubicHook::run_batch(int party,
                                proto::IChannel& ch,
                                proto::BeaverMul64& mul,
                                const uint64_t* hatx_public,
                                const uint64_t* arith_share_in,
                                size_t arith_stride,
                                const uint64_t*,
                                size_t,
                                size_t N,
                                uint64_t* haty_share_out) const {
  if (arith_stride < 4) return;
  std::vector<uint64_t> x_share(N, 0);
  for (size_t i = 0; i < N; ++i) {
    uint64_t hx = hatx_public ? hatx_public[i] : 0ull;
    x_share[i] = (party == 0) ? proto::sub_mod(hx, r_in_share) : proto::sub_mod(0ull, r_in_share);
  }

  std::vector<uint64_t> t1(N, 0);
  for (size_t i = 0; i < N; ++i) {
    const uint64_t* coeff = arith_share_in + i * arith_stride;
    t1[i] = mul.mul(coeff[3], x_share[i]);
  }

  std::vector<uint64_t> t2(N, 0);
  for (size_t i = 0; i < N; ++i) {
    const uint64_t* coeff = arith_share_in + i * arith_stride;
    uint64_t sum = proto::add_mod(t1[i], coeff[2]);
    t2[i] = mul.mul(sum, x_share[i]);
  }

  std::vector<uint64_t> t3(N, 0);
  for (size_t i = 0; i < N; ++i) {
    const uint64_t* coeff = arith_share_in + i * arith_stride;
    uint64_t sum = proto::add_mod(t2[i], coeff[1]);
    t3[i] = mul.mul(sum, x_share[i]);
  }

  for (size_t i = 0; i < N; ++i) {
    const uint64_t* coeff = arith_share_in + i * arith_stride;
    uint64_t acc = proto::add_mod(t3[i], coeff[0]);
    haty_share_out[i] = acc;
  }
}

}  // namespace gates
