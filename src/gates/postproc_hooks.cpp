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
  const auto& trunc = ensure_trunc_bundle();
  const auto& tkey = (party == 0) ? trunc.keys.k0 : trunc.keys.k1;
  gates::PostProcHook* trunc_hook = (party == 0) ? trunc.hook0.get() : trunc.hook1.get();
  if (!trunc_hook) {
    throw std::runtime_error("HornerCubicHook: missing truncation hook");
  }

  auto do_trunc = [&](const std::vector<uint64_t>& in_shares,
                      std::vector<uint64_t>& out_shares) {
    out_shares.resize(in_shares.size());
    std::vector<uint64_t> masked(in_shares.size());
    for (size_t i = 0; i < in_shares.size(); ++i) {
      masked[i] = proto::add_mod(in_shares[i], tkey.r_in_share);
    }
    std::vector<uint64_t> other(masked.size(), 0);
    size_t byte_len = masked.size() * sizeof(uint64_t);
    if (party == 0) {
      ch.send_bytes(masked.data(), byte_len);
      ch.recv_bytes(other.data(), byte_len);
    } else {
      ch.recv_bytes(other.data(), byte_len);
      ch.send_bytes(masked.data(), byte_len);
    }
    std::vector<uint64_t> hat(masked.size());
    for (size_t i = 0; i < masked.size(); ++i) {
      hat[i] = proto::add_mod(masked[i], other[i]);
    }
    nn::TensorView<uint64_t> out_view;
    out_view.data = out_shares.data();
    out_view.dims = 1;
    out_view.shape[0] = out_shares.size();
    runtime::PfssSuperBatch trunc_batch;
    trunc_batch.enqueue_truncation(trunc, tkey, *trunc_hook, std::move(hat), out_view);
    trunc_batch.flush_and_finalize(party, *backend, ch);
    trunc_batch.clear();
  };

  std::vector<uint64_t> x_share(N, 0);
  for (size_t i = 0; i < N; ++i) {
    uint64_t hx = hatx_public ? hatx_public[i] : 0ull;
    x_share[i] = (party == 0) ? proto::sub_mod(hx, r_in_share) : proto::sub_mod(0ull, r_in_share);
  }

  std::vector<uint64_t> t1(N, 0), t1_trunc;
  for (size_t i = 0; i < N; ++i) {
    const uint64_t* coeff = arith_share_in + i * arith_stride;
    t1[i] = mul.mul(coeff[3], x_share[i]);
  }
  do_trunc(t1, t1_trunc);

  std::vector<uint64_t> t2(N, 0), t2_trunc;
  for (size_t i = 0; i < N; ++i) {
    const uint64_t* coeff = arith_share_in + i * arith_stride;
    uint64_t sum = proto::add_mod(t1_trunc[i], coeff[2]);
    t2[i] = mul.mul(sum, x_share[i]);
  }
  do_trunc(t2, t2_trunc);

  std::vector<uint64_t> t3(N, 0), t3_trunc;
  for (size_t i = 0; i < N; ++i) {
    const uint64_t* coeff = arith_share_in + i * arith_stride;
    uint64_t sum = proto::add_mod(t2_trunc[i], coeff[1]);
    t3[i] = mul.mul(sum, x_share[i]);
  }
  do_trunc(t3, t3_trunc);

  for (size_t i = 0; i < N; ++i) {
    const uint64_t* coeff = arith_share_in + i * arith_stride;
    uint64_t acc = proto::add_mod(t3_trunc[i], coeff[0]);
    haty_share_out[i] = acc;
  }
}

}  // namespace gates
