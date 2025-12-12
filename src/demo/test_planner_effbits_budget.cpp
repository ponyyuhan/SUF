#include <cassert>
#include <cstdint>
#include <random>
#include <thread>
#include <atomic>
#include <string>
#include <sstream>
#include <vector>

#include "proto/reference_backend.hpp"
#include "runtime/phase_executor.hpp"
#include "runtime/pfss_phase_planner.hpp"
#include "runtime/phase_tasks.hpp"
#include "gates/nexp_composite.hpp"
#include "mpc/local_chan.hpp"

namespace {

struct RunError {
  std::atomic<bool> fail{false};
  std::string err;
};

}  // namespace

// Regression: eff_bits should propagate into PFSS planner cost accounting (cost_effbits),
// allowing tighter budgets than the naive 64-bit-per-element assumption.
int main() {
  const int fb = 8;
  const size_t N = 256;

  std::mt19937_64 rng(123);
  proto::ReferenceBackend keygen_be;
  gates::NExpGateParams nexp_params{fb, /*segments=*/16};
  auto nexp_mat = gates::dealer_make_nexp_task_material(keygen_be, nexp_params, rng, /*triple_need=*/0, /*batch_N=*/N);
  auto nexp_bundle = runtime::CubicPolyBundle{};
  nexp_bundle.suf = &nexp_mat.suf;
  nexp_bundle.key0 = &nexp_mat.keys.k0;
  nexp_bundle.key1 = &nexp_mat.keys.k1;
  nexp_bundle.trunc_f = &nexp_mat.trunc_f;
  nexp_bundle.trunc_2f = &nexp_mat.trunc_2f;
  nexp_bundle.frac_bits = fb;

  int eff_bits = nexp_mat.keys.k0.compiled.pred.eff_bits;
  if (eff_bits <= 0 || eff_bits > 64) {
    throw std::runtime_error("eff_bits hint missing/invalid");
  }
  // This test is only meaningful when we are actually packing below 64 bits.
  assert(eff_bits < 64);

  runtime::PfssLayerPlanner::Limits lim;
  lim.max_phases = 2;
  lim.max_coeff_jobs = 1ull << 16;
  lim.max_trunc_jobs = 1ull << 16;
  lim.max_coeff_flushes = 1ull << 10;
  lim.max_trunc_flushes = 1ull << 10;
  lim.max_coeff_hatx_words = 1ull << 16;
  lim.max_trunc_hatx_words = 1ull << 16;
  lim.max_coeff_hatx_bytes = lim.max_coeff_hatx_words * sizeof(uint64_t);
  lim.max_trunc_hatx_bytes = lim.max_trunc_hatx_words * sizeof(uint64_t);
  lim.max_coeff_active_elems = 1ull << 28;
  lim.max_trunc_active_elems = 1ull << 28;
  lim.max_coeff_cost_effbits = N * static_cast<size_t>(eff_bits) + 64;
  lim.max_trunc_cost_effbits = 1ull << 32;

  std::vector<uint64_t> x_plain(N, 0);
  for (auto& v : x_plain) {
    v = static_cast<uint64_t>(rng() % (1ull << fb));
  }
  std::vector<uint64_t> x0 = x_plain;
  std::vector<uint64_t> x1(N, 0ull);
  std::vector<uint64_t> y0(N, 0ull);
  std::vector<uint64_t> y1(N, 0ull);

  mpc::net::LocalChan::Shared sh;
  mpc::net::LocalChan c0(&sh, /*is_party0=*/true);
  mpc::net::LocalChan c1(&sh, /*is_party0=*/false);

  RunError run_err;
  auto record_err = [&](const std::exception& e,
                        const runtime::PfssLayerPlanner& p0,
                        const runtime::PfssLayerPlanner& p1) {
    run_err.fail.store(true, std::memory_order_relaxed);
    std::ostringstream oss;
    oss << e.what();
    const auto t0 = p0.totals();
    const auto t1 = p1.totals();
    oss << " | p0 coeff_cost_effbits=" << t0.coeff_cost_effbits << "/" << lim.max_coeff_cost_effbits
        << " p1 coeff_cost_effbits=" << t1.coeff_cost_effbits << "/" << lim.max_coeff_cost_effbits;
    run_err.err = oss.str();
  };

  auto run_party = [&](int party,
                       mpc::net::LocalChan& net,
                       runtime::PfssLayerPlanner& planner,
                       proto::ReferenceBackend& be,
                       const std::vector<uint64_t>& x,
                       std::vector<uint64_t>& y) {
    runtime::PhaseExecutor pe;
    pe.set_lazy_mode(false);
    pe.set_keep_batches(true);
    pe.begin_phase(runtime::PhaseExecutor::Phase::kSoftmax);

    runtime::PhaseResources R{};
    R.party = party;
    R.net_chan = &net;
    runtime::ProtoChanFromNet pch(net);
    R.pfss_backend = &be;
    R.pfss_chan = &pch;
    R.pfss_coeff = &pe.pfss_coeff_batch();
    R.pfss_trunc = &pe.pfss_trunc_batch();
    R.opens = &pe.open_collector();

    planner.set_limits(lim);
    planner.begin_layer();
    planner.enter_phase();

    pe.add_task(std::make_unique<runtime::CubicPolyTask>(
        nexp_bundle,
        std::span<const uint64_t>(x.data(), x.size()),
        std::span<uint64_t>(y.data(), y.size())));
    pe.run(R);
    planner.finalize_layer(party, be, pe.pfss_coeff_batch(), pe.pfss_trunc_batch(), pch);
  };

  runtime::PfssLayerPlanner planner0, planner1;
  proto::ReferenceBackend be0, be1;

  std::thread th([&] {
    try {
      run_party(1, c1, planner1, be1, x1, y1);
    } catch (const std::exception& e) {
      record_err(e, planner0, planner1);
    }
  });
  try {
    run_party(0, c0, planner0, be0, x0, y0);
  } catch (const std::exception& e) {
    record_err(e, planner0, planner1);
  }
  th.join();

  if (run_err.fail.load()) {
    throw std::runtime_error("planner eff_bits budget regression: " + run_err.err);
  }

  // Tight eff_bits budget should hold on both parties.
  assert(planner0.totals().coeff_cost_effbits <= lim.max_coeff_cost_effbits);
  assert(planner1.totals().coeff_cost_effbits <= lim.max_coeff_cost_effbits);
  return 0;
}

