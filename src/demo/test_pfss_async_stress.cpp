#include <cassert>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "compiler/truncation_lowering.hpp"
#include "proto/reference_backend.hpp"
#include "runtime/pfss_phase_planner.hpp"
#include "runtime/pfss_async_runner.hpp"
#include "nn/tensor_view.hpp"

// Dummy proto channel: ignores payload, fills zeros on recv.
struct NullProtoChan : proto::IChannel {
  void send_bytes(const void*, size_t) override {}
  void recv_bytes(void* data, size_t n) override { std::memset(data, 0, n); }
};

int main() {
  using namespace runtime;
  proto::ReferenceBackend backend;
  PfssSuperBatch coeff, trunc;
  PfssLayerPlanner planner;
  PfssLayerPlanner::Limits lim;
  lim.max_coeff_jobs = 32;
  lim.max_trunc_jobs = 32;
  lim.max_coeff_flushes = 8;
  lim.max_trunc_flushes = 8;
  lim.max_coeff_hatx_words = 1ull << 20;
  lim.max_trunc_hatx_words = 1ull << 20;
  planner.set_limits(lim);
  planner.begin_layer();

  // Build a trunc bundle and enqueue multiple jobs to stress batching.
  compiler::GateParams p;
  p.kind = compiler::GateKind::GapARS;
  p.frac_bits = 8;
  std::mt19937_64 rng(42);
  auto bundle = compiler::lower_truncation_gate(backend, rng, p, /*N=*/4);
  // Zero masks for determinism.
  if (!bundle.keys.k0.r_out_share.empty()) bundle.keys.k0.r_out_share.assign(bundle.keys.k0.r_out_share.size(), 0);
  if (!bundle.keys.k1.r_out_share.empty()) bundle.keys.k1.r_out_share.assign(bundle.keys.k1.r_out_share.size(), 0);

  constexpr size_t kJobs = 8;
  constexpr size_t kTruncJobs = 4;
  constexpr size_t kElems = 4;
  for (size_t i = 0; i < kJobs; ++i) {
    runtime::PreparedCompositeJob job;
    job.suf = &bundle.suf;
    job.key = &bundle.keys.k0;
    job.hook = bundle.hook0.get();
    job.hatx_public = {1ull << p.frac_bits};
    coeff.enqueue_composite(std::move(job));
  }
  std::vector<uint64_t> trunc_in(kElems, 1ull << p.frac_bits);
  std::vector<std::vector<uint64_t>> trunc_out(kTruncJobs, std::vector<uint64_t>(kElems, 0));
  for (size_t i = 0; i < kTruncJobs; ++i) {
    trunc.enqueue_truncation(bundle,
                             bundle.keys.k0,
                             *bundle.hook0,
                             trunc_in,
                             nn::TensorView<uint64_t>(trunc_out[i].data(), {trunc_out[i].size()}));
  }

  NullProtoChan nch;
  PfssAsyncRunner runner;
  planner.finalize_layer(/*party=*/0,
                         backend,
                         coeff,
                         trunc,
                         nch,
                         &runner,
                         /*wait=*/false,
                         /*chan_mu=*/nullptr);

  auto stats_opt = runner.take_stats();
  assert(stats_opt.has_value());
  auto stats = *stats_opt;
  // All jobs flushed in one go; totals tracked by planner.
  assert(stats.coeff.jobs == kJobs);
  assert(stats.has_trunc);
  assert(stats.trunc.jobs == kTruncJobs);
  assert(planner.totals().coeff_jobs == kJobs);
  assert(planner.totals().trunc_jobs == kTruncJobs);
  assert(planner.totals().coeff_flushes <= lim.max_coeff_flushes);
  assert(planner.totals().trunc_flushes <= lim.max_trunc_flushes);
  return 0;
}
