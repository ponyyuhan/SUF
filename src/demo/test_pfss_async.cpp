#include <cassert>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "compiler/truncation_lowering.hpp"
#include "proto/reference_backend.hpp"
#include "runtime/pfss_phase_planner.hpp"
#include "runtime/pfss_async_runner.hpp"

// Dummy proto channel: ignores payload, fills zeros on recv.
struct NullProtoChan : proto::IChannel {
  void send_bytes(const void*, size_t) override {}
  void recv_bytes(void* data, size_t n) override { std::memset(data, 0, n); }
};

int main() {
  using namespace runtime;
  proto::ReferenceBackend backend;
  PfssSuperBatch pfss;
  PfssAsyncRunner runner;
  PfssLayerPlanner planner;
  planner.begin_layer();

  // One-word truncation job staged directly into the batch.
  compiler::GateParams p;
  p.kind = compiler::GateKind::GapARS;
  p.frac_bits = 8;
  std::mt19937_64 rng(123);
  auto bundle = compiler::lower_truncation_gate(backend, rng, p, /*N=*/1);
  // Zero masks to make outputs deterministic to inspect.
  if (!bundle.keys.k0.r_out_share.empty()) bundle.keys.k0.r_out_share[0] = 0;
  if (!bundle.keys.k1.r_out_share.empty()) bundle.keys.k1.r_out_share[0] = 0;

  runtime::PreparedCompositeJob job;
  job.suf = &bundle.suf;
  job.key = &bundle.keys.k0;
  job.hook = bundle.hook0.get();
  job.hatx_public = {1ull << p.frac_bits};  // x = 1.0 in fixed-point
  auto handle = pfss.enqueue_composite(std::move(job));

  NullProtoChan nch;
  planner.finalize_layer(/*party=*/0,
                         backend,
                         pfss,
                         pfss,
                         nch,
                         &runner,
                         /*wait=*/false,
                         /*chan_mu=*/nullptr);

  // Join async flush and pull stats.
  auto stats_opt = runner.take_stats();
  assert(stats_opt.has_value());
  auto stats = *stats_opt;
  assert(stats.coeff.jobs == 1);
  assert(stats.coeff.flushes == 1);

  // Handle should remain valid after batch clear.
  assert(handle.slot && handle.slot->ready.load());
  assert(handle.slot->arith_storage && !handle.slot->arith_storage->empty());
  // Ensure some output was produced; we rely on slot lifetime after async clear.
  uint64_t y = handle.slot->arith_storage->at(0);
  (void)y;
  return 0;
}
