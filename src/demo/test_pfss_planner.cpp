#include <cassert>
#include <cstring>
#include <iostream>

#include "proto/reference_backend.hpp"
#include "runtime/pfss_phase_planner.hpp"
#include "runtime/pfss_async_runner.hpp"

// Dummy channel that discards bytes; safe because we never enqueue PFSS jobs in this test.
struct NullChan : proto::IChannel {
  void send_bytes(const void*, size_t) override {}
  void recv_bytes(void* data, size_t n) override {
    // Zero out receive buffer to keep callers happy.
    std::memset(data, 0, n);
  }
};

int main() {
  using namespace runtime;
  proto::ReferenceBackend backend;
  PfssSuperBatch coeff, trunc;
  PfssAsyncRunner async_runner;
  NullChan null_ch;

  PfssLayerPlanner::Limits lim;
  lim.max_phases = 2;
  PfssLayerPlanner layer;
  layer.set_limits(lim);
  layer.begin_layer();
  layer.enter_phase();
  layer.enter_phase();  // exactly at phase limit

  // Barrier with no pending work should be a no-op.
  PfssLayerPlanner::BarrierPolicy pol;
  pol.drain_all = true;
  layer.barrier(/*party=*/0, backend, coeff, trunc, null_ch, nullptr, nullptr, pol);

  // Finalize layer: no jobs enqueued, should not throw and should clear batches.
  layer.finalize_layer(/*party=*/0, backend, coeff, trunc, null_ch, &async_runner, /*wait=*/true);

  const auto& totals = layer.totals();
  assert(totals.phases == 2);
  assert(totals.coeff_jobs == 0 && totals.trunc_jobs == 0);
  std::cout << "planner totals phases=" << totals.phases << " ok\n";
  return 0;
}
