#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <thread>
#include <vector>

#include "compiler/truncation_lowering.hpp"
#include "proto/reference_backend.hpp"
#include "runtime/phase_tasks.hpp"
#include "runtime/pfss_superbatch.hpp"
#include "runtime/staged_executor.hpp"
#include "mpc/net.hpp"
#include "runtime/pfss_superbatch.hpp"
#include "runtime/open_collector.hpp"
#include "proto/channel.hpp"

namespace {

struct LocalChan : net::Chan {
  struct Shared {
    std::mutex m;
    std::condition_variable cv;
    std::queue<uint64_t> q0to1, q1to0;
  };
  Shared* sh = nullptr;
  bool is0 = false;
  LocalChan(Shared* s, bool p) : sh(s), is0(p) {}
  void send_u64(uint64_t v) override {
    std::unique_lock<std::mutex> lk(sh->m);
    auto& q = is0 ? sh->q0to1 : sh->q1to0;
    q.push(v);
    sh->cv.notify_all();
  }
  uint64_t recv_u64() override {
    std::unique_lock<std::mutex> lk(sh->m);
    auto& q = is0 ? sh->q1to0 : sh->q0to1;
    sh->cv.wait(lk, [&] { return !q.empty(); });
    uint64_t v = q.front();
    q.pop();
    return v;
  }
};

}  // namespace

// Demonstrates staged executor: two trunc tasks (cascade) share one PFSS flush.
int main() {
  try {
    const int frac_bits = 8;
    const size_t N = 8;
    std::vector<uint64_t> x(N);
    for (size_t i = 0; i < N; ++i) {
      x[i] = static_cast<uint64_t>((i + 1) << frac_bits);
    }

    proto::ReferenceBackend backend;
    std::mt19937_64 rng(123);
    compiler::GateParams p;
    p.kind = compiler::GateKind::FaithfulTR;
    p.frac_bits = frac_bits;
    auto bundle = compiler::lower_truncation_gate(backend, rng, p, N);
    std::fill(bundle.keys.k0.r_out_share.begin(), bundle.keys.k0.r_out_share.end(), 0ull);
    std::fill(bundle.keys.k1.r_out_share.begin(), bundle.keys.k1.r_out_share.end(), 0ull);

    LocalChan::Shared sh;
    LocalChan c0(&sh, true), c1(&sh, false);

    auto run_party = [&](int party,
                         const std::vector<uint64_t>& in0,
                         const std::vector<uint64_t>& in1,
                         std::vector<uint64_t>& out0,
                         std::vector<uint64_t>& out1,
                         runtime::PfssSuperBatch& coeff_batch,
                         runtime::PfssSuperBatch& trunc_batch,
                         runtime::OpenCollector& opens,
                         LocalChan& net) {
      runtime::StagedExecutor se;
      runtime::PhaseResources R;
      R.party = party;
      runtime::ProtoChanFromNet pch(net);
      R.pfss_backend = &backend;
      R.pfss_chan = &pch;
      R.net_chan = &net;
      R.pfss_coeff = &coeff_batch;
      R.pfss_trunc = &trunc_batch;
      R.opens = &opens;

      se.add_task(std::make_unique<runtime::TruncTask>(
          &bundle,
          std::span<const uint64_t>(in0.data(), in0.size()),
          std::span<uint64_t>(out0.data(), out0.size())));
      se.add_task(std::make_unique<runtime::TruncTask>(
          &bundle,
          std::span<const uint64_t>(in1.data(), in1.size()),
          std::span<uint64_t>(out1.data(), out1.size())));
      se.run(R, backend, pch);
    };

    std::vector<uint64_t> in0 = x;
    std::vector<uint64_t> in1 = x;
    std::vector<uint64_t> out0(N, 0), out1(N, 0), out0_b(N, 0), out1_b(N, 0);
    runtime::PfssSuperBatch coeffA, truncA, coeffB, truncB;
    runtime::OpenCollector opensA, opensB;
    std::thread t0([&] { run_party(0, in0, in1, out0, out1, coeffA, truncA, opensA, c0); });
    std::thread t1([&] { run_party(1, in0, in1, out0_b, out1_b, coeffB, truncB, opensB, c1); });
    t0.join();
    t1.join();

    // Compare against reference logical shift (faithful truncation).
    for (size_t i = 0; i < N; ++i) {
      uint64_t y0 = out0[i] + out0_b[i];
      uint64_t y1 = out1[i] + out1_b[i];
      uint64_t expect0 = in0[i] >> frac_bits;
      if (y0 != expect0 || y1 != expect0) {
        std::cerr << "Mismatch at " << i << " got (" << y0 << "," << y1 << ") expect ("
                  << expect0 << "," << expect0 << ")\n";
        return 1;
      }
    }
    if (truncA.stats().flushes > 1 || truncB.stats().flushes > 1) {
      std::cerr << "Expected single trunc flush per party, got " << truncA.stats().flushes
                << " and " << truncB.stats().flushes << "\n";
      return 1;
    }

    std::cout << "Staged trunc test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Staged trunc test failed: " << e.what() << std::endl;
    return 1;
  }
}
