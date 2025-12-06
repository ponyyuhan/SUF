#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <thread>

#include "proto/reference_backend.hpp"
#include "runtime/phase_tasks.hpp"
#include "mpc/net.hpp"
#include "proto/pfss_backend.hpp"
#include "proto/pfss_backend_batch.hpp"
#include "proto/pfss_utils.hpp"
#include "runtime/open_collector.hpp"

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

void ensure_triples(compiler::TruncationLoweringResult& bundle, size_t need, std::mt19937_64& rng) {
  auto fill = [&](std::vector<proto::BeaverTriple64Share>& dst0,
                  std::vector<proto::BeaverTriple64Share>& dst1) {
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
  fill(bundle.keys.k0.triples, bundle.keys.k1.triples);
}

}  // namespace

// Minimal harness to exercise TruncTask directly and check it behaves like x >> f.
int main() {
  const int frac_bits = 16;
  const size_t N = 4;

  // Simple inputs in Q16.
  std::vector<uint64_t> x = {
      static_cast<uint64_t>(1ull << (frac_bits + 3)),
      static_cast<uint64_t>(3ull << (frac_bits + 2)),
      static_cast<uint64_t>(5ull << frac_bits),
      static_cast<uint64_t>(7ull << (frac_bits - 1)),
  };

  proto::ReferenceBackend backend;
  std::mt19937_64 rng(12345);
  compiler::GateParams p;
  p.kind = compiler::GateKind::GapARS;
  p.frac_bits = frac_bits;
  auto trunc_bundle = compiler::lower_truncation_gate(backend, rng, p);
  ensure_triples(trunc_bundle, 1024, rng);
  // Zero r_out so we can read raw payload.
  std::fill(trunc_bundle.keys.k0.r_out_share.begin(), trunc_bundle.keys.k0.r_out_share.end(), 0ull);
  std::fill(trunc_bundle.keys.k1.r_out_share.begin(), trunc_bundle.keys.k1.r_out_share.end(), 0ull);

  auto run_party = [&](int party, const std::vector<uint64_t>& in, LocalChan& net_ch) {
    runtime::PhaseExecutor pe;
    runtime::PhaseResources R;
    R.party = party;
    R.pfss_backend = &backend;
    runtime::ProtoChanFromNet pch(net_ch);
    R.pfss_chan = &pch;
    R.net_chan = &net_ch;
    R.pfss = &pe.pfss_batch();
    R.opens = &pe.open_collector();

    std::vector<uint64_t> out(in.size(), 0);
    auto task = std::make_unique<runtime::TruncTask>(&trunc_bundle,
                                                     std::span<const uint64_t>(in.data(), in.size()),
                                                     std::span<uint64_t>(out.data(), out.size()));
    pe.add_task(std::move(task));
    pe.run(R);
    return out;
  };

  // Additive shares: party0 gets input, party1 gets zeros for simplicity.
  LocalChan::Shared sh;
  LocalChan c0(&sh, true), c1(&sh, false);

  std::vector<uint64_t> y0, y1;
  bool party_fail = false;
  std::string party_err;
  std::vector<uint64_t> zero(x.size(), 0);
  std::thread th1([&] {
    try {
      y1 = run_party(1, zero, c1);
    } catch (const std::exception& e) {
      party_fail = true;
      party_err = e.what();
    }
  });
  try {
    y0 = run_party(0, x, c0);
  } catch (const std::exception& e) {
    party_fail = true;
    party_err = e.what();
  }
  th1.join();
  if (party_fail) {
    std::cerr << "party execution failed: " << party_err << "\n";
    return 1;
  }

  bool ok = true;
  for (size_t i = 0; i < N; ++i) {
    uint64_t recon = proto::add_mod(y0[i], y1[i]);
    uint64_t expect = x[i] >> frac_bits;
    if (recon != expect) {
      ok = false;
      std::cerr << "trunc mismatch idx " << i << " got " << recon << " expected " << expect << "\n";
    }
  }

  if (ok) {
    std::cout << "trunc task ok\n";
    return 0;
  }
  return 1;
}
