#include <atomic>
#include <cstdint>
#include <random>
#include <thread>
#include <vector>
#include <iostream>
#include <condition_variable>
#include <queue>

#include "nn/transformer_layer.hpp"
#include "proto/reference_backend.hpp"
#include "runtime/phase_executor.hpp"
#include "runtime/pfss_async_runner.hpp"

namespace {

struct LocalChan : net::Chan {
  struct Shared {
    std::mutex m;
    std::condition_variable cv;
    std::queue<uint64_t> q0to1, q1to0;
    std::atomic<size_t> sent0to1{0};
    std::atomic<size_t> sent1to0{0};
  };
  Shared* sh = nullptr;
  bool is0 = false;
  LocalChan(Shared* s, bool p) : sh(s), is0(p) {}
  void send_u64(uint64_t v) override {
    std::unique_lock<std::mutex> lk(sh->m);
    auto& q = is0 ? sh->q0to1 : sh->q1to0;
    q.push(v);
    if (is0) {
      sh->sent0to1.fetch_add(1, std::memory_order_relaxed);
    } else {
      sh->sent1to0.fetch_add(1, std::memory_order_relaxed);
    }
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

int main() {
  using namespace nn;
  using runtime::PhaseExecutor;

  TransformerConfig cfg;
  cfg.frac_bits = 8;
  cfg.attn.D = 2;
  cfg.attn.H = 1;
  cfg.attn.Dh = 2;
  cfg.attn.S_max = 2;
  cfg.mlp.Hidden = 2;

  const size_t B = 1, T = 1, D = cfg.attn.D;
  std::vector<int64_t> X_plain(B * T * D, 1);
  std::vector<int64_t> Wqkv(D * 3 * D, 1);
  std::vector<int64_t> Wout(D * D, 1);
  std::vector<int64_t> W1(D * cfg.mlp.Hidden, 1);
  std::vector<int64_t> W2(cfg.mlp.Hidden * D, 1);

  // Shares: party0 holds the value, party1 holds zero for simplicity.
  std::vector<uint64_t> X0(X_plain.size(), 0), X1(X_plain.size(), 0);
  for (size_t i = 0; i < X_plain.size(); ++i) {
    X0[i] = proto::add_mod(static_cast<uint64_t>(X_plain[i] << cfg.frac_bits), 0);
    X1[i] = 0;
  }
  std::vector<uint64_t> Y0(B * T * D, 0), Y1(B * T * D, 0);

  KVCache cache0(B, cfg.attn.H, cfg.attn.S_max, cfg.attn.Dh);
  KVCache cache1(B, cfg.attn.H, cfg.attn.S_max, cfg.attn.Dh);

  proto::ReferenceBackend be0, be1;
  compiler::TruncationPassContext trunc0(be0, 0xfeedfaceull);
  compiler::TruncationPassContext trunc1(be1, 0xfeedfaceull);
  LayerContext ctx0, ctx1;
  ctx0.trunc_ctx = &trunc0;
  ctx1.trunc_ctx = &trunc1;
  ctx0.frac_bits = cfg.frac_bits;
  ctx1.frac_bits = cfg.frac_bits;

  // Dedicated PFSS channels to allow async overlap.
  LocalChan::Shared sh_net, sh_pfss;
  LocalChan c0(&sh_net, true), c1(&sh_net, false);
  LocalChan pfss0(&sh_pfss, true), pfss1(&sh_pfss, false);
  ctx0.pfss_net_chan = &pfss0;
  ctx1.pfss_net_chan = &pfss1;

  PhaseExecutor pe0, pe1;

  std::atomic<bool> fail{false};
  std::string err;
  auto record_err = [&](const std::exception& e) {
    fail.store(true, std::memory_order_relaxed);
    err = e.what();
  };

  std::thread t1([&] {
    try {
      transformer_layer_forward(cfg,
                                /*party=*/1,
                                c1,
                                view3(X1.data(), B, T, D),
                                view2(Wqkv.data(), D, 3 * D),
                                view2(Wout.data(), D, D),
                                view2(W1.data(), D, cfg.mlp.Hidden),
                                view2(W2.data(), cfg.mlp.Hidden, D),
                                cache1,
                                view3(Y1.data(), B, T, D),
                                &ctx1,
                                &pe1);
    } catch (const std::exception& e) {
      record_err(e);
    }
  });

  try {
    transformer_layer_forward(cfg,
                              /*party=*/0,
                              c0,
                              view3(X0.data(), B, T, D),
                              view2(Wqkv.data(), D, 3 * D),
                              view2(Wout.data(), D, D),
                              view2(W1.data(), D, cfg.mlp.Hidden),
                              view2(W2.data(), cfg.mlp.Hidden, D),
                              cache0,
                              view3(Y0.data(), B, T, D),
                              &ctx0,
                              &pe0);
  } catch (const std::exception& e) {
    record_err(e);
  }

  t1.join();
  if (fail.load()) {
    std::cerr << "transformer async failed: " << err << "\n";
    return 1;
  }

  if (!ctx0.allow_async_pfss || !ctx1.allow_async_pfss) {
    std::cerr << "expected allow_async_pfss\n";
    return 1;
  }

  // Async runners should have flushed batches; stats must exist.
  if (!ctx0.pfss_async_runner || !ctx1.pfss_async_runner) {
    std::cerr << "pfss_async_runner not created\n";
    return 1;
  }
  auto s0 = ctx0.pfss_async_runner->take_stats();
  auto s1 = ctx1.pfss_async_runner->take_stats();
  if (!s0 || !s1) {
    std::cerr << "async stats missing\n";
    return 1;
  }

  // Basic share shape check.
  if (Y0.size() != Y1.size()) {
    std::cerr << "output size mismatch\n";
    return 1;
  }
  return 0;
}
