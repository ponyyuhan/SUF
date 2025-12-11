#include <cassert>
#include <cstdint>
#include <random>
#include <thread>
#include <vector>
#include <queue>
#include <condition_variable>

#include "nn/attention_block.hpp"
#include "nn/transformer_layer.hpp"
#include "proto/reference_backend.hpp"
#include "runtime/pfss_phase_planner.hpp"

namespace {

// Minimal LocalChan for two-party simulation.
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

void split_shares(const std::vector<uint64_t>& plain, std::mt19937_64& rng,
                  std::vector<uint64_t>& a, std::vector<uint64_t>& b) {
  a.resize(plain.size());
  b.resize(plain.size());
  for (size_t i = 0; i < plain.size(); ++i) {
    uint64_t a0 = rng();
    uint64_t a1 = plain[i] - a0;
    a[i] = a0;
    b[i] = a1;
  }
}

}  // namespace

int main() {
  using namespace nn;
  const int B = 1;
  const int T = 4;
  const int D = 4;
  const int H = 1;
  const int Dh = D / H;
  const int fb = 8;

  TransformerConfig cfg;
  cfg.frac_bits = fb;
  cfg.attn.D = D;
  cfg.attn.Dh = Dh;
  cfg.attn.H = H;
  cfg.attn.S_max = T;
  cfg.attn.frac_bits = fb;
  cfg.attn.legacy_softmax = false;
  cfg.mlp.Hidden = D;
  cfg.mlp.frac_bits = fb;

  std::vector<uint64_t> X(B * T * D, 0);
  for (size_t i = 0; i < X.size(); ++i) X[i] = static_cast<uint64_t>((i + 1) << fb);
  std::vector<int64_t> Wqkv(D * 3 * D, 0), Wout(D * D, 0), W1(D * D, 0), W2(D * D, 0);
  for (size_t i = 0; i < Wqkv.size(); ++i) Wqkv[i] = (i % 7) - 3;
  for (size_t i = 0; i < Wout.size(); ++i) Wout[i] = (i % 5) - 2;
  for (size_t i = 0; i < W1.size(); ++i) W1[i] = (i % 3) - 1;
  for (size_t i = 0; i < W2.size(); ++i) W2[i] = (i % 4) - 2;

  std::mt19937_64 rng(123);
  std::vector<uint64_t> X0, X1;
  split_shares(X, rng, X0, X1);
  std::vector<uint64_t> Y0(T * D, 0), Y1(T * D, 0);
  KVCache cache0(B, H, T, Dh), cache1(B, H, T, Dh);

  proto::ReferenceBackend be0, be1;
  compiler::TruncationPassContext trunc0(be0, 0xabcdu);
  compiler::TruncationPassContext trunc1(be1, 0xabcdu);
  LayerContext ctx0, ctx1;
  ctx0.trunc_ctx = &trunc0;
  ctx1.trunc_ctx = &trunc1;
  ctx0.frac_bits = fb;
  ctx1.frac_bits = fb;
  runtime::PfssLayerPlanner planner0, planner1;
  runtime::PfssLayerPlanner::Limits lim;
  lim.max_coeff_flushes = 128;
  lim.max_trunc_flushes = 128;
  lim.max_coeff_jobs = 2048;
  lim.max_trunc_jobs = 2048;
  lim.max_coeff_hatx_words = 512;
  lim.max_trunc_hatx_words = 512;
  lim.max_coeff_hatx_bytes = lim.max_coeff_hatx_words * sizeof(uint64_t);
  lim.max_trunc_hatx_bytes = lim.max_trunc_hatx_words * sizeof(uint64_t);
  lim.max_phases = 32;
  lim.max_coeff_active_elems = 1ull << 20;
  lim.max_trunc_active_elems = 1ull << 20;
  lim.max_coeff_cost_effbits = 1ull << 24;
  lim.max_trunc_cost_effbits = 1ull << 24;
  planner0.set_limits(lim);
  planner1.set_limits(lim);
  ctx0.pfss_layer_planner = &planner0;
  ctx1.pfss_layer_planner = &planner1;

  LocalChan::Shared sh_main, sh_pfss;
  LocalChan c0(&sh_main, true), c1(&sh_main, false);
  LocalChan pfss0(&sh_pfss, true), pfss1(&sh_pfss, false);
  ctx0.pfss_net_chan = &pfss0;
  ctx1.pfss_net_chan = &pfss1;

  std::atomic<bool> fail{false};
  std::string err;
  auto record_err = [&](const std::exception& e) {
    fail.store(true, std::memory_order_relaxed);
    err = e.what();
  };

  std::thread t1([&] {
    try {
      transformer_layer_forward(cfg, 1, c1,
                                view3(X1.data(), B, T, D),
                                view2(Wqkv.data(), D, 3 * D),
                                view2(Wout.data(), D, D),
                                view2(W1.data(), D, cfg.mlp.Hidden),
                                view2(W2.data(), cfg.mlp.Hidden, D),
                                cache1,
                                view3(Y1.data(), B, T, D),
                                &ctx1);
    } catch (const std::exception& e) {
      record_err(e);
    }
  });
  try {
    transformer_layer_forward(cfg, 0, c0,
                              view3(X0.data(), B, T, D),
                              view2(Wqkv.data(), D, 3 * D),
                              view2(Wout.data(), D, D),
                              view2(W1.data(), D, cfg.mlp.Hidden),
                              view2(W2.data(), cfg.mlp.Hidden, D),
                              cache0,
                              view3(Y0.data(), B, T, D),
                              &ctx0);
  } catch (const std::exception& e) {
    record_err(e);
  }
  t1.join();
  if (fail.load()) {
    throw std::runtime_error("transformer causal planner test: " + err);
  }

  // Reconstruct and sanity-check nonzero outputs.
  for (size_t i = 0; i < Y0.size(); ++i) {
    uint64_t plain = Y0[i] + Y1[i];
    assert(plain != 0);
  }

  // Budget/flush sanity: causal valid_lens=T should respect planner limits.
  assert(planner0.totals().coeff_flushes <= lim.max_coeff_flushes);
  assert(planner0.totals().trunc_flushes <= lim.max_trunc_flushes);
  assert(planner1.totals().coeff_flushes <= lim.max_coeff_flushes);
  assert(planner1.totals().trunc_flushes <= lim.max_trunc_flushes);
  assert(planner0.totals().coeff_jobs <= lim.max_coeff_jobs);
  assert(planner1.totals().coeff_jobs <= lim.max_coeff_jobs);
  assert(planner0.totals().trunc_jobs <= lim.max_trunc_jobs);
  assert(planner1.totals().trunc_jobs <= lim.max_trunc_jobs);
  assert(planner0.totals().coeff_hatx_words <= lim.max_coeff_hatx_words);
  assert(planner1.totals().coeff_hatx_words <= lim.max_coeff_hatx_words);
  assert(planner0.totals().trunc_hatx_words <= lim.max_trunc_hatx_words);
  assert(planner1.totals().trunc_hatx_words <= lim.max_trunc_hatx_words);
  assert(planner0.totals().coeff_hatx_bytes <= lim.max_coeff_hatx_bytes);
  assert(planner1.totals().coeff_hatx_bytes <= lim.max_coeff_hatx_bytes);
  assert(planner0.totals().trunc_hatx_bytes <= lim.max_trunc_hatx_bytes);
  assert(planner1.totals().trunc_hatx_bytes <= lim.max_trunc_hatx_bytes);
  assert(planner0.totals().phases <= lim.max_phases);
  assert(planner1.totals().phases <= lim.max_phases);
  assert(planner0.totals().coeff_active_elems <= lim.max_coeff_active_elems);
  assert(planner1.totals().coeff_active_elems <= lim.max_coeff_active_elems);
  assert(planner0.totals().trunc_active_elems <= lim.max_trunc_active_elems);
  assert(planner1.totals().trunc_active_elems <= lim.max_trunc_active_elems);
  assert(planner0.totals().coeff_cost_effbits <= lim.max_coeff_cost_effbits);
  assert(planner1.totals().coeff_cost_effbits <= lim.max_coeff_cost_effbits);
  assert(planner0.totals().trunc_cost_effbits <= lim.max_trunc_cost_effbits);
  assert(planner1.totals().trunc_cost_effbits <= lim.max_trunc_cost_effbits);
  assert(planner0.totals().coeff_active_elems > 0);
  assert(planner1.totals().coeff_active_elems > 0);

  return 0;
}
