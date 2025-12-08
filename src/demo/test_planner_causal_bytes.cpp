#include <cassert>
#include <cstdint>
#include <random>
#include <vector>

#include "nn/transformer_layer.hpp"
#include "nn/attention_block.hpp"
#include "nn/mlp_block.hpp"
#include "nn/layer_context.hpp"
#include "proto/backend_clear.hpp"
#include "runtime/pfss_phase_planner.hpp"
#include "runtime/pfss_superbatch.hpp"
#include "runtime/phase_executor.hpp"
#include "mpc/local_chan.hpp"

// Regression: ensure causal/ragged packing respects byte budgets and still finishes.
int main() {
  const int B = 1;
  const int T = 4;
  const int D = 4;
  const int H = 1;
  const int Dh = D / H;
  const int fb = 8;

  nn::TransformerConfig cfg;
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

  std::mt19937_64 rng(321);
  std::vector<uint64_t> X0, X1;
  split_shares(X, rng, X0, X1);
  std::vector<uint64_t> Y0(T * D, 0), Y1(T * D, 0);
  nn::KVCache cache0(B, H, T, Dh), cache1(B, H, T, Dh);

  proto::ReferenceBackend be0, be1;
  compiler::TruncationPassContext trunc0(be0, 0xabcdu);
  compiler::TruncationPassContext trunc1(be1, 0xabcdu);
  nn::LayerContext ctx0, ctx1;
  ctx0.trunc_ctx = &trunc0;
  ctx1.trunc_ctx = &trunc1;
  ctx0.frac_bits = fb;
  ctx1.frac_bits = fb;

  runtime::PfssLayerPlanner planner0, planner1;
  runtime::PfssLayerPlanner::Limits lim;
  lim.max_coeff_flushes = 4;
  lim.max_trunc_flushes = 4;
  lim.max_coeff_jobs = 16;
  lim.max_trunc_jobs = 16;
  lim.max_coeff_hatx_words = 48;  // tighter than default
  lim.max_trunc_hatx_words = 48;
  lim.max_coeff_hatx_bytes = lim.max_coeff_hatx_words * sizeof(uint64_t);
  lim.max_trunc_hatx_bytes = lim.max_trunc_hatx_words * sizeof(uint64_t);
  lim.max_phases = 32;
  planner0.set_limits(lim);
  planner1.set_limits(lim);
  ctx0.pfss_layer_planner = &planner0;
  ctx1.pfss_layer_planner = &planner1;

  runtime::PfssSuperBatch::Limits pfss_lim;
  pfss_lim.max_pending_jobs = 1ull << 10;
  pfss_lim.max_pending_hatx_words = 1ull << 16;
  pfss_lim.max_pending_hatx_bytes = pfss_lim.max_pending_hatx_words * sizeof(uint64_t);
  pfss_lim.max_flushes = 1ull << 8;

  mpc::LocalChan::Shared sh_main, sh_pfss;
  mpc::LocalChan c0(&sh_main, true), c1(&sh_main, false);
  mpc::LocalChan pfss0(&sh_pfss, true), pfss1(&sh_pfss, false);
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
      ctx1.pfss_trunc_batch_limits = pfss_lim;
      ctx1.pfss_coeff_batch_limits = pfss_lim;
      nn::transformer_layer_forward(cfg, 1, c1,
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
    ctx0.pfss_trunc_batch_limits = pfss_lim;
    ctx0.pfss_coeff_batch_limits = pfss_lim;
    nn::transformer_layer_forward(cfg, 0, c0,
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
    throw std::runtime_error("transformer causal bytes test: " + err);
  }

  // Reconstruct and sanity-check nonzero outputs.
  for (size_t i = 0; i < Y0.size(); ++i) {
    uint64_t plain = Y0[i] + Y1[i];
    assert(plain != 0);
  }

  // Budget/flush sanity under tightened byte limits.
  assert(planner0.totals().coeff_flushes <= lim.max_coeff_flushes);
  assert(planner0.totals().trunc_flushes <= lim.max_trunc_flushes);
  assert(planner0.totals().coeff_jobs <= lim.max_coeff_jobs);
  assert(planner0.totals().trunc_jobs <= lim.max_trunc_jobs);
  assert(planner0.totals().coeff_hatx_bytes <= lim.max_coeff_hatx_bytes);
  assert(planner0.totals().trunc_hatx_bytes <= lim.max_trunc_hatx_bytes);
  assert(planner1.totals().coeff_hatx_bytes <= lim.max_coeff_hatx_bytes);
  assert(planner1.totals().trunc_hatx_bytes <= lim.max_trunc_hatx_bytes);
  return 0;
}
