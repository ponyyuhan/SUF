#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <thread>
#include <vector>
#include <atomic>

#include "gates/silu_spline_gate.hpp"
#include "gates/rsqrt_gate.hpp"
#include "gates/tables/rsqrt_piecewise_affine_init.hpp"
#include "gates/tables/silu_spline_table.hpp"
#include "gates/nexp_gate.hpp"
#include "gates/reciprocal_gate.hpp"
#include "nn/attention_block.hpp"
#include "nn/matmul_publicW.hpp"
#include "nn/transformer_layer.hpp"
#include "mpc/net.hpp"
#include "proto/reference_backend.hpp"
#include "compiler/truncation_pass_runner.hpp"

using namespace nn;

struct LocalChan : net::Chan {
  struct Shared {
    std::mutex m;
    std::condition_variable cv;
    std::queue<uint64_t> q0to1, q1to0;
  };
  Shared* s = nullptr;
  bool is0 = false;
  LocalChan() = default;
  LocalChan(Shared* sh, bool p) : s(sh), is0(p) {}
  void send_u64(uint64_t v) override {
    std::unique_lock<std::mutex> lk(s->m);
    auto& q = is0 ? s->q0to1 : s->q1to0;
    q.push(v);
    s->cv.notify_all();
  }
  uint64_t recv_u64() override {
    std::unique_lock<std::mutex> lk(s->m);
    auto& q = is0 ? s->q1to0 : s->q0to1;
    s->cv.wait(lk, [&]{ return !q.empty(); });
    uint64_t v = q.front(); q.pop(); return v;
  }
};

static inline int64_t to_signed(uint64_t v) { return static_cast<int64_t>(v); }
static inline uint64_t to_ring(int64_t v) { return static_cast<uint64_t>(v); }

static std::vector<int64_t> matmul_ref(const std::vector<int64_t>& X,
                                       const std::vector<int64_t>& W,
                                       size_t B,
                                       size_t M,
                                       size_t K,
                                       size_t N,
                                       int frac_bits,
                                       bool w_transposed = false) {
  std::vector<int64_t> out(B * M * N, 0);
  for (size_t b = 0; b < B; ++b) {
    for (size_t m = 0; m < M; ++m) {
      for (size_t n = 0; n < N; ++n) {
        __int128 acc = 0;
        for (size_t k = 0; k < K; ++k) {
          size_t xidx = (b * M + m) * K + k;
          size_t widx = w_transposed ? (n * K + k) : (k * N + n);
          acc += static_cast<__int128>(X[xidx]) * static_cast<__int128>(W[widx]);
        }
        out[(b * M + m) * N + n] = static_cast<int64_t>(acc >> frac_bits);
      }
    }
  }
  return out;
}

static void split_shares(const std::vector<int64_t>& plain,
                         std::mt19937_64& rng,
                         int frac_bits,
                         std::vector<uint64_t>& s0,
                         std::vector<uint64_t>& s1) {
  s0.resize(plain.size());
  s1.resize(plain.size());
  uint64_t mask = (frac_bits >= 63) ? ~uint64_t(0) : ((uint64_t(1) << frac_bits) - 1);
  for (size_t i = 0; i < plain.size(); ++i) {
    uint64_t r = rng();
    if (frac_bits > 0) r &= ~mask;  // avoid truncation carry when rescaling
    s0[i] = r;
    s1[i] = to_ring(plain[i] - static_cast<int64_t>(r));
  }
}

static std::vector<int64_t> attention_ref(const AttentionConfig& cfg,
                                          const std::vector<int64_t>& X,
                                          const std::vector<int64_t>& Wqkv,
                                          const std::vector<int64_t>& Wout) {
  size_t B = 1;
  size_t T = X.size() / cfg.D;
  size_t D = cfg.D, H = cfg.H, Dh = cfg.Dh;
  int fb = cfg.frac_bits;
  auto qkv = matmul_ref(X, Wqkv, B, T, D, 3 * D, fb);

  std::vector<int64_t> K_cache(cfg.S_max * H * Dh, 0);
  std::vector<int64_t> V_cache(cfg.S_max * H * Dh, 0);
  size_t cur_len = 0;

  gates::NExpGateParams nexp_params;
  nexp_params.frac_bits = fb;
  nexp_params.segments = 16;
  auto nexp_spec = gates::make_nexp_spec(nexp_params);
  auto recip_spec =
      gates::make_recip_affine_init_spec(fb, static_cast<double>(std::max(cfg.S_max, T)));
  int64_t inv_sqrt = static_cast<int64_t>(
      std::llround((1.0 / std::sqrt(static_cast<double>(Dh))) * std::ldexp(1.0, fb)));
  if (inv_sqrt == 0) inv_sqrt = 1;

  std::vector<int64_t> ctx(B * T * H * Dh, 0);

  for (size_t t = 0; t < T; ++t) {
    const int64_t* q_ptr = qkv.data() + t * 3 * D;
    const int64_t* k_ptr = q_ptr + D;
    const int64_t* v_ptr = q_ptr + 2 * D;

    for (size_t h = 0; h < H; ++h) {
      for (size_t d = 0; d < Dh; ++d) {
        size_t slot = (h * cfg.S_max + cur_len) * Dh + d;
        K_cache[slot] = k_ptr[h * Dh + d];
        V_cache[slot] = v_ptr[h * Dh + d];
      }
    }
    cur_len += 1;

    for (size_t h = 0; h < H; ++h) {
      std::vector<int64_t> scores(cur_len, 0);
      for (size_t s = 0; s < cur_len; ++s) {
        __int128 acc = 0;
        for (size_t d = 0; d < Dh; ++d) {
          size_t idx = (h * cfg.S_max + s) * Dh + d;
          acc += static_cast<__int128>(q_ptr[h * Dh + d]) * static_cast<__int128>(K_cache[idx]);
        }
        acc >>= fb;
        acc = (acc * static_cast<__int128>(inv_sqrt)) >> fb;
        scores[s] = static_cast<int64_t>(acc);
      }

      int64_t max_sc = scores.empty() ? 0 : *std::max_element(scores.begin(), scores.end());
      std::vector<int64_t> expv(cur_len, 0);
      int64_t sum = 0;
      for (size_t s = 0; s < cur_len; ++s) {
        int64_t diff = max_sc - scores[s];
        expv[s] = gates::ref_nexp_fixed(nexp_spec, diff);
        sum += expv[s];
      }
      if (sum == 0) sum = 1;
      int64_t inv = gates::ref_reciprocal_fixed(recip_spec, sum, fb, 1);

      std::vector<int64_t> prob(cur_len, 0);
      for (size_t s = 0; s < cur_len; ++s) {
        __int128 p = static_cast<__int128>(expv[s]) * static_cast<__int128>(inv);
        prob[s] = static_cast<int64_t>(p >> fb);
      }

      for (size_t d = 0; d < Dh; ++d) {
        __int128 acc = 0;
        for (size_t s = 0; s < cur_len; ++s) {
          size_t idx = (h * cfg.S_max + s) * Dh + d;
          acc += static_cast<__int128>(prob[s]) * static_cast<__int128>(V_cache[idx]);
        }
        ctx[(t * H + h) * Dh + d] = static_cast<int64_t>(acc >> fb);
      }
    }
  }

  // Merge heads and project out.
  std::vector<int64_t> merged(B * T * D, 0);
  for (size_t t = 0; t < T; ++t) {
    for (size_t h = 0; h < H; ++h) {
      for (size_t d = 0; d < Dh; ++d) {
        size_t dst = t * D + h * Dh + d;
        size_t src = (t * H + h) * Dh + d;
        merged[dst] = ctx[src];
      }
    }
  }

  return matmul_ref(merged, Wout, B, T, D, D, fb);
}

static std::vector<int64_t> layernorm_ref(const std::vector<int64_t>& x,
                                          size_t D,
                                          int frac_bits,
                                          const gates::PiecewisePolySpec& rsqrt_spec,
                                          int rsqrt_iters) {
  int64_t sum = 0;
  for (auto v : x) sum += v;
  int64_t mu = D == 0 ? 0 : (sum / static_cast<int64_t>(D));

  int64_t var_acc = 0;
  for (auto v : x) {
    int64_t d = v - mu;
    __int128 sq = static_cast<__int128>(d) * static_cast<__int128>(d);
    var_acc += static_cast<int64_t>(sq >> frac_bits);
  }
  int64_t var = D == 0 ? 0 : (var_acc / static_cast<int64_t>(D));
  int64_t eps_fixed = static_cast<int64_t>(std::llround((1.0 / 1024.0) * std::ldexp(1.0, frac_bits)));
  int64_t r = gates::ref_rsqrt_fixed(rsqrt_spec, var + eps_fixed, frac_bits, rsqrt_iters);

  std::vector<int64_t> out(D, 0);
  for (size_t i = 0; i < D; ++i) {
    __int128 prod = static_cast<__int128>(x[i] - mu) * static_cast<__int128>(r);
    out[i] = static_cast<int64_t>(prod >> frac_bits);
  }
  return out;
}

static std::vector<int64_t> mlp_ref(const MLPConfig& cfg,
                                    const std::vector<int64_t>& X,
                                    const std::vector<int64_t>& W1,
                                    const std::vector<int64_t>& W2) {
  size_t B = 1;
  size_t T = X.size() / cfg.D;
  auto hidden = matmul_ref(X, W1, B, T, cfg.D, cfg.Hidden, cfg.frac_bits);
  auto spec = gates::make_silu_spec({cfg.frac_bits, 16});
  for (auto& v : hidden) v = gates::ref_silu_fixed(spec, v);
  return matmul_ref(hidden, W2, B, T, cfg.Hidden, cfg.D, cfg.frac_bits);
}

static std::vector<int64_t> transformer_ref(const TransformerConfig& cfg,
                                            const std::vector<int64_t>& X,
                                            const std::vector<int64_t>& Wqkv,
                                            const std::vector<int64_t>& Wout,
                                            const std::vector<int64_t>& W1,
                                            const std::vector<int64_t>& W2) {
  size_t D = cfg.attn.D;
  size_t T = X.size() / D;

  gates::RsqrtParams rp;
  rp.frac_bits = cfg.frac_bits;
  rp.eps = 1.0 / 1024.0;
  rp.vmax = 16.0;
  auto rsqrt_spec = gates::make_rsqrt_affine_init_spec(rp.frac_bits, rp.eps, rp.vmax);
  int rsqrt_iters = 1;

  std::vector<int64_t> ln1;
  ln1.reserve(X.size());
  for (size_t t = 0; t < T; ++t) {
    std::vector<int64_t> slice(X.begin() + t * D, X.begin() + (t + 1) * D);
    auto normed = layernorm_ref(slice, D, cfg.frac_bits, rsqrt_spec, rsqrt_iters);
    ln1.insert(ln1.end(), normed.begin(), normed.end());
  }

  auto attn = attention_ref(cfg.attn, ln1, Wqkv, Wout);
  std::vector<int64_t> resid1(attn.size(), 0);
  for (size_t i = 0; i < attn.size(); ++i) resid1[i] = attn[i] + X[i];

  std::vector<int64_t> ln2;
  ln2.reserve(resid1.size());
  for (size_t t = 0; t < T; ++t) {
    std::vector<int64_t> slice(resid1.begin() + t * D, resid1.begin() + (t + 1) * D);
    auto normed = layernorm_ref(slice, D, cfg.frac_bits, rsqrt_spec, rsqrt_iters);
    ln2.insert(ln2.end(), normed.begin(), normed.end());
  }

  auto mlp_out = mlp_ref(cfg.mlp, ln2, W1, W2);
  std::vector<int64_t> y(mlp_out.size(), 0);
  for (size_t i = 0; i < y.size(); ++i) y[i] = mlp_out[i] + resid1[i];
  return y;
}

static void test_attention_correctness() {
  std::mt19937_64 rng(10);
  AttentionConfig cfg;
  cfg.D = 4;
  cfg.H = 1;
  cfg.Dh = 4;
  cfg.S_max = 6;
  cfg.frac_bits = 8;

  size_t T = 3;
  std::vector<int64_t> X(T * cfg.D), Wqkv(cfg.D * cfg.D * 3), Wout(cfg.D * cfg.D);
  for (auto& v : X) v = static_cast<int64_t>(rng() % 64);
  for (auto& w : Wqkv) w = static_cast<int64_t>(rng() % 32);
  for (auto& w : Wout) w = static_cast<int64_t>(rng() % 32);

  std::vector<uint64_t> X0, X1;
  split_shares(X, rng, cfg.frac_bits, X0, X1);

  std::vector<uint64_t> Y0(T * cfg.D), Y1(T * cfg.D);
  KVCache cache0(1, cfg.H, cfg.S_max, cfg.Dh), cache1(1, cfg.H, cfg.S_max, cfg.Dh);
  proto::ReferenceBackend trunc_backend0, trunc_backend1;
  compiler::TruncationPassContext trunc_ctx0(trunc_backend0, 0x6174743064756c6cull);
  compiler::TruncationPassContext trunc_ctx1(trunc_backend1, 0x6174743064756c6cull);
  LayerContext ctx0, ctx1;
  ctx0.trunc_ctx = &trunc_ctx0;
  ctx1.trunc_ctx = &trunc_ctx1;
  ctx0.frac_bits = cfg.frac_bits;
  ctx1.frac_bits = cfg.frac_bits;
  runtime::PhaseExecutor pe0, pe1;

  std::atomic<bool> fail{false};
  std::string err;
  std::mutex err_mu;
  auto record_err = [&](const std::exception& e) {
    fail.store(true, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(err_mu);
    if (err.empty()) err = e.what();
  };

  LocalChan::Shared sh;
  LocalChan c0(&sh, true), c1(&sh, false);
  std::thread th1([&] {
    try {
      attention_forward(cfg, 1, c1,
                        view3(X1.data(), 1, T, cfg.D),
                        view2(Wqkv.data(), cfg.D, cfg.D * 3),
                        view2(Wout.data(), cfg.D, cfg.D),
                        cache1,
                        view3(Y1.data(), 1, T, cfg.D),
                        &ctx1,
                        &pe1);
    } catch (const std::exception& e) {
      record_err(e);
    }
  });
  try {
    attention_forward(cfg, 0, c0,
                      view3(X0.data(), 1, T, cfg.D),
                      view2(Wqkv.data(), cfg.D, cfg.D * 3),
                      view2(Wout.data(), cfg.D, cfg.D),
                      cache0,
                      view3(Y0.data(), 1, T, cfg.D),
                      &ctx0,
                      &pe0);
  } catch (const std::exception& e) {
    record_err(e);
  }
  th1.join();
  if (fail.load(std::memory_order_relaxed)) {
    throw std::runtime_error("attention_correctness: " + err);
  }

  auto plain = attention_ref(cfg, X, Wqkv, Wout);
  for (size_t i = 0; i < plain.size(); ++i) {
    int64_t got = to_signed(Y0[i]) + to_signed(Y1[i]);
    assert(got == plain[i]);
  }
}

static void test_attention_step_vs_batch() {
  std::mt19937_64 rng(11);
  AttentionConfig cfg;
  cfg.D = 6;
  cfg.H = 2;
  cfg.Dh = 3;
  cfg.S_max = 6;
  cfg.frac_bits = 8;

  size_t T = 3;
  std::vector<int64_t> X(T * cfg.D), Wqkv(cfg.D * cfg.D * 3), Wout(cfg.D * cfg.D);
  for (auto& v : X) v = static_cast<int64_t>(rng() % 64);
  for (auto& w : Wqkv) w = static_cast<int64_t>(rng() % 32);
  for (auto& w : Wout) w = static_cast<int64_t>(rng() % 32);

  std::vector<uint64_t> X0, X1;
  split_shares(X, rng, cfg.frac_bits, X0, X1);

  // Batch path.
  std::vector<uint64_t> Y0(T * cfg.D), Y1(T * cfg.D);
  KVCache cache0(1, cfg.H, cfg.S_max, cfg.Dh), cache1(1, cfg.H, cfg.S_max, cfg.Dh);
  proto::ReferenceBackend trunc_backend0, trunc_backend1;
  compiler::TruncationPassContext trunc_ctx0(trunc_backend0, 0x6174743164756c6cull);
  compiler::TruncationPassContext trunc_ctx1(trunc_backend1, 0x6174743164756c6cull);
  LayerContext ctx0, ctx1;
  ctx0.trunc_ctx = &trunc_ctx0;
  ctx1.trunc_ctx = &trunc_ctx1;
  ctx0.frac_bits = cfg.frac_bits;
  ctx1.frac_bits = cfg.frac_bits;
  runtime::PhaseExecutor pe0, pe1;
  std::atomic<bool> fail{false};
  std::string err;
  std::mutex err_mu;
  auto record_err = [&](const std::exception& e) {
    fail.store(true, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(err_mu);
    if (err.empty()) err = e.what();
  };
  LocalChan::Shared sh_batch;
  LocalChan cb0(&sh_batch, true), cb1(&sh_batch, false);
  std::thread th1([&] {
    try {
      attention_forward(cfg, 1, cb1,
                        view3(X1.data(), 1, T, cfg.D),
                        view2(Wqkv.data(), cfg.D, cfg.D * 3),
                        view2(Wout.data(), cfg.D, cfg.D),
                        cache1,
                        view3(Y1.data(), 1, T, cfg.D),
                        &ctx1,
                        &pe1);
    } catch (const std::exception& e) {
      record_err(e);
    }
  });
  try {
    attention_forward(cfg, 0, cb0,
                      view3(X0.data(), 1, T, cfg.D),
                      view2(Wqkv.data(), cfg.D, cfg.D * 3),
                      view2(Wout.data(), cfg.D, cfg.D),
                      cache0,
                      view3(Y0.data(), 1, T, cfg.D),
                      &ctx0,
                      &pe0);
  } catch (const std::exception& e) {
    record_err(e);
  }
  th1.join();
  if (fail.load(std::memory_order_relaxed)) {
    throw std::runtime_error("attention_batch: " + err);
  }

  std::vector<int64_t> batch_out(T * cfg.D, 0);
  for (size_t i = 0; i < batch_out.size(); ++i) {
    batch_out[i] = to_signed(Y0[i]) + to_signed(Y1[i]);
  }

  // Step mode.
  KVCache step0(1, cfg.H, cfg.S_max, cfg.Dh), step1(1, cfg.H, cfg.S_max, cfg.Dh);
  LayerContext sctx0, sctx1;
  sctx0.trunc_ctx = &trunc_ctx0;
  sctx1.trunc_ctx = &trunc_ctx1;
  sctx0.frac_bits = cfg.frac_bits;
  sctx1.frac_bits = cfg.frac_bits;
  runtime::PhaseExecutor spe0, spe1;
  std::atomic<bool> fail_step{false};
  std::string err_step;
  std::mutex err_step_mu;
  auto record_err_step = [&](const std::exception& e) {
    fail_step.store(true, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(err_step_mu);
    if (err_step.empty()) err_step = e.what();
  };
  std::vector<int64_t> step_out;
  LocalChan::Shared sh_step;
  LocalChan cs0(&sh_step, true), cs1(&sh_step, false);
  for (size_t t = 0; t < T; ++t) {
    std::vector<uint64_t> sX0(cfg.D), sX1(cfg.D), sY0(cfg.D), sY1(cfg.D);
    for (size_t d = 0; d < cfg.D; ++d) {
      sX0[d] = X0[t * cfg.D + d];
      sX1[d] = X1[t * cfg.D + d];
    }
    std::thread t1([&] {
      try {
        attention_forward(cfg, 1, cs1,
                          view3(sX1.data(), 1, 1, cfg.D),
                          view2(Wqkv.data(), cfg.D, cfg.D * 3),
                          view2(Wout.data(), cfg.D, cfg.D),
                          step1,
                          view3(sY1.data(), 1, 1, cfg.D),
                          &sctx1,
                          &spe1);
      } catch (const std::exception& e) {
        record_err_step(e);
      }
    });
    try {
      attention_forward(cfg, 0, cs0,
                        view3(sX0.data(), 1, 1, cfg.D),
                        view2(Wqkv.data(), cfg.D, cfg.D * 3),
                        view2(Wout.data(), cfg.D, cfg.D),
                        step0,
                        view3(sY0.data(), 1, 1, cfg.D),
                        &sctx0,
                        &spe0);
    } catch (const std::exception& e) {
      record_err_step(e);
    }
    t1.join();
    if (fail_step.load(std::memory_order_relaxed)) {
      throw std::runtime_error("attention_step: " + err_step);
    }
    for (size_t d = 0; d < cfg.D; ++d) {
      step_out.push_back(to_signed(sY0[d]) + to_signed(sY1[d]));
    }
  }

  assert(batch_out.size() == step_out.size());
  for (size_t i = 0; i < batch_out.size(); ++i) {
    assert(batch_out[i] == step_out[i]);
  }
}

static void test_transformer_layer() {
  std::mt19937_64 rng(12);
  size_t D = 4;
  TransformerConfig cfg;
  cfg.attn = {D, 1, D, 6, 8};
  cfg.mlp = {D, 6, 8};
  cfg.frac_bits = 8;

  size_t T = 2;
  std::vector<int64_t> X(T * D), Wqkv(D * D * 3), Wout(D * D), W1(D * cfg.mlp.Hidden), W2(cfg.mlp.Hidden * D);
  for (auto& v : X) v = static_cast<int64_t>(rng() % 64);
  for (auto& w : Wqkv) w = static_cast<int64_t>(rng() % 32);
  for (auto& w : Wout) w = static_cast<int64_t>(rng() % 32);
  for (auto& w : W1) w = static_cast<int64_t>(rng() % 32);
  for (auto& w : W2) w = static_cast<int64_t>(rng() % 32);

  std::vector<uint64_t> X0, X1;
  split_shares(X, rng, cfg.frac_bits, X0, X1);

  std::vector<uint64_t> Y0(T * D), Y1(T * D);
  KVCache cache0(1, cfg.attn.H, cfg.attn.S_max, cfg.attn.Dh), cache1(1, cfg.attn.H, cfg.attn.S_max, cfg.attn.Dh);

  proto::ReferenceBackend trunc_backend0, trunc_backend1;
  // Deterministic seeds so both parties derive identical truncation material.
  compiler::TruncationPassContext trunc_ctx0(trunc_backend0, 0x74723130ull);
  compiler::TruncationPassContext trunc_ctx1(trunc_backend1, 0x74723130ull);
  LayerContext ctx0, ctx1;
  ctx0.trunc_ctx = &trunc_ctx0;
  ctx1.trunc_ctx = &trunc_ctx1;
  ctx0.frac_bits = cfg.frac_bits;
  ctx1.frac_bits = cfg.frac_bits;

  std::atomic<bool> fail{false};
  std::string err;
  std::mutex err_mu;
  auto record_err = [&](const std::exception& e) {
    fail.store(true, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(err_mu);
    if (err.empty()) err = e.what();
  };

  LocalChan::Shared sh;
  LocalChan c0(&sh, true), c1(&sh, false);
  std::thread th1([&] {
    try {
      transformer_layer_forward(cfg, 1, c1,
                                view3(X1.data(), 1, T, D),
                                view2(Wqkv.data(), D, D * 3),
                                view2(Wout.data(), D, D),
                                view2(W1.data(), D, cfg.mlp.Hidden),
                                view2(W2.data(), cfg.mlp.Hidden, D),
                                cache1,
                                view3(Y1.data(), 1, T, D),
                                &ctx1);
    } catch (const std::exception& e) {
      record_err(e);
    }
  });
  try {
    transformer_layer_forward(cfg, 0, c0,
                              view3(X0.data(), 1, T, D),
                              view2(Wqkv.data(), D, D * 3),
                              view2(Wout.data(), D, D),
                              view2(W1.data(), D, cfg.mlp.Hidden),
                              view2(W2.data(), cfg.mlp.Hidden, D),
                              cache0,
                              view3(Y0.data(), 1, T, D),
                              &ctx0);
  } catch (const std::exception& e) {
    record_err(e);
  }
  th1.join();
  if (fail.load(std::memory_order_relaxed)) {
    throw std::runtime_error("transformer_layer: " + err);
  }

  auto plain = transformer_ref(cfg, X, Wqkv, Wout, W1, W2);
  for (size_t i = 0; i < plain.size(); ++i) {
    int64_t got = to_signed(Y0[i]) + to_signed(Y1[i]);
    if (std::llabs(got - plain[i]) > 1) {
      std::cerr << "transformer mismatch idx=" << i << " got=" << got << " expected=" << plain[i]
                << " y0=" << to_signed(Y0[i]) << " y1=" << to_signed(Y1[i]) << "\n";
      assert(std::llabs(got - plain[i]) <= 1);
    }
  }
}

static void test_mlp_only() {
  std::mt19937_64 rng(21);
  size_t D = 4;
  size_t T = 2;
  MLPConfig cfg{static_cast<int>(D), 6, 8};
  std::vector<int64_t> X(T * D), W1(D * cfg.Hidden), W2(cfg.Hidden * D);
  for (auto& v : X) v = static_cast<int64_t>(rng() % 64);
  for (auto& w : W1) w = static_cast<int64_t>(rng() % 32);
  for (auto& w : W2) w = static_cast<int64_t>(rng() % 32);

  std::vector<uint64_t> X0, X1;
  split_shares(X, rng, cfg.frac_bits, X0, X1);
  std::vector<uint64_t> Y0(T * D), Y1(T * D);

  proto::ReferenceBackend trunc_backend0, trunc_backend1;
  compiler::TruncationPassContext trunc_ctx0(trunc_backend0, 0x6d6c7030ull);
  compiler::TruncationPassContext trunc_ctx1(trunc_backend1, 0x6d6c7030ull);
  LayerContext ctx0, ctx1;
  ctx0.trunc_ctx = &trunc_ctx0;
  ctx1.trunc_ctx = &trunc_ctx1;
  ctx0.frac_bits = cfg.frac_bits;
  ctx1.frac_bits = cfg.frac_bits;

  runtime::PhaseExecutor pe0, pe1;
  ctx0.pfss_batch = &pe0.pfss_trunc_batch();
  ctx1.pfss_batch = &pe1.pfss_trunc_batch();
  ctx0.open_collector = &pe0.open_collector();
  ctx1.open_collector = &pe1.open_collector();

  LocalChan::Shared sh;
  LocalChan c0(&sh, true), c1(&sh, false);
  std::atomic<bool> fail{false};
  std::string err;
  std::mutex err_mu;
  auto record_err = [&](const std::exception& e) {
    fail.store(true, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(err_mu);
    if (err.empty()) err = e.what();
  };

  std::thread th1([&] {
    try {
      mlp_forward(cfg,
                  view3(X1.data(), 1, T, D),
                  view2(W1.data(), D, cfg.Hidden),
                  view2(W2.data(), cfg.Hidden, D),
                  view3(Y1.data(), 1, T, D),
                  1,
                  c1,
                  &ctx1,
                  &pe1);
    } catch (const std::exception& e) {
      record_err(e);
    }
  });
  try {
    mlp_forward(cfg,
                view3(X0.data(), 1, T, D),
                view2(W1.data(), D, cfg.Hidden),
                view2(W2.data(), cfg.Hidden, D),
                view3(Y0.data(), 1, T, D),
                0,
                c0,
                &ctx0,
                &pe0);
  } catch (const std::exception& e) {
    record_err(e);
  }
  th1.join();
  if (fail.load(std::memory_order_relaxed)) {
    throw std::runtime_error("mlp_only: " + err);
  }

  auto plain = mlp_ref(cfg, X, W1, W2);
  for (size_t i = 0; i < plain.size(); ++i) {
    int64_t got = to_signed(Y0[i]) + to_signed(Y1[i]);
    if (std::llabs(got - plain[i]) > 1) {
      std::cerr << "mlp mismatch idx=" << i << " got=" << got << " expected=" << plain[i]
                << " y0=" << to_signed(Y0[i]) << " y1=" << to_signed(Y1[i]) << "\n";
      assert(std::llabs(got - plain[i]) <= 1);
    }
  }
}

int main() {
  test_attention_correctness();
  test_attention_step_vs_batch();
  test_mlp_only();
  test_transformer_layer();
  std::cout << "Attention/Transformer tests passed\n";
  return 0;
}
