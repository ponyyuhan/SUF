#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <thread>
#include <vector>

#include "core/ring.hpp"
#include "gates/layernorm_block.hpp"
#include "gates/nexp_gate.hpp"
#include "gates/reciprocal_gate.hpp"
#include "gates/rsqrt_gate.hpp"
#include "gates/silu_spline_gate.hpp"
#include "gates/softmax_block.hpp"
#include "mpc/net.hpp"
#include "pfss/backend_cleartext.hpp"

using R = core::Z2n<64>;
using namespace gates;

struct LocalChan : net::Chan {
  struct Shared {
    std::mutex m;
    std::condition_variable cv;
    std::queue<uint64_t> q0to1, q1to0;
  };
  Shared* shared = nullptr;
  bool is0 = false;

  LocalChan() = default;
  LocalChan(Shared* s, bool is0_) : shared(s), is0(is0_) {}

  void send_u64(uint64_t v) override {
    std::unique_lock<std::mutex> lk(shared->m);
    auto& q = is0 ? shared->q0to1 : shared->q1to0;
    q.push(v);
    shared->cv.notify_all();
  }

  uint64_t recv_u64() override {
    std::unique_lock<std::mutex> lk(shared->m);
    auto& q = is0 ? shared->q1to0 : shared->q0to1;
    shared->cv.wait(lk, [&] { return !q.empty(); });
    uint64_t v = q.front();
    q.pop();
    return v;
  }
};

static int64_t to_fixed(double x, int frac_bits) {
  return static_cast<int64_t>(std::llround(x * std::ldexp(1.0, frac_bits)));
}

constexpr int64_t kTol = 512;  // allow small ring rounding slack

static void test_silu_gate() {
  pfss::CleartextBackendCoeff backend;
  auto pp = backend.setup(128);
  SiLUGateParams params;
  auto spec = make_silu_spec(params);

  std::mt19937_64 rng(123);
  std::uniform_real_distribution<double> dist(-9.0, 9.0);
  for (int i = 0; i < 8; ++i) {
    auto keys = dealer_make_silu_keys(backend, pp, params, rng);
    uint64_t rin = keys.party0.r_in_share.s.v + keys.party1.r_in_share.s.v;
    uint64_t rout = keys.party0.r_out_share.s.v + keys.party1.r_out_share.s.v;
    int64_t x = to_fixed(dist(rng), params.frac_bits);
    uint64_t x_hat = static_cast<uint64_t>(x) + rin;
    LocalChan::Shared sh;
    LocalChan c0{&sh, true}, c1{&sh, false};
    mpc::AddShare<R> y0, y1;
    std::thread t0([&] { y0 = eval_silu_gate(keys.party0, 0, c0, backend, pp, x_hat); });
    std::thread t1([&] { y1 = eval_silu_gate(keys.party1, 1, c1, backend, pp, x_hat); });
    t0.join();
    t1.join();
    uint64_t y_hat = y0.s.v + y1.s.v;
    int64_t y_plain = static_cast<int64_t>(y_hat - rout);
    int64_t expected = ref_silu_fixed(spec, x);
    if (std::llabs(y_plain - expected) > kTol) {
      std::cerr << "SiLU mismatch x=" << x << " got=" << y_plain << " expected=" << expected << "\n";
      std::cerr << "rin=" << rin << " x_hat=" << x_hat << " degree=" << keys.party0.degree << "\n";
      std::cerr << "y_hat=" << y_hat << " rout=" << rout << "\n";
      std::cerr << "spec intervals=" << spec.intervals.size() << "\n";
      for (size_t i = 0; i < spec.intervals.size(); ++i) {
        auto iv = spec.intervals[i];
        std::cerr << "spec[" << i << "] start=" << static_cast<int64_t>(iv.start)
                  << " end=" << static_cast<int64_t>(iv.end) << "\n";
      }
      int out_words = 0;
      auto desc = build_piecewise_poly_desc(spec, rin, keys.party0.degree, out_words);
      size_t idx = 0;
      for (const auto& p : desc.pieces) {
        std::cerr << "piece" << idx++ << " [" << p.L << "," << p.U << ")\n";
      }
      for (const auto& p : desc.pieces) {
        bool in = false;
        if (p.L <= p.U) in = (x_hat >= p.L && x_hat < p.U);
        else in = (x_hat >= p.L || x_hat < p.U);
        if (in) {
          std::cerr << "interval [" << p.L << "," << p.U << ") payload:";
          for (auto v : p.payload) std::cerr << " " << static_cast<int64_t>(v);
          std::cerr << "\n";
          break;
        }
      }
      assert(false);
    }
  }
}

static void test_nexp_gate() {
  pfss::CleartextBackendCoeff backend;
  auto pp = backend.setup(128);
  NExpGateParams params;
  auto spec = make_nexp_spec(params);
  std::mt19937_64 rng(456);
  std::uniform_real_distribution<double> dist(-1.0, 17.0);
  for (int i = 0; i < 8; ++i) {
    auto keys = dealer_make_nexp_keys(backend, pp, params, rng);
    uint64_t rin = keys.party0.r_in_share.s.v + keys.party1.r_in_share.s.v;
    uint64_t rout = keys.party0.r_out_share.s.v + keys.party1.r_out_share.s.v;
    int64_t x = to_fixed(dist(rng), params.frac_bits);
    uint64_t x_hat = static_cast<uint64_t>(x) + rin;
    LocalChan::Shared sh;
    LocalChan c0{&sh, true}, c1{&sh, false};
    mpc::AddShare<R> y0, y1;
    std::thread t0([&] { y0 = eval_nexp_gate(keys.party0, 0, c0, backend, pp, x_hat); });
    std::thread t1([&] { y1 = eval_nexp_gate(keys.party1, 1, c1, backend, pp, x_hat); });
    t0.join();
    t1.join();
    uint64_t y_hat = y0.s.v + y1.s.v;
    int64_t y_plain = static_cast<int64_t>(y_hat - rout);
    int64_t expected = ref_nexp_fixed(spec, x);
    if (std::llabs(y_plain - expected) > kTol) {
      std::cerr << "nExp mismatch\n";
      assert(false);
    }
  }
}

static void test_reciprocal_gate() {
  pfss::CleartextBackendCoeff backend;
  auto pp = backend.setup(128);
  ReciprocalParams params;
  params.nmax = 256.0;
  params.nr_iters = 1;
  params.frac_bits = 16;
  std::mt19937_64 rng(789);
  auto spec = make_recip_affine_init_spec(params.frac_bits, params.nmax);

  std::uniform_real_distribution<double> dist(1.0, 200.0);

  for (int i = 0; i < 8; ++i) {
    auto keys = dealer_make_recip_keys(backend, pp, params, rng);
    uint64_t rin = keys.party0.init_key.r_in_share.s.v + keys.party1.init_key.r_in_share.s.v;
    uint64_t rout = keys.party0.out_mask.s.v + keys.party1.out_mask.s.v;
    int64_t x = to_fixed(dist(rng), params.frac_bits);
    uint64_t x_hat = static_cast<uint64_t>(x) + rin;
    LocalChan::Shared sh;
    LocalChan c0{&sh, true}, c1{&sh, false};
    mpc::AddShare<R> y0, y1;
    std::thread t0([&] { y0 = eval_reciprocal_gate(keys.party0, 0, c0, backend, pp, x_hat); });
    std::thread t1([&] { y1 = eval_reciprocal_gate(keys.party1, 1, c1, backend, pp, x_hat); });
    t0.join();
    t1.join();
    uint64_t y_hat = y0.s.v + y1.s.v;
    int64_t y_plain = static_cast<int64_t>(y_hat - rout);
    int64_t expected = ref_reciprocal_fixed(spec, x, params.frac_bits, params.nr_iters);
    if (std::llabs(y_plain - expected) > kTol) {
      std::cerr << "reciprocal mismatch\n";
      assert(false);
    }
  }
}

static void test_rsqrt_gate() {
  pfss::CleartextBackendCoeff backend;
  auto pp = backend.setup(128);
  RsqrtParams params;
  params.nr_iters = 1;
  params.frac_bits = 16;
  params.eps = 1.0 / 2048.0;
  std::mt19937_64 rng(321);
  auto spec = make_rsqrt_affine_init_spec(params.frac_bits, params.eps, params.vmax);

  std::uniform_real_distribution<double> dist(params.eps, 8.0);

  for (int i = 0; i < 8; ++i) {
    auto keys = dealer_make_rsqrt_keys(backend, pp, params, rng);
    uint64_t rin = keys.party0.init_key.r_in_share.s.v + keys.party1.init_key.r_in_share.s.v;
    uint64_t rout = keys.party0.out_mask.s.v + keys.party1.out_mask.s.v;
    int64_t x = to_fixed(dist(rng), params.frac_bits);
    uint64_t x_hat = static_cast<uint64_t>(x) + rin;
    LocalChan::Shared sh;
    LocalChan c0{&sh, true}, c1{&sh, false};
    mpc::AddShare<R> y0, y1;
    std::thread t0([&] { y0 = eval_rsqrt_gate(keys.party0, 0, c0, backend, pp, x_hat); });
    std::thread t1([&] { y1 = eval_rsqrt_gate(keys.party1, 1, c1, backend, pp, x_hat); });
    t0.join();
    t1.join();
    uint64_t y_hat = y0.s.v + y1.s.v;
    int64_t y_plain = static_cast<int64_t>(y_hat - rout);
    int64_t expected = ref_rsqrt_fixed(spec, x, params.frac_bits, params.nr_iters);
    if (std::llabs(y_plain - expected) > kTol) {
      std::cerr << "rsqrt mismatch\n";
      assert(false);
    }
  }
}

static std::vector<int64_t> softmax_ref(
    const PiecewisePolySpec& nexp_spec,
    const PiecewisePolySpec& recip_spec,
    const std::vector<int64_t>& x,
    int frac_bits,
    int recip_iters) {
  int64_t max_x = x.empty() ? 0 : x[0];
  for (size_t i = 1; i < x.size(); ++i) if (x[i] > max_x) max_x = x[i];
  std::vector<int64_t> e(x.size(), 0);
  int64_t sum = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    e[i] = ref_nexp_fixed(nexp_spec, max_x - x[i]);
    sum += e[i];
  }
  if (sum == 0) sum = 1;
  int64_t inv = ref_reciprocal_fixed(recip_spec, sum, frac_bits, recip_iters);
  std::vector<int64_t> y(x.size(), 0);
  for (size_t i = 0; i < x.size(); ++i) {
    __int128 prod = static_cast<__int128>(e[i]) * static_cast<__int128>(inv);
    y[i] = static_cast<int64_t>(prod >> frac_bits);
  }
  return y;
}

static void test_softmax_block() {
  SoftmaxBlockParams params;
  params.L = 8;
  std::mt19937_64 rng(999);
  auto keys = dealer_make_softmax_keys(params, rng);

  std::vector<mpc::AddShare<R>> xs0(params.L), xs1(params.L);
  std::uniform_real_distribution<double> dist(-3.0, 3.0);
  for (size_t i = 0; i < params.L; ++i) {
    int64_t val = to_fixed(dist(rng), params.frac_bits);
    uint64_t share0 = rng();
    uint64_t share1 = static_cast<uint64_t>(val) - share0;
    xs0[i] = {R(share0)};
    xs1[i] = {R(share1)};
  }

  LocalChan::Shared sh;
  LocalChan c0{&sh, true}, c1{&sh, false};
  std::vector<mpc::AddShare<R>> out0, out1;
  std::thread t0([&] { eval_softmax_block(keys.party0, 0, c0, xs0, out0); });
  std::thread t1([&] { eval_softmax_block(keys.party1, 1, c1, xs1, out1); });
  t0.join();
  t1.join();

  std::vector<int64_t> plain(params.L, 0);
  for (size_t i = 0; i < params.L; ++i) {
    plain[i] = static_cast<int64_t>(xs0[i].s.v + xs1[i].s.v);
  }
  auto expected = softmax_ref(keys.party0.nexp_spec, keys.party0.recip_init_spec,
                              plain, params.frac_bits, keys.party0.recip_iters);

  for (size_t i = 0; i < params.L; ++i) {
    uint64_t mask = keys.party0.out_masks[i].s.v + keys.party1.out_masks[i].s.v;
    uint64_t y_hat = out0[i].s.v + out1[i].s.v;
    int64_t y_plain = static_cast<int64_t>(y_hat - mask);
    if (std::llabs(y_plain - expected[i]) > kTol) {
      std::cerr << "softmax mismatch at " << i << "\n";
      assert(false);
    }
  }
}

static std::vector<int64_t> layernorm_ref(
    const PiecewisePolySpec& rsqrt_spec,
    int64_t eps_fixed,
    const std::vector<int64_t>& x,
    const std::vector<int64_t>* gamma,
    const std::vector<int64_t>* beta,
    int frac_bits,
    int iters) {
  int64_t sum = 0;
  for (auto v : x) sum += v;
  int64_t mu = x.empty() ? 0 : sum / static_cast<int64_t>(x.size());
  int64_t var_acc = 0;
  for (auto v : x) {
    int64_t d = v - mu;
    __int128 sq = static_cast<__int128>(d) * static_cast<__int128>(d);
    var_acc += static_cast<int64_t>(sq >> frac_bits);
  }
  int64_t var = x.empty() ? 0 : var_acc / static_cast<int64_t>(x.size());
  int64_t var_eps = var + eps_fixed;
  int64_t r = ref_rsqrt_fixed(rsqrt_spec, var_eps, frac_bits, iters);
  int64_t one = static_cast<int64_t>(uint64_t(1) << frac_bits);
  std::vector<int64_t> out(x.size(), 0);
  for (size_t i = 0; i < x.size(); ++i) {
    int64_t d = x[i] - mu;
    __int128 prod = static_cast<__int128>(d) * static_cast<__int128>(r);
    int64_t z = static_cast<int64_t>(prod >> frac_bits);
    int64_t g = (gamma && i < gamma->size()) ? (*gamma)[i] : one;
    int64_t b = (beta && i < beta->size()) ? (*beta)[i] : 0;
    __int128 zg = static_cast<__int128>(z) * static_cast<__int128>(g);
    out[i] = static_cast<int64_t>(zg >> frac_bits) + b;
  }
  return out;
}

static void test_layernorm_block() {
  LayerNormParams params;
  params.L = 8;
  std::mt19937_64 rng(2024);
  auto keys = dealer_make_layernorm_keys(params, rng);

  std::vector<mpc::AddShare<R>> xs0(params.L), xs1(params.L);
  std::vector<mpc::AddShare<R>> gs0(params.L), gs1(params.L);
  std::vector<mpc::AddShare<R>> bs0(params.L), bs1(params.L);
  std::uniform_real_distribution<double> dist(-2.0, 2.0);
  std::uniform_real_distribution<double> dist_gamma(0.5, 1.5);
  for (size_t i = 0; i < params.L; ++i) {
    int64_t val = to_fixed(dist(rng), params.frac_bits);
    uint64_t share0 = rng();
    uint64_t share1 = static_cast<uint64_t>(val) - share0;
    xs0[i] = {R(share0)};
    xs1[i] = {R(share1)};

    int64_t gval = to_fixed(dist_gamma(rng), params.frac_bits);
    uint64_t g0 = rng();
    uint64_t g1 = static_cast<uint64_t>(gval) - g0;
    gs0[i] = {R(g0)};
    gs1[i] = {R(g1)};

    int64_t bval = to_fixed(dist(rng), params.frac_bits);
    uint64_t b0 = rng();
    uint64_t b1 = static_cast<uint64_t>(bval) - b0;
    bs0[i] = {R(b0)};
    bs1[i] = {R(b1)};
  }

  LocalChan::Shared sh;
  LocalChan c0{&sh, true}, c1{&sh, false};
  std::vector<mpc::AddShare<R>> out0, out1;
  std::thread t0([&] {
    eval_layernorm_block(keys.party0, 0, c0, xs0, &gs0, &bs0, out0);
  });
  std::thread t1([&] {
    eval_layernorm_block(keys.party1, 1, c1, xs1, &gs1, &bs1, out1);
  });
  t0.join();
  t1.join();

  std::vector<int64_t> plain(params.L, 0), gamma_plain(params.L, 0), beta_plain(params.L, 0);
  for (size_t i = 0; i < params.L; ++i) {
    plain[i] = static_cast<int64_t>(xs0[i].s.v + xs1[i].s.v);
    gamma_plain[i] = static_cast<int64_t>(gs0[i].s.v + gs1[i].s.v);
    beta_plain[i] = static_cast<int64_t>(bs0[i].s.v + bs1[i].s.v);
  }
  auto expected = layernorm_ref(keys.party0.rsqrt_init_spec, keys.party0.eps_fixed,
                                plain, &gamma_plain, &beta_plain,
                                params.frac_bits, keys.party0.rsqrt_iters);

  for (size_t i = 0; i < params.L; ++i) {
    uint64_t mask = keys.party0.out_masks[i].s.v + keys.party1.out_masks[i].s.v;
    uint64_t y_hat = out0[i].s.v + out1[i].s.v;
    int64_t y_plain = static_cast<int64_t>(y_hat - mask);
    if (std::llabs(y_plain - expected[i]) > kTol) {
      std::cerr << "layernorm mismatch at " << i << "\n";
      assert(false);
    }
  }
}

int main() {
  test_silu_gate();
  test_nexp_gate();
  test_reciprocal_gate();
  test_rsqrt_gate();
  test_softmax_block();
  test_layernorm_block();
  std::cout << "All LLM gate tests passed\n";
  return 0;
}
