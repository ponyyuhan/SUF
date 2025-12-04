#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <string>
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

template<typename Fn>
double time_it(Fn&& fn) {
  auto start = std::chrono::steady_clock::now();
  fn();
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> diff = end - start;
  return diff.count();
}

static void bench_scalar(const std::string& gate, size_t N, int frac_bits) {
  pfss::CleartextBackendCoeff backend;
  auto pp = backend.setup(128);
  std::mt19937_64 rng(42);
  std::uniform_real_distribution<double> dist(-4.0, 4.0);

  if (gate == "silu") {
    SiLUGateParams params;
    params.frac_bits = frac_bits;
    auto keys = dealer_make_silu_keys(backend, pp, params, rng);
    uint64_t rin = keys.party0.r_in_share.s.v + keys.party1.r_in_share.s.v;
    LocalChan::Shared sh;
    LocalChan c0{&sh, true}, c1{&sh, false};
    double secs = time_it([&] {
      for (size_t i = 0; i < N; ++i) {
        int64_t x = to_fixed(dist(rng), frac_bits);
        uint64_t x_hat = static_cast<uint64_t>(x) + rin;
        mpc::AddShare<R> y0, y1;
        y0 = eval_silu_gate(keys.party0, 0, c0, backend, pp, x_hat);
        y1 = eval_silu_gate(keys.party1, 1, c1, backend, pp, x_hat);
        (void)y0; (void)y1;
      }
    });
    std::cout << "silu N=" << N << " time=" << secs << "s elements/s=" << (double)N / secs << "\n";
  } else if (gate == "nexp") {
    NExpGateParams params;
    params.frac_bits = frac_bits;
    auto keys = dealer_make_nexp_keys(backend, pp, params, rng);
    uint64_t rin = keys.party0.r_in_share.s.v + keys.party1.r_in_share.s.v;
    LocalChan::Shared sh;
    LocalChan c0{&sh, true}, c1{&sh, false};
    double secs = time_it([&] {
      for (size_t i = 0; i < N; ++i) {
        int64_t x = to_fixed(dist(rng), frac_bits);
        uint64_t x_hat = static_cast<uint64_t>(x) + rin;
        auto y0 = eval_nexp_gate(keys.party0, 0, c0, backend, pp, x_hat);
        auto y1 = eval_nexp_gate(keys.party1, 1, c1, backend, pp, x_hat);
        (void)y0; (void)y1;
      }
    });
    std::cout << "nexp N=" << N << " time=" << secs << "s elements/s=" << (double)N / secs << "\n";
  } else if (gate == "recip") {
    ReciprocalParams params;
    params.frac_bits = frac_bits;
    params.nr_iters = 1;
    auto keys = dealer_make_recip_keys(backend, pp, params, rng);
    uint64_t rin = keys.party0.init_key.r_in_share.s.v + keys.party1.init_key.r_in_share.s.v;
    LocalChan::Shared sh;
    LocalChan c0{&sh, true}, c1{&sh, false};
    std::uniform_real_distribution<double> pdist(1.0, 64.0);
    double secs = time_it([&] {
      for (size_t i = 0; i < N; ++i) {
        int64_t x = to_fixed(pdist(rng), frac_bits);
        uint64_t x_hat = static_cast<uint64_t>(x) + rin;
        auto y0 = eval_reciprocal_gate(keys.party0, 0, c0, backend, pp, x_hat);
        auto y1 = eval_reciprocal_gate(keys.party1, 1, c1, backend, pp, x_hat);
        (void)y0; (void)y1;
      }
    });
    std::cout << "reciprocal N=" << N << " time=" << secs << "s elements/s=" << (double)N / secs << "\n";
  } else if (gate == "rsqrt") {
    RsqrtParams params;
    params.frac_bits = frac_bits;
    params.nr_iters = 1;
    auto keys = dealer_make_rsqrt_keys(backend, pp, params, rng);
    uint64_t rin = keys.party0.init_key.r_in_share.s.v + keys.party1.init_key.r_in_share.s.v;
    LocalChan::Shared sh;
    LocalChan c0{&sh, true}, c1{&sh, false};
    std::uniform_real_distribution<double> pdist(params.eps, 16.0);
    double secs = time_it([&] {
      for (size_t i = 0; i < N; ++i) {
        int64_t x = to_fixed(pdist(rng), frac_bits);
        uint64_t x_hat = static_cast<uint64_t>(x) + rin;
        auto y0 = eval_rsqrt_gate(keys.party0, 0, c0, backend, pp, x_hat);
        auto y1 = eval_rsqrt_gate(keys.party1, 1, c1, backend, pp, x_hat);
        (void)y0; (void)y1;
      }
    });
    std::cout << "rsqrt N=" << N << " time=" << secs << "s elements/s=" << (double)N / secs << "\n";
  }
}

static void bench_softmax(size_t batches, size_t L, int frac_bits) {
  SoftmaxBlockParams params;
  params.L = L;
  params.frac_bits = frac_bits;
  params.nr_iters_recip = 1;
  std::mt19937_64 rng(77);
  auto keys = dealer_make_softmax_keys(params, rng);
  std::uniform_real_distribution<double> dist(-3.0, 3.0);

  double secs = time_it([&] {
    for (size_t b = 0; b < batches; ++b) {
      std::vector<mpc::AddShare<R>> xs0(L), xs1(L), out0, out1;
      for (size_t i = 0; i < L; ++i) {
        int64_t val = to_fixed(dist(rng), frac_bits);
        uint64_t s0 = rng();
        uint64_t s1 = static_cast<uint64_t>(val) - s0;
        xs0[i] = {R(s0)};
        xs1[i] = {R(s1)};
      }
      LocalChan::Shared sh;
      LocalChan c0{&sh, true}, c1{&sh, false};
      std::thread t0([&] { eval_softmax_block(keys.party0, 0, c0, xs0, out0); });
      std::thread t1([&] { eval_softmax_block(keys.party1, 1, c1, xs1, out1); });
      t0.join();
      t1.join();
    }
  });
  double elems = static_cast<double>(batches * L);
  std::cout << "softmax batches=" << batches << " L=" << L << " time=" << secs
            << "s elems/s=" << elems / secs << "\n";
}

static void bench_layernorm(size_t batches, size_t L, int frac_bits) {
  LayerNormParams params;
  params.L = L;
  params.frac_bits = frac_bits;
  params.nr_iters = 1;
  std::mt19937_64 rng(88);
  auto keys = dealer_make_layernorm_keys(params, rng);
  std::uniform_real_distribution<double> dist(-2.0, 2.0);
  std::uniform_real_distribution<double> dist_g(0.5, 1.5);

  double secs = time_it([&] {
    for (size_t b = 0; b < batches; ++b) {
      std::vector<mpc::AddShare<R>> xs0(L), xs1(L), gs0(L), gs1(L), bs0(L), bs1(L), out0, out1;
      for (size_t i = 0; i < L; ++i) {
        int64_t val = to_fixed(dist(rng), frac_bits);
        uint64_t s0 = rng();
        uint64_t s1 = static_cast<uint64_t>(val) - s0;
        xs0[i] = {R(s0)};
        xs1[i] = {R(s1)};
        int64_t gval = to_fixed(dist_g(rng), frac_bits);
        uint64_t g0 = rng();
        uint64_t g1 = static_cast<uint64_t>(gval) - g0;
        gs0[i] = {R(g0)};
        gs1[i] = {R(g1)};
        int64_t bval = to_fixed(dist(rng), frac_bits);
        uint64_t b0 = rng();
        uint64_t b1 = static_cast<uint64_t>(bval) - b0;
        bs0[i] = {R(b0)};
        bs1[i] = {R(b1)};
      }
      LocalChan::Shared sh;
      LocalChan c0{&sh, true}, c1{&sh, false};
      std::thread t0([&] {
        eval_layernorm_block(keys.party0, 0, c0, xs0, &gs0, &bs0, out0);
      });
      std::thread t1([&] {
        eval_layernorm_block(keys.party1, 1, c1, xs1, &gs1, &bs1, out1);
      });
      t0.join();
      t1.join();
    }
  });
  double elems = static_cast<double>(batches * L);
  std::cout << "layernorm batches=" << batches << " L=" << L << " time=" << secs
            << "s elems/s=" << elems / secs << "\n";
}

int main(int argc, char** argv) {
  std::string gate = "silu";
  size_t N = 1000;
  size_t L = 128;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--gate=", 0) == 0) gate = arg.substr(7);
    else if (arg.rfind("--N=", 0) == 0) N = static_cast<size_t>(std::strtoull(arg.c_str() + 4, nullptr, 10));
    else if (arg.rfind("--L=", 0) == 0) L = static_cast<size_t>(std::strtoull(arg.c_str() + 4, nullptr, 10));
  }
  int frac_bits = 16;
  if (gate == "softmax") {
    bench_softmax(N, L, frac_bits);
  } else if (gate == "layernorm") {
    bench_layernorm(N, L, frac_bits);
  } else {
    bench_scalar(gate, N, frac_bits);
  }
  return 0;
}
