#include <cassert>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <sstream>
#include <random>
#include <thread>
#include <vector>

#include "proto/backend_clear.hpp"
#include "proto/beaver.hpp"
#include "proto/reluars_dealer.hpp"
#include "proto/reluars_online_complete.hpp"
#include "proto/gelu_batch_step_dcf.hpp"
#include "gates/postproc_hooks.hpp"

// Simple local channel for two-party exchange.
struct LocalChan : proto::IChannel {
  struct Shared {
    std::mutex m;
    std::condition_variable cv;
    std::queue<std::vector<uint8_t>> q0to1, q1to0;
  };
  Shared* shared = nullptr;
  bool is0 = false;
  LocalChan() = default;
  LocalChan(Shared* s, bool is0_) : shared(s), is0(is0_) {}
  void send_bytes(const void* data, size_t n) override {
    std::vector<uint8_t> buf(n);
    std::memcpy(buf.data(), data, n);
    {
      std::lock_guard<std::mutex> lk(shared->m);
      auto& q = is0 ? shared->q0to1 : shared->q1to0;
      q.push(std::move(buf));
    }
    shared->cv.notify_all();
  }
  void recv_bytes(void* data, size_t n) override {
    std::unique_lock<std::mutex> lk(shared->m);
    auto& q = is0 ? shared->q1to0 : shared->q0to1;
    shared->cv.wait(lk, [&]{ return !q.empty(); });
    auto buf = std::move(q.front());
    q.pop();
    if (buf.size() != n) throw std::runtime_error("LocalChan size mismatch");
    std::memcpy(data, buf.data(), n);
  }
};

// Trivial channel for unused hooks.
struct NullChan : proto::IChannel {
  void send_bytes(const void*, size_t) override {}
  void recv_bytes(void*, size_t) override {}
};

static void test_reluars_postproc() {
  proto::ClearBackend backend;
  proto::ReluARSParams params;
  params.f = 12;
  std::mt19937_64 rng(42);
  for (auto& v : params.delta) v = rng();

  proto::BeaverDealer dealer;
  auto kp = proto::ReluARSDealer::keygen(params, backend, dealer);

  const size_t N = 8;
  const size_t need64 = N * proto::reluars_triples64_needed();
  auto top_up_triples = [&](std::vector<proto::BeaverTriple64Share>& v0,
                            std::vector<proto::BeaverTriple64Share>& v1) {
    while (v0.size() < need64) {
      auto [a, b] = dealer.gen_triple64();
      v0.push_back(a);
      v1.push_back(b);
    }
  };
  top_up_triples(kp.k0.triples64, kp.k1.triples64);

  std::vector<uint64_t> xs(N);
  for (auto& x : xs) x = rng();
  std::vector<uint64_t> hatx(N);
  for (size_t i = 0; i < N; i++) {
    hatx[i] = proto::add_mod(xs[i], proto::add_mod(kp.k0.r_in_share, kp.k1.r_in_share));
  }

  uint64_t r_in = proto::add_mod(kp.k0.r_in_share, kp.k1.r_in_share);
  uint64_t r_hi = proto::add_mod(kp.k0.r_hi_share, kp.k1.r_hi_share);
  uint64_t r_out = proto::add_mod(kp.k0.r_out_share, kp.k1.r_out_share);
  uint64_t wrap_sign = proto::add_mod(kp.k0.wrap_sign_share, kp.k1.wrap_sign_share) & 1ull;
  uint64_t mask_f = (params.f == 64) ? ~0ull : ((uint64_t(1) << params.f) - 1);
  uint64_t r_low = (params.f == 64) ? r_in : (r_in & mask_f);
  uint64_t r_low_plus1 = (params.f == 64) ? (r_low + 1) : ((r_low + 1) & mask_f);

  // Build inputs for hook from plaintext bits.
  std::vector<uint64_t> bool0(N * 3, 0), bool1(N * 3, 0);
  std::vector<uint64_t> arith0(N, kp.k0.r_out_share), arith1(N, kp.k1.r_out_share);
  std::vector<uint64_t> haty_expected(N, 0);
  for (size_t i = 0; i < N; i++) {
    uint64_t a = (hatx[i] < r_in) ? 1ull : 0ull;
    uint64_t thr2 = r_in + (uint64_t(1) << 63);
    uint64_t b = (hatx[i] < thr2) ? 1ull : 0ull;
    uint64_t na = 1ull - a;
    uint64_t u = b & na;
    uint64_t wrap_or = na | b;
    uint64_t w_plain = wrap_sign ? wrap_or : u;

    uint64_t hatz = proto::add_mod(hatx[i], (params.f == 0) ? 0ull : (uint64_t(1) << (params.f - 1)));
    uint64_t hatz_low = (params.f == 64) ? hatz : (hatz & mask_f);
    uint64_t t_plain = (hatz_low < r_low) ? 1ull : 0ull;
    uint64_t s_plain = (hatz_low < r_low_plus1) ? 1ull : 0ull;
    uint64_t d_plain = s_plain & (1ull - t_plain);

    uint64_t H = (params.f == 64) ? 0ull : (hatz >> params.f);
    uint64_t q = proto::sub_mod(proto::sub_mod(H, r_hi), t_plain);
    uint64_t y_plain = proto::mul_mod(w_plain, q);
    if (params.delta.size() >= 8) {
      size_t idx = (static_cast<size_t>(w_plain) << 2) |
                   (static_cast<size_t>(t_plain) << 1) |
                   static_cast<size_t>(d_plain);
      y_plain = proto::add_mod(y_plain, params.delta[idx]);
    }
    haty_expected[i] = proto::add_mod(y_plain, r_out);

    auto [w0, w1] = dealer.split_add(w_plain);
    auto [t0, t1] = dealer.split_add(t_plain);
    auto [d0, d1] = dealer.split_add(d_plain);
    bool0[i * 3 + 0] = w0;
    bool0[i * 3 + 1] = t0;
    bool0[i * 3 + 2] = d0;
    bool1[i * 3 + 0] = w1;
    bool1[i * 3 + 1] = t1;
    bool1[i * 3 + 2] = d1;
  }

  gates::ReluARSPostProc hook;
  hook.f = params.f;
  hook.delta.assign(std::begin(params.delta), std::end(params.delta));
  hook.r_hi_share = kp.k0.r_hi_share; // overwritten per party below
  hook.wrap_sign_share = kp.k0.wrap_sign_share;

  gates::ReluARSPostProc hook0 = hook;
  gates::ReluARSPostProc hook1 = hook;
  gates::PostProcHook& base_hook0 = hook0;
  gates::PostProcHook& base_hook1 = hook1;
  compiler::PortLayout layout;
  layout.bool_ports = {"w","t","d"};
  layout.arith_ports = {"y"};
  base_hook0.configure(layout);
  base_hook1.configure(layout);

  std::vector<uint64_t> out_share0(N, 0), out_share1(N, 0);
  {
    LocalChan::Shared sh;
    LocalChan c0{&sh, true}, c1{&sh, false};
    proto::BeaverMul64 mul0{0, c0, kp.k0.triples64, 0};
    proto::BeaverMul64 mul1{1, c1, kp.k1.triples64, 0};
    std::thread t1([&](){
      hook1.r_hi_share = kp.k1.r_hi_share;
      hook1.wrap_sign_share = kp.k1.wrap_sign_share;
      base_hook1.run_batch(1, c1, mul1, hatx.data(), arith1.data(), 1,
                           bool1.data(), 3, N, out_share1.data());
    });
    hook0.r_hi_share = kp.k0.r_hi_share;
    hook0.wrap_sign_share = kp.k0.wrap_sign_share;
    base_hook0.run_batch(0, c0, mul0, hatx.data(), arith0.data(), 1,
                         bool0.data(), 3, N, out_share0.data());
    t1.join();
  }

  for (size_t i = 0; i < N; i++) {
    uint64_t ref = haty_expected[i];
    uint64_t got = proto::add_mod(out_share0[i], out_share1[i]);
    if (ref != got) {
      uint64_t w_recon = proto::add_mod(bool0[i * 3], bool1[i * 3]);
      uint64_t t_recon = proto::add_mod(bool0[i * 3 + 1], bool1[i * 3 + 1]);
      uint64_t d_recon = proto::add_mod(bool0[i * 3 + 2], bool1[i * 3 + 2]);
      std::ostringstream oss;
      oss << "ReluARSPostProc mismatch at idx " << i
          << " ref=" << ref << " got=" << got
          << " w=" << w_recon << " t=" << t_recon << " d=" << d_recon;
      throw std::runtime_error(oss.str());
    }
  }
}

static void test_gelu_postproc() {
  const size_t N = 16;
  std::mt19937_64 rng(7);
  std::vector<uint64_t> x_plus0(N), x_plus1(N), delta0(N), delta1(N);
  for (size_t i = 0; i < N; i++) {
    uint64_t x = rng();
    uint64_t d = rng();
    x_plus0[i] = rng();
    x_plus1[i] = proto::sub_mod(x, x_plus0[i]);
    delta0[i] = rng();
    delta1[i] = proto::sub_mod(d, delta0[i]);
  }
  std::vector<uint64_t> arith0(N * 2), arith1(N * 2);
  for (size_t i = 0; i < N; i++) {
    arith0[i * 2 + 0] = x_plus0[i];
    arith0[i * 2 + 1] = delta0[i];
    arith1[i * 2 + 0] = x_plus1[i];
    arith1[i * 2 + 1] = delta1[i];
  }

  compiler::PortLayout layout;
  layout.arith_ports = {"x_plus", "delta"};

  gates::GeLUPostProc hook;
  gates::PostProcHook& base = hook;
  base.configure(layout);
  NullChan ch;
  proto::BeaverMul64 dummy{0, ch, std::vector<proto::BeaverTriple64Share>{}, 0};
  std::vector<uint64_t> out0(N), out1(N);
  base.run_batch(0, ch, dummy, nullptr, arith0.data(), 2, nullptr, 0, N, out0.data());
  base.run_batch(1, ch, dummy, nullptr, arith1.data(), 2, nullptr, 0, N, out1.data());

  for (size_t i = 0; i < N; i++) {
    uint64_t expect = proto::add_mod(proto::add_mod(x_plus0[i], x_plus1[i]),
                                     proto::add_mod(delta0[i], delta1[i]));
    uint64_t got = proto::add_mod(out0[i], out1[i]);
    if (expect != got) {
      throw std::runtime_error("GeLUPostProc mismatch at idx " + std::to_string(i));
    }
  }
}

int main() {
  try {
    test_reluars_postproc();
    test_gelu_postproc();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "test_postproc_hooks error: " << e.what() << "\n";
    return 1;
  }
}
