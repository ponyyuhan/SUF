#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include "suf/ref_eval.hpp"
#include "suf/validate.hpp"
#include "gates/composite_fss.hpp"
#include "proto/sigma_fast_backend_ext.hpp"
#include "proto/channel.hpp"

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
    if (buf.size() != n) throw std::runtime_error("recv_bytes size mismatch");
    std::memcpy(data, buf.data(), n);
  }
};

static suf::SUF<uint64_t> make_sample_suf() {
  suf::SUF<uint64_t> s;
  s.n_bits = 64;
  s.r_out = 1;
  s.l_out = 1;
  s.degree = 1;
  s.alpha = {0, 1000, 5000, std::numeric_limits<uint64_t>::max()};
  s.primitive_preds.push_back(suf::Pred_X_lt_const{2048}); // p0
  s.primitive_preds.push_back(suf::Pred_X_mod2f_lt{4, 7}); // p1
  auto make_piece = [&](uint64_t c0) {
    suf::SufPiece<uint64_t> p;
    suf::Poly<uint64_t> poly;
    poly.coeffs = {c0, 2};
    p.polys.push_back(poly);
    suf::BoolExpr b0{suf::BAnd{std::make_unique<suf::BoolExpr>(suf::BoolExpr{suf::BVar{0}}),
                               std::make_unique<suf::BoolExpr>(suf::BoolExpr{suf::BNot{std::make_unique<suf::BoolExpr>(suf::BoolExpr{suf::BVar{1}})}})}};
    p.bool_outs.push_back(std::move(b0));
    return p;
  };
  s.pieces.push_back(make_piece(1));
  s.pieces.push_back(make_piece(2));
  s.pieces.push_back(make_piece(3));
  return s;
}

int main() {
  try {
  proto::SigmaFastBackend backend;
  auto suf = make_sample_suf();
  validate_suf(suf);
  std::mt19937_64 rng(123);
  auto run_batch = [&](size_t N)->double {
    auto kp = gates::composite_gen_backend(suf, backend, rng, N);
    std::vector<uint64_t> xs(N);
    for (auto& v : xs) v = rng();
    LocalChan::Shared sh;
    std::vector<uint64_t> out0(N), out1(N);
    auto t0 = std::chrono::high_resolution_clock::now();
    std::thread th1([&](){
      LocalChan ch{&sh, false};
      for (size_t i = 0; i < N; i++) {
        uint64_t hatx = xs[i] + kp.k0.r_in_share + kp.k1.r_in_share;
        auto res = gates::composite_eval_share_backend(1, backend, ch, kp.k1, suf, hatx);
        out1[i] = res[0];
      }
    });
    LocalChan ch0{&sh, true};
    for (size_t i = 0; i < N; i++) {
      uint64_t hatx = xs[i] + kp.k0.r_in_share + kp.k1.r_in_share;
      auto res = gates::composite_eval_share_backend(0, backend, ch0, kp.k0, suf, hatx);
      out0[i] = res[0];
    }
    th1.join();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() * 1e6 / static_cast<double>(N);
  };

  for (size_t N : {static_cast<size_t>(1 << 12), static_cast<size_t>(1 << 14), static_cast<size_t>(1000000)}) {
    double ns_per = run_batch(N);
    std::cout << "SigmaFast gate bench (packed preds) N=" << N
              << " ns/elem=" << ns_per << "\n";
  }
  return 0;
  } catch (const std::exception& e) {
    std::cerr << "bench_sigmafast_gates error: " << e.what() << "\n";
    return 1;
  }
}
