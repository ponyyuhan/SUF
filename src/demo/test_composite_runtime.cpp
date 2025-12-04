#include <cassert>
#include <iostream>
#include <random>
#include <cstring>
#include "suf/ref_eval.hpp"
#include "suf/validate.hpp"
#include "gates/composite_fss.hpp"
#include "proto/myl7_fss_backend.hpp"
#include "proto/channel.hpp"
#include <condition_variable>
#include <queue>
#include <mutex>
#include <thread>
#include <limits>

using namespace suf;

// simple in-memory duplex channel
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

// Reuse sample SUF from previous tests.
static SUF<uint64_t> make_sample_suf() {
  SUF<uint64_t> s;
  s.n_bits = 64;
  s.r_out = 1;
  s.l_out = 1;
  s.degree = 1;
  s.alpha = {0, 1000, 5000, std::numeric_limits<uint64_t>::max()};
  s.primitive_preds.push_back(Pred_X_lt_const{2048}); // p0
  s.primitive_preds.push_back(Pred_X_mod2f_lt{4, 7}); // p1
  auto make_piece = [&](uint64_t c0) {
    SufPiece<uint64_t> p;
    Poly<uint64_t> poly;
    poly.coeffs = {c0, 2};  // y = c0 + 2x
    p.polys.push_back(poly);
    BoolExpr b0{BAnd{std::make_unique<BoolExpr>(BoolExpr{BVar{0}}),
                     std::make_unique<BoolExpr>(BoolExpr{BNot{std::make_unique<BoolExpr>(BoolExpr{BVar{1}})}})}};
    p.bool_outs.push_back(std::move(b0));
    return p;
  };
  s.pieces.push_back(make_piece(1));
  s.pieces.push_back(make_piece(2));
  s.pieces.push_back(make_piece(3));
  return s;
}

int main() {
  auto suf = make_sample_suf();
  validate_suf(suf);

  proto::Myl7FssBackend backend;
  std::mt19937_64 rng(2027);

  const size_t N = 128;
  auto kp = gates::composite_gen_backend(suf, backend, rng, N);

  LocalChan::Shared sh;
  for (size_t t = 0; t < N; t++) {
    uint64_t x = static_cast<uint64_t>(t);
    uint64_t hatx = x + kp.k0.r_in_share + kp.k1.r_in_share;

    uint64_t y0 = 0, y1 = 0;
    std::thread th1([&](){
      LocalChan ch{&sh, false};
      auto out1 = gates::composite_eval_share_backend(1, backend, ch, kp.k1, suf, hatx);
      y1 = out1[0];
    });
    LocalChan ch0{&sh, true};
    auto out0 = gates::composite_eval_share_backend(0, backend, ch0, kp.k0, suf, hatx);
    y0 = out0[0];
    th1.join();

    auto ref = eval_suf_ref(suf, x);
    uint64_t y_expect = ref.arith[0] + kp.k0.r_out_share[0] + kp.k1.r_out_share[0];
    if ((y0 + y1) != y_expect) {
      std::cerr << "mismatch at x=" << x << " y0=" << y0 << " y1=" << y1 << " expect=" << y_expect << "\n";
      return 1;
    }
  }

  std::cout << "Composite runtime (ClearBackend) equivalence passed\n";
  return 0;
}
