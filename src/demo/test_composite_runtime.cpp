#include <cassert>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <queue>
#include <random>
#include <thread>
#include "gates/composite_fss.hpp"
#include "proto/channel.hpp"
#include "proto/myl7_fss_backend.hpp"
#include "runtime/pfss_superbatch.hpp"
#include "suf/ref_eval.hpp"
#include "suf/validate.hpp"

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

// Reuse sample new from previous tests.
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

static SUF<uint64_t> make_offset_suf(uint64_t c0) {
  auto s = make_sample_suf();
  for (auto& piece : s.pieces) {
    if (!piece.polys.empty() && !piece.polys[0].coeffs.empty()) {
      piece.polys[0].coeffs[0] = c0;
    }
  }
  return s;
}

int main() {
  auto suf1 = make_sample_suf();
  auto suf2 = make_offset_suf(5);
  validate_suf(suf1);
  validate_suf(suf2);

  proto::Myl7FssBackend backend;
  std::mt19937_64 rng(2027);

  const size_t N = 128;
  auto kp1 = gates::composite_gen_backend(suf1, backend, rng, N);
  auto kp2 = gates::composite_gen_backend(suf2, backend, rng, N);

  std::vector<uint64_t> hatx1, hatx2;
  hatx1.reserve(N);
  hatx2.reserve(N);
  uint64_t mask1 = kp1.k0.r_in_share + kp1.k1.r_in_share;
  uint64_t mask2 = kp2.k0.r_in_share + kp2.k1.r_in_share;
  for (size_t t = 0; t < N; t++) {
    uint64_t x = static_cast<uint64_t>(t);
    hatx1.push_back(x + mask1);
    hatx2.push_back(x + mask2);
  }

  runtime::PfssSuperBatch batch0, batch1;
  gates::NoopPostProc noop;

  LocalChan::Shared sh;
  std::vector<uint64_t> out1_0(N), out1_1(N), out2_0(N), out2_1(N);
  std::thread t1([&](){
    LocalChan ch{&sh, false};
    batch1.enqueue_composite({&suf1, &kp1.k1, &noop, hatx1, nn::TensorView<uint64_t>(out1_1.data(), {N})});
    batch1.enqueue_composite({&suf2, &kp2.k1, &noop, hatx2, nn::TensorView<uint64_t>(out2_1.data(), {N})});
    batch1.flush_and_finalize(1, backend, ch);
  });
  {
    LocalChan ch{&sh, true};
    batch0.enqueue_composite({&suf1, &kp1.k0, &noop, hatx1, nn::TensorView<uint64_t>(out1_0.data(), {N})});
    batch0.enqueue_composite({&suf2, &kp2.k0, &noop, hatx2, nn::TensorView<uint64_t>(out2_0.data(), {N})});
    batch0.flush_and_finalize(0, backend, ch);
  }
  t1.join();

  for (size_t i = 0; i < N; ++i) {
    auto ref1 = eval_suf_ref(suf1, static_cast<uint64_t>(i));
    auto ref2 = eval_suf_ref(suf2, static_cast<uint64_t>(i));
    uint64_t y1 = out1_0[i] + out1_1[i];
    uint64_t y2 = out2_0[i] + out2_1[i];
    if (y1 != ref1.arith[0] || y2 != ref2.arith[0]) {
      std::cerr << "mismatch at i=" << i << " y1=" << y1 << " exp1=" << ref1.arith[0]
                << " y2=" << y2 << " exp2=" << ref2.arith[0] << "\n";
      return 1;
    }
  }

  std::cout << "Composite runtime batched flush passed\n";
  return 0;
}
