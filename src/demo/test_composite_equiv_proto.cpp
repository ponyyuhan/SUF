#include <random>
#include <thread>
#include <condition_variable>
#include <queue>
#include "suf/ref_eval.hpp"
#include "suf/validate.hpp"
#include "gates/composite_fss.hpp"
#include "proto/backend_clear.hpp"
#include "proto/channel.hpp"

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
    poly.coeffs = {c0, 2};  // y = c0 + 2x
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
  auto suf = make_sample_suf();
  validate_suf(suf);
  proto::ClearBackend backend;
  std::mt19937_64 rng(42);
  const size_t N = 200;
  auto kp = gates::composite_gen_backend(suf, backend, rng, N);

  LocalChan::Shared sh;
  for (size_t t = 0; t < N; t++) {
    uint64_t x = rng();
    uint64_t hatx = x + kp.k0.r_in_share + kp.k1.r_in_share;
    uint64_t y0 = 0, y1 = 0;
    uint64_t b0 = 0, b1 = 0;
    std::thread th1([&](){
      LocalChan ch{&sh, false};
      auto out1 = gates::composite_eval_share_backend(1, backend, ch, kp.k1, suf, hatx);
      y1 = out1[0];
      b1 = out1.size() > 1 ? out1[1] : 0;
    });
    LocalChan ch0{&sh, true};
    auto out0 = gates::composite_eval_share_backend(0, backend, ch0, kp.k0, suf, hatx);
    y0 = out0[0];
    b0 = out0.size() > 1 ? out0[1] : 0;
    th1.join();

    auto ref = suf::eval_suf_ref(suf, x);
    uint64_t y_expect = ref.arith[0] + kp.k0.r_out_share[0] + kp.k1.r_out_share[0];
    uint64_t b_expect = ref.bools.empty() ? 0ull : (ref.bools[0] ? 1ull : 0ull);
    if ((y0 + y1) != y_expect || ((b0 + b1) & 1ull) != b_expect) {
      throw std::runtime_error("composite equiv failed");
    }
  }
  return 0;
}
