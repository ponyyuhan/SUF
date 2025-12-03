#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <thread>
#include <vector>

#include "compiler/suf_to_pfss.hpp"
#include "gates/gate_api.hpp"
#include "gates/relu_gate.hpp"
#include "gates/reluars_gate.hpp"
#include "gates/gelu_spline_gate.hpp"
#include "mpc/beaver.hpp"
#include "pfss/backend_cleartext.hpp"

using R = core::Z2n<64>;

// In-memory synchronous channel for two-party demos.
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

struct PartyResult {
  gates::GateEvalResult<R> gate;
};

void run_once(uint64_t x,
              const compiler::CompiledSUFKeys& keys,
              pfss::Backend<pfss::PredPayload>& pred_backend,
              pfss::Backend<pfss::CoeffPayload>& coeff_backend,
              const pfss::PublicParams& pp_pred,
              const pfss::PublicParams& pp_coeff) {
  uint64_t r_in = keys.r_in_share0 + keys.r_in_share1;
  uint64_t x_hat = x + r_in;

  std::mt19937_64 rng(12345);
  uint64_t x0 = rng();
  uint64_t x1 = x - x0;
  mpc::AddShare<R> xs0{R(x0)};
  mpc::AddShare<R> xs1{R(x1)};

  auto triple_pair = mpc::dealer_make_tripleA<R>(rng);
  std::vector<mpc::BeaverTripleA<R>> triples0{triple_pair.first};
  std::vector<mpc::BeaverTripleA<R>> triples1{triple_pair.second};

  LocalChan::Shared shared;
  LocalChan ch0{&shared, true};
  LocalChan ch1{&shared, false};

  PartyResult r0, r1;
  std::thread t0([&] {
    r0.gate = gates::eval_compiled_suf_gate<R>(
        0, ch0, pred_backend, coeff_backend, pp_pred, pp_coeff, keys, x_hat, xs0, triples0);
  });
  std::thread t1([&] {
    r1.gate = gates::eval_compiled_suf_gate<R>(
        1, ch1, pred_backend, coeff_backend, pp_pred, pp_coeff, keys, x_hat, xs1, triples1);
  });
  t0.join();
  t1.join();

  if (r0.gate.y_hat_shares.empty() || r1.gate.y_hat_shares.empty()) {
    std::cerr << "gate evaluation failed\n";
    return;
  }

  uint64_t y_hat = r0.gate.y_hat_shares[0].s.v + r1.gate.y_hat_shares[0].s.v;
  uint64_t r_out = keys.r_out_share0[0] + keys.r_out_share1[0];
  uint64_t y = y_hat - r_out;

  int64_t signed_x = static_cast<int64_t>(x);
  uint64_t expected = signed_x < 0 ? 0ull : x;

  std::cout << "x=" << signed_x << " -> masked y=" << y_hat
            << " unmasked y=" << y << " expected=" << expected << "\n";
}

int main() {
  pfss::CleartextBackendPred pred_backend;
  pfss::CleartextBackendCoeff coeff_backend;
  auto pp_pred = pred_backend.setup(128);
  auto pp_coeff = coeff_backend.setup(128);

  auto F = gates::make_relu_suf_64();
  std::mt19937_64 dealer_rng(42);
  auto keys = compiler::dealer_compile_suf_gate(pred_backend, coeff_backend, pp_pred, pp_coeff, F,
                                                dealer_rng);

  run_once(5, keys, pred_backend, coeff_backend, pp_pred, pp_coeff);
  run_once(static_cast<uint64_t>(-3), keys, pred_backend, coeff_backend, pp_pred, pp_coeff);
  return 0;
}
