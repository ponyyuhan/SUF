#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "proto/backend_clear.hpp"
#include "proto/beaver.hpp"
#include "proto/channel.hpp"
#include "proto/gelu_online_step_dcf.hpp"
#include "proto/gelu_spline_dealer.hpp"
#include "proto/myl7_fss_backend.hpp"
#include "proto/reluars_dealer.hpp"
#include "proto/reluars_online_complete.hpp"

using namespace proto;

struct LocalChan : IChannel {
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
    std::unique_lock<std::mutex> lk(shared->m);
    auto& q = is0 ? shared->q0to1 : shared->q1to0;
    q.push(std::move(buf));
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

void demo_reluars() {
  std::cout << "=== ReluARS end-to-end (clear backend) ===\n";
  ClearBackend backend;
  BeaverDealer dealer;
  ReluARSParams params;
  params.f = 4;
  params.delta.fill(0);
  auto keys_off = ReluARSDealer::keygen(params, backend, dealer);

  // Build online keys
  ReluARSPartyKeyOnline k0, k1;
  auto fill = [&](ReluARSPartyKeyOnline& ko, const ReluARSPartyKey& src) {
    ko.f = src.params.f;
    ko.r_in_share = src.r_in_share;
    ko.r_hi_share = src.r_hi_share;
    ko.r_out_share = src.r_out_share;
    ko.wrap_sign = src.wrap_sign;
    ko.dcf_hat_lt_r = src.dcf_hat_lt_r;
    ko.dcf_hat_lt_r_plus_2p63 = src.dcf_hat_lt_r_plus_2p63;
    ko.dcf_low_lt_r_low = src.dcf_low_lt_r_low;
    ko.dcf_low_lt_r_low_plus1 = src.dcf_low_lt_r_low_plus1;
    ko.params.delta = src.params.delta;
    ko.triples64 = src.triples64;
  };
  fill(k0, keys_off.k0);
  fill(k1, keys_off.k1);

  u64 r_in = add_mod(keys_off.k0.r_in_share, keys_off.k1.r_in_share);
  std::vector<int64_t> xs = { -20, -1, 0, 5, 33 };
  for (auto x_signed : xs) {
    u64 x = static_cast<u64>(x_signed);
    u64 hatx = add_mod(x, r_in);

    LocalChan::Shared sh;
    LocalChan c0{&sh, true}, c1{&sh, false};
    ReluARSOut out0, out1;
    std::thread t0([&]{ out0 = eval_reluars_one(0, backend, c0, k0, hatx); });
    std::thread t1([&]{ out1 = eval_reluars_one(1, backend, c1, k1, hatx); });
    t0.join(); t1.join();

    u64 haty = add_mod(out0.haty_share, out1.haty_share);
    u64 r_out = add_mod(k0.r_out_share, k1.r_out_share);
    u64 y = sub_mod(haty, r_out);
    int64_t y_plain = static_cast<int64_t>(x_signed >= 0 ? (x_signed >> params.f) : 0);
    std::cout << "x=" << x_signed << " -> y=" << static_cast<int64_t>(y)
              << " expected=" << y_plain << "\n";
  }
}

void demo_gelu() {
  std::cout << "\n=== GeLU (step-DCF) end-to-end (clear backend) ===\n";
  ClearBackend backend;
  BeaverDealer dealer;

  GeluSplineParams params;
  params.f = 4;
  params.d = 1;
  params.T = 32;
  params.a = { -static_cast<int64_t>(params.T), 0, static_cast<int64_t>(params.T) };
  params.coeffs = {
      {0,0},    // [-T,0): delta=0
      {0,1},    // [0,T): delta=x
  };
  auto keys_off = GeluSplineDealer::keygen(params, backend, dealer);

  GeluStepDCFPartyKey k0, k1;
  auto fill = [&](GeluStepDCFPartyKey& ko, const GeluSplinePartyKey& src) {
    ko.d = src.params.d;
    ko.r_in_share = src.r_in_share;
    ko.r_out_share = src.r_out_share;
    ko.wrap_sign = src.wrap_sign;
    ko.dcf_hat_lt_r = src.dcf_hat_lt_r;
    ko.dcf_hat_lt_r_plus_2p63 = src.dcf_hat_lt_r_plus_2p63;
    ko.base_coeff = src.base_coeff;
    ko.cuts.clear();
    for (const auto& c : src.cuts) {
      StepCutVec sc;
      sc.dcf_key = c.party0.dcf_key;  // party0/party1 same start/delta; dealer stored both
      sc.delta_vec = c.delta;
      ko.cuts.push_back(sc);
    }
    ko.triples64 = src.triples64;
  };
  fill(k0, keys_off.k0);
  fill(k1, keys_off.k1);

  u64 r_in = add_mod(keys_off.k0.r_in_share, keys_off.k1.r_in_share);
  std::vector<int64_t> xs = { -40, -10, 0, 8, 40 };
  for (auto x_signed : xs) {
    u64 x = static_cast<u64>(x_signed);
    u64 hatx = add_mod(x, r_in);

    LocalChan::Shared sh;
    LocalChan c0{&sh, true}, c1{&sh, false};
    GeluOut out0, out1;
    std::thread t0([&]{ out0 = eval_gelu_step_dcf_one(0, backend, c0, k0, hatx); });
    std::thread t1([&]{ out1 = eval_gelu_step_dcf_one(1, backend, c1, k1, hatx); });
    t0.join(); t1.join();

    u64 haty = add_mod(out0.haty_share, out1.haty_share);
    u64 r_out = add_mod(k0.r_out_share, k1.r_out_share);
    u64 y = sub_mod(haty, r_out);

    // Plain reference: y = x_plus + delta; delta = piecewise {0, x in [0,T): x}
    int64_t plain = x_signed;
    int64_t delta = (x_signed >= 0 && x_signed < static_cast<int64_t>(params.T)) ? x_signed : 0;
    int64_t xplus = (x_signed >= 0) ? x_signed : 0;
    int64_t expected = xplus + delta;
    std::cout << "x=" << x_signed << " -> y=" << static_cast<int64_t>(y)
              << " expected=" << expected << "\n";
  }
}

int main() {
  demo_reluars();
  demo_gelu();
  return 0;
}
