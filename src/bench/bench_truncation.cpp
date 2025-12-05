#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "gates/ars_faithful_gate.hpp"
#include "gates/gapars_gate.hpp"
#include "gates/trunc_faithful_gate.hpp"
#include "mpc/net.hpp"

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

template <typename Keys, typename EvalFn>
double run_bench(const Keys& keys,
                 const EvalFn& eval,
                 const std::vector<mpc::AddShare<core::Z2n<64>>>& xs0,
                 const std::vector<mpc::AddShare<core::Z2n<64>>>& xs1,
                 int iters) {
  auto run_once = [&]() {
    LocalChan::Shared sh;
    LocalChan c0(&sh, true), c1(&sh, false);
    std::vector<mpc::AddShare<core::Z2n<64>>> ys0, ys1;
    std::thread th([&] {
      ys1 = eval(keys.party1, 1, c1, xs1);
    });
    ys0 = eval(keys.party0, 0, c0, xs0);
    th.join();
  };

  run_once();  // warmup
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    run_once();
  }
  auto end = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(end - start).count();
  return ms / static_cast<double>(iters);
}

int main(int argc, char** argv) {
  std::string mode = "tr";
  int frac_bits = 8;
  size_t N = 1 << 12;
  int iters = 5;

  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg.rfind("--mode=", 0) == 0) {
      mode = arg.substr(7);
    } else if (arg.rfind("--frac_bits=", 0) == 0) {
      frac_bits = std::stoi(arg.substr(12));
    } else if (arg.rfind("--N=", 0) == 0) {
      N = static_cast<size_t>(std::stoul(arg.substr(4)));
    } else if (arg.rfind("--iters=", 0) == 0) {
      iters = std::stoi(arg.substr(8));
    }
  }

  std::mt19937_64 rng(42);
  std::uniform_int_distribution<int64_t> dist_signed(-1'000'000, 1'000'000);
  std::uniform_int_distribution<int64_t> dist_unsigned(0, 1'000'000);

  std::vector<int64_t> plain(N);
  if (mode == "tr") {
    for (auto& v : plain) v = dist_unsigned(rng);
  } else {
    for (auto& v : plain) v = dist_signed(rng);
  }

  std::vector<uint64_t> x0(N), x1(N);
  for (size_t i = 0; i < N; ++i) {
    uint64_t r = rng();
    x0[i] = r;
    x1[i] = static_cast<uint64_t>(plain[i] - static_cast<int64_t>(r));
  }

  std::vector<mpc::AddShare<core::Z2n<64>>> xs0(N), xs1(N);
  for (size_t i = 0; i < N; ++i) {
    xs0[i] = {core::Z2n<64>(x0[i])};
    xs1[i] = {core::Z2n<64>(x1[i])};
  }

  double avg_ms = 0.0;
  if (mode == "tr") {
    gates::TruncParams params{frac_bits};
    auto keys = gates::dealer_make_trunc_keys(params, N, rng);
    avg_ms = run_bench(keys, gates::eval_trunc_faithful, xs0, xs1, iters);
  } else if (mode == "gapars") {
    gates::GapArsParams params{frac_bits};
    auto keys = gates::dealer_make_gapars_keys(params, N, rng);
    avg_ms = run_bench(keys, gates::eval_gapars, xs0, xs1, iters);
  } else {
    gates::ArsParams params{frac_bits};
    auto keys = gates::dealer_make_ars_keys(params, N, rng);
    avg_ms = run_bench(keys, gates::eval_ars_faithful, xs0, xs1, iters);
  }

  double elems_per_sec = (static_cast<double>(N) / (avg_ms / 1000.0));
  std::cout << "mode=" << mode
            << " N=" << N
            << " frac_bits=" << frac_bits
            << " avg_ms=" << avg_ms
            << " elems/s=" << static_cast<uint64_t>(elems_per_sec)
            << "\n";
  return 0;
}
