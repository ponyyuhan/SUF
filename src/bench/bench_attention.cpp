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

#include "nn/attention_block.hpp"
#include "nn/kv_cache.hpp"
#include "mpc/net.hpp"

using namespace nn;

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

static inline uint64_t to_ring(int64_t v) { return static_cast<uint64_t>(v); }

static void split_shares(const std::vector<int64_t>& plain,
                         std::mt19937_64& rng,
                         int frac_bits,
                         std::vector<uint64_t>& s0,
                         std::vector<uint64_t>& s1) {
  s0.resize(plain.size());
  s1.resize(plain.size());
  uint64_t mask = (frac_bits >= 63) ? ~uint64_t(0) : ((uint64_t(1) << frac_bits) - 1);
  for (size_t i = 0; i < plain.size(); ++i) {
    uint64_t r = rng();
    if (frac_bits > 0) r &= ~mask;
    s0[i] = r;
    s1[i] = to_ring(plain[i] - static_cast<int64_t>(r));
  }
}

int main(int argc, char** argv) {
  size_t B = 1, T = 4, D = 64, H = 4;
  int repeats = 8;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--B=", 0) == 0) B = std::strtoull(arg.c_str() + 4, nullptr, 10);
    else if (arg.rfind("--T=", 0) == 0) T = std::strtoull(arg.c_str() + 4, nullptr, 10);
    else if (arg.rfind("--D=", 0) == 0) D = std::strtoull(arg.c_str() + 4, nullptr, 10);
    else if (arg.rfind("--H=", 0) == 0) H = std::strtoull(arg.c_str() + 4, nullptr, 10);
    else if (arg.rfind("--repeats=", 0) == 0) repeats = std::atoi(arg.c_str() + 10);
  }
  if (H == 0 || D % H != 0) {
    std::cerr << "Invalid head config\n";
    return 1;
  }
  size_t Dh = D / H;

  AttentionConfig cfg;
  cfg.D = D;
  cfg.H = H;
  cfg.Dh = Dh;
  cfg.S_max = T;
  cfg.frac_bits = 8;

  std::mt19937_64 rng(42);
  std::uniform_int_distribution<int64_t> dist_x(-64, 64);
  std::uniform_int_distribution<int64_t> dist_w(-32, 32);

  std::vector<int64_t> X_plain(B * T * D), Wqkv(D * D * 3), Wout(D * D);
  for (auto& v : X_plain) v = dist_x(rng);
  for (auto& v : Wqkv) v = dist_w(rng);
  for (auto& v : Wout) v = dist_w(rng);

  std::vector<uint64_t> X0, X1;
  split_shares(X_plain, rng, cfg.frac_bits, X0, X1);

  auto run_once = [&](const std::vector<uint64_t>& in0,
                      const std::vector<uint64_t>& in1) {
    KVCache cache0(B, cfg.H, cfg.S_max, cfg.Dh), cache1(B, cfg.H, cfg.S_max, cfg.Dh);
    std::vector<uint64_t> Y0(B * T * D), Y1(B * T * D);
    LocalChan::Shared sh;
    LocalChan c0(&sh, true), c1(&sh, false);
    std::thread th1([&] {
      attention_forward(cfg,
                        1,
                        c1,
                        view3(const_cast<uint64_t*>(in1.data()), B, T, D),
                        view2(Wqkv.data(), D, D * 3),
                        view2(Wout.data(), D, D),
                        cache1,
                        view3(Y1.data(), B, T, D));
    });
    attention_forward(cfg,
                      0,
                      c0,
                      view3(const_cast<uint64_t*>(in0.data()), B, T, D),
                      view2(Wqkv.data(), D, D * 3),
                      view2(Wout.data(), D, D),
                      cache0,
                      view3(Y0.data(), B, T, D));
    th1.join();
  };

  // Warmup
  run_once(X0, X1);

  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeats; ++r) run_once(X0, X1);
  auto end = std::chrono::steady_clock::now();
  double secs = std::chrono::duration<double>(end - start).count();
  std::cout << "attention B=" << B << " T=" << T << " D=" << D
            << " H=" << H << " repeats=" << repeats
            << " time=" << secs << "s"
            << " per_call=" << (secs / repeats) << "s\n";
  return 0;
}
