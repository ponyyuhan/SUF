#include <cassert>
#include <condition_variable>
#include <cstring>
#include <cstdint>
#include <exception>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

#include "gates/composite_fss.hpp"
#include "gates/ars_faithful_gate.hpp"
#include "gates/gapars_gate.hpp"
#include "gates/trunc_faithful_gate.hpp"
#include "mpc/net.hpp"
#include "proto/backend_clear.hpp"
#include "proto/channel.hpp"
#include "proto/pfss_backend.hpp"
#include "gates/postproc_hooks.hpp"

using gates::TruncKeys;
using gates::TruncParams;
using gates::dealer_make_trunc_keys;
using gates::eval_trunc_faithful;
using gates::ArsKeys;
using gates::ArsParams;
using gates::dealer_make_ars_keys;
using gates::eval_ars_faithful;
using gates::GapArsKeys;
using gates::GapArsParams;
using gates::dealer_make_gapars_keys;
using gates::eval_gapars;

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

struct ProtoLocalChan : proto::IChannel {
  struct Shared {
    std::mutex m;
    std::condition_variable cv;
    std::queue<std::vector<uint8_t>> q0to1, q1to0;
  };
  Shared* s = nullptr;
  bool is0 = false;
  ProtoLocalChan() = default;
  ProtoLocalChan(Shared* sh, bool p) : s(sh), is0(p) {}
  void send_bytes(const void* data, size_t n) override {
    std::vector<uint8_t> buf(static_cast<const uint8_t*>(data),
                             static_cast<const uint8_t*>(data) + n);
    std::unique_lock<std::mutex> lk(s->m);
    auto& q = is0 ? s->q0to1 : s->q1to0;
    q.push(std::move(buf));
    s->cv.notify_all();
  }
  void recv_bytes(void* data, size_t n) override {
    std::unique_lock<std::mutex> lk(s->m);
    auto& q = is0 ? s->q1to0 : s->q0to1;
    s->cv.wait(lk, [&]{ return !q.empty(); });
    auto buf = std::move(q.front());
    q.pop();
    lk.unlock();
    if (buf.size() != n) throw std::runtime_error("ProtoLocalChan: size mismatch");
    std::memcpy(data, buf.data(), n);
  }
};

static inline uint64_t to_ring(int64_t v) { return static_cast<uint64_t>(v); }
static inline int64_t to_signed(uint64_t v) { return static_cast<int64_t>(v); }

static void test_trunc_cases(int frac_bits, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int64_t> dist(0, 1'000'000);
  size_t n = 32;
  std::vector<int64_t> plain(n);
  for (auto& v : plain) v = dist(rng);
  // Add a few edge values near the carry boundary for the chosen frac bits.
  plain[0] = (1ll << frac_bits) - 1;
  plain[1] = (1ll << frac_bits);
  plain[2] = (1ll << frac_bits) + 1;
  plain[3] = (1ll << (frac_bits + 1)) - 1;

  // Share inputs.
  std::vector<uint64_t> x0(n), x1(n);
  for (size_t i = 0; i < n; ++i) {
    uint64_t r = rng();
    x0[i] = r;
    x1[i] = to_ring(plain[i] - static_cast<int64_t>(r));
  }

  TruncParams params{frac_bits};
  TruncKeys keys = dealer_make_trunc_keys(params, n, rng);

  std::vector<mpc::AddShare<core::Z2n<64>>> xs0(n), xs1(n);
  for (size_t i = 0; i < n; ++i) {
    xs0[i] = {core::Z2n<64>(x0[i])};
    xs1[i] = {core::Z2n<64>(x1[i])};
  }

  LocalChan::Shared sh;
  LocalChan c0(&sh, true), c1(&sh, false);
  std::vector<mpc::AddShare<core::Z2n<64>>> ys0, ys1;

  std::thread th1([&] {
    ys1 = eval_trunc_faithful(keys.party1, 1, c1, xs1);
  });
  ys0 = eval_trunc_faithful(keys.party0, 0, c0, xs0);
  th1.join();

  for (size_t i = 0; i < n; ++i) {
    uint64_t out_mask = keys.party0.out_mask[i] + keys.party1.out_mask[i];
    uint64_t opened = ys0[i].s.v + ys1[i].s.v - out_mask;
    uint64_t expect = static_cast<uint64_t>(plain[i]) >> frac_bits;
    assert(opened == expect);
  }
  std::cout << "Exact truncation ok for frac_bits=" << frac_bits << "\n";
}

static void test_ars_cases(int frac_bits, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int64_t> dist(-1'000'000, 1'000'000);
  size_t n = 32;
  std::vector<int64_t> plain(n);
  for (auto& v : plain) v = dist(rng);

  std::vector<uint64_t> x0(n), x1(n);
  for (size_t i = 0; i < n; ++i) {
    uint64_t r = rng();
    x0[i] = r;
    x1[i] = to_ring(plain[i] - static_cast<int64_t>(r));
  }

  ArsParams params{frac_bits};
  ArsKeys keys = dealer_make_ars_keys(params, n, rng);

  std::vector<mpc::AddShare<core::Z2n<64>>> xs0(n), xs1(n);
  for (size_t i = 0; i < n; ++i) {
    xs0[i] = {core::Z2n<64>(x0[i])};
    xs1[i] = {core::Z2n<64>(x1[i])};
  }

  LocalChan::Shared sh;
  LocalChan c0(&sh, true), c1(&sh, false);
  std::vector<mpc::AddShare<core::Z2n<64>>> ys0, ys1;
  std::thread th1([&] {
    ys1 = eval_ars_faithful(keys.party1, 1, c1, xs1);
  });
  ys0 = eval_ars_faithful(keys.party0, 0, c0, xs0);
  th1.join();

  for (size_t i = 0; i < n; ++i) {
    int64_t out_mask = to_signed(keys.party0.out_mask[i] + keys.party1.out_mask[i]);
    int64_t opened = to_signed(ys0[i].s.v + ys1[i].s.v) - out_mask;
    int64_t expect = plain[i] >> frac_bits;
    assert(opened == expect);
  }
  std::cout << "ARS exact for frac_bits=" << frac_bits << "\n";
}

static void test_gapars_cases(int frac_bits, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int64_t> dist(-500'000, 500'000);
  size_t n = 16;
  std::vector<int64_t> plain(n);
  for (auto& v : plain) v = dist(rng);

  std::vector<uint64_t> x0(n), x1(n);
  for (size_t i = 0; i < n; ++i) {
    uint64_t r = rng();
    x0[i] = r;
    x1[i] = to_ring(plain[i] - static_cast<int64_t>(r));
  }

  GapArsParams params{frac_bits};
  GapArsKeys keys = dealer_make_gapars_keys(params, n, rng);

  std::vector<mpc::AddShare<core::Z2n<64>>> xs0(n), xs1(n);
  for (size_t i = 0; i < n; ++i) {
    xs0[i] = {core::Z2n<64>(x0[i])};
    xs1[i] = {core::Z2n<64>(x1[i])};
  }

  LocalChan::Shared sh;
  LocalChan c0(&sh, true), c1(&sh, false);
  std::vector<mpc::AddShare<core::Z2n<64>>> ys0, ys1;
  std::thread th1([&] {
    ys1 = eval_gapars(keys.party1, 1, c1, xs1);
  });
  ys0 = eval_gapars(keys.party0, 0, c0, xs0);
  th1.join();

  for (size_t i = 0; i < n; ++i) {
    int64_t out_mask = to_signed(keys.party0.out_mask[i] + keys.party1.out_mask[i]);
    int64_t opened = to_signed(ys0[i].s.v + ys1[i].s.v) - out_mask;
    int64_t expect = plain[i] >> frac_bits;
    assert(opened == expect);
  }
  std::cout << "GapARS (faithful path) exact for frac_bits=" << frac_bits << "\n";
}

int main() {
  std::fprintf(stderr, "enter main\n");
  try {
    std::cout << "Start trunc tests" << std::endl;
    test_trunc_cases(/*frac_bits=*/8, /*seed=*/123);
    std::cout << "trunc frac8 ok" << std::endl;
    test_trunc_cases(/*frac_bits=*/4, /*seed=*/321);
    std::cout << "trunc frac4 ok" << std::endl;
    test_ars_cases(/*frac_bits=*/8, /*seed=*/555);
    std::cout << "ars frac8 ok" << std::endl;
    test_ars_cases(/*frac_bits=*/5, /*seed=*/777);
    std::cout << "ars frac5 ok" << std::endl;
    test_gapars_cases(/*frac_bits=*/6, /*seed=*/999);
    std::cout << "gapars frac6 ok" << std::endl;
    // Composite + postproc path smoke tests.
    auto run_composite = [&](compiler::GateKind kind, int frac_bits, bool signed_input) {
      std::cout << "Composite path kind=" << static_cast<int>(kind)
                << " frac_bits=" << frac_bits
                << " signed=" << signed_input << "\n";
      const size_t N = 16;
      std::mt19937_64 rng_local(2025 + frac_bits);
      std::vector<int64_t> plain(N);
      if (signed_input) {
        std::uniform_int_distribution<int64_t> dist(-500'000, 500'000);
        for (auto& v : plain) v = dist(rng_local);
      } else {
        std::uniform_int_distribution<int64_t> dist(0, 500'000);
        for (auto& v : plain) v = dist(rng_local);
      }

      proto::ClearBackend backend;
      suf::SUF<uint64_t> trunc_suf;
      auto kp = gates::composite_gen_trunc_gate(backend, rng_local, frac_bits, kind, /*batch_N=*/N, &trunc_suf);
      const auto& compiled = kp.k0.compiled;
      std::vector<uint64_t> hatx(N);
      for (size_t i = 0; i < N; i++) {
        hatx[i] = static_cast<uint64_t>(plain[i]) + compiled.r_in;
      }
      gates::CompositeBatchInput in{hatx.data(), N};

      gates::CompositeBatchOutput out0, out1;
      gates::FaithfulTruncPostProc tr_hook0, tr_hook1;
      gates::FaithfulArsPostProc ars_hook0, ars_hook1;
      gates::GapArsPostProc gap_hook0, gap_hook1;
      gates::PostProcHook* hook0 = nullptr;
      gates::PostProcHook* hook1 = nullptr;
      if (kind == compiler::GateKind::FaithfulTR) {
        tr_hook0.f = tr_hook1.f = frac_bits;
        hook0 = &tr_hook0;
        hook1 = &tr_hook1;
      } else if (kind == compiler::GateKind::FaithfulARS) {
        ars_hook0.f = ars_hook1.f = frac_bits;
        hook0 = &ars_hook0;
        hook1 = &ars_hook1;
      } else {
        gap_hook0.f = gap_hook1.f = frac_bits;
        hook0 = &gap_hook0;
        hook1 = &gap_hook1;
      }

      ProtoLocalChan::Shared sh;
      ProtoLocalChan c0(&sh, true), c1(&sh, false);
      std::cout << " launching composite eval threads\n";
      std::exception_ptr thread_exc;
      std::thread t1([&](){
        try {
          auto k1 = kp.k1;
          out1 = gates::composite_eval_batch_with_postproc(1, backend, c1, k1, trunc_suf, in, *hook1);
        } catch (...) {
          thread_exc = std::current_exception();
        }
      });
      auto k0 = kp.k0;
      out0 = gates::composite_eval_batch_with_postproc(0, backend, c0, k0, trunc_suf, in, *hook0);
      std::cout << " waiting for thread\n";
      t1.join();
      std::cout << " composite eval done\n";
      if (thread_exc) std::rethrow_exception(thread_exc);

      uint64_t mask_out = kp.k0.r_out_share.empty() ? 0ull
                           : proto::add_mod(kp.k0.r_out_share[0], kp.k1.r_out_share[0]);
      uint64_t r_hi = kp.k0.r_hi_share + kp.k1.r_hi_share;
      for (size_t i = 0; i < N; i++) {
        uint64_t recon = out0.haty_share[i] + out1.haty_share[i];
        recon = proto::sub_mod(recon, mask_out);
        int64_t expect = signed_input ? (plain[i] >> frac_bits)
                                      : (static_cast<uint64_t>(plain[i]) >> frac_bits);
        if (recon != static_cast<uint64_t>(expect)) {
          std::ostringstream oss;
          oss << "Composite trunc/ars mismatch kind=" << static_cast<int>(kind)
              << " idx=" << i << " got=" << recon << " expect=" << expect;
          // Print some diagnostics for the first mismatch.
          if (i == 0) {
            uint64_t hatx_val = hatx[i];
            uint64_t carry = (compiled.ell > 0) ? proto::add_mod(out0.bool_share[i * compiled.ell], out1.bool_share[i * compiled.ell]) : 0ull;
            uint64_t sign = (compiled.ell > 1) ? proto::add_mod(out0.bool_share[i * compiled.ell + 1], out1.bool_share[i * compiled.ell + 1]) : 0ull;
            uint64_t mask_low = (frac_bits >= 64) ? ~uint64_t(0) : ((uint64_t(1) << frac_bits) - 1);
            uint64_t expected_carry = (frac_bits == 0) ? 0ull : (((hatx_val & mask_low) < compiled.extra_u64[1]) ? 1ull : 0ull);
            uint64_t expected_sign = (signed_input && plain[i] < 0) ? 1ull : 0ull;
            oss << " hatx=" << hatx_val << " r_in=" << compiled.r_in
                << " r_hi=" << r_hi << " carry=" << carry << " sign=" << sign
                << " mask_out=" << mask_out << " expected_carry=" << expected_carry
                << " expected_sign=" << expected_sign;
          }
          throw std::runtime_error(oss.str());
        }
      }
    };
    bool run_comp = true;  // composite trunc/ARS path smoke test
    if (run_comp) {
      try {
        run_composite(compiler::GateKind::FaithfulTR, 6, /*signed_input=*/false);
        run_composite(compiler::GateKind::FaithfulARS, 5, /*signed_input=*/true);
        run_composite(compiler::GateKind::GapARS, 4, /*signed_input=*/true);
      } catch (const std::exception& e) {
        std::cerr << "Composite truncation smoke failed: " << e.what() << "\n";
        return 1;
      }
    }
    std::cout << "Faithful truncation tests passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "test_truncation exception: " << e.what() << "\n";
    return 1;
  }
}
