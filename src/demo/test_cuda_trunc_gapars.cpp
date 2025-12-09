#include <cassert>
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

#include "gates/composite_fss.hpp"
#include "proto/backend_clear.hpp"
#include "proto/backend_gpu.hpp"
#include "gates/postproc_hooks.hpp"

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

int main() {
#ifndef SUF_HAVE_CUDA
  std::cout << "SUF_HAVE_CUDA not defined; skipping CUDA trunc tests.\n";
  return 0;
#else
  proto::ClearBackend cpu;
  auto gpu0 = proto::make_real_gpu_backend();
  auto gpu1 = proto::make_real_gpu_backend();
  if (!gpu0 || !gpu1) {
    std::cout << "GPU backend unavailable; skipping.\n";
    return 0;
  }
  const size_t N = 4;
  int frac_bits = 4;
  std::vector<int64_t> plain = {16, -8, 3, -1};

  suf::SUF<uint64_t> suf_gap, suf_faith;
  std::mt19937_64 rng_gap(1), rng_gap_cpu(1), rng_faith(2), rng_faith_cpu(2);
  auto kp_gap = gates::composite_gen_trunc_gate(*gpu0, rng_gap, frac_bits, compiler::GateKind::GapARS, N, &suf_gap);
  auto kp_gap_cpu = gates::composite_gen_trunc_gate(cpu, rng_gap_cpu, frac_bits, compiler::GateKind::GapARS, N, nullptr);
  auto kp_faith = gates::composite_gen_trunc_gate(*gpu0, rng_faith, frac_bits, compiler::GateKind::FaithfulTR, N, &suf_faith);
  auto kp_faith_cpu = gates::composite_gen_trunc_gate(cpu, rng_faith_cpu, frac_bits, compiler::GateKind::FaithfulTR, N, nullptr);

  std::vector<uint64_t> hatx_gap(N), hatx_faith(N);
  for (size_t i = 0; i < N; i++) {
    hatx_gap[i] = static_cast<uint64_t>(plain[i]) + kp_gap.k0.compiled.r_in;
    hatx_faith[i] = static_cast<uint64_t>(plain[i]) + kp_faith.k0.compiled.r_in;
  }
  gates::CompositeBatchInput in_gap{hatx_gap.data(), N, nullptr};
  gates::CompositeBatchInput in_faith{hatx_faith.data(), N, nullptr};

  ProtoLocalChan::Shared sh_gap, sh_faith;
  ProtoLocalChan c0_gap(&sh_gap, true), c1_gap(&sh_gap, false);
  ProtoLocalChan c0_faith(&sh_faith, true), c1_faith(&sh_faith, false);

  gates::FaithfulTruncPostProc hook_gap0, hook_gap1, hook_f0, hook_f1;
  hook_gap0.f = hook_gap1.f = hook_f0.f = hook_f1.f = frac_bits;

  gates::CompositeBatchOutput out_gap0, out_gap1, out_gap_cpu0, out_gap_cpu1;
  gates::CompositeBatchOutput out_f0, out_f1, out_f_cpu0, out_f_cpu1;

  std::exception_ptr thr_exc;
  std::thread t_gap([&]() {
    try {
      out_gap1 = gates::composite_eval_batch_with_postproc(1, *gpu1, c1_gap, kp_gap.k1, suf_gap, in_gap, hook_gap1);
      out_f1 = gates::composite_eval_batch_with_postproc(1, *gpu1, c1_faith, kp_faith.k1, suf_faith, in_faith, hook_f1);
    } catch (...) { thr_exc = std::current_exception(); }
  });
  out_gap0 = gates::composite_eval_batch_with_postproc(0, *gpu0, c0_gap, kp_gap.k0, suf_gap, in_gap, hook_gap0);
  out_f0 = gates::composite_eval_batch_with_postproc(0, *gpu0, c0_faith, kp_faith.k0, suf_faith, in_faith, hook_f0);
  t_gap.join();
  if (thr_exc) std::rethrow_exception(thr_exc);

  // CPU baselines
  out_gap_cpu0 = gates::composite_eval_batch_with_postproc(0, cpu, c0_gap, kp_gap_cpu.k0, suf_gap, in_gap, hook_gap0);
  out_gap_cpu1 = gates::composite_eval_batch_with_postproc(1, cpu, c1_gap, kp_gap_cpu.k1, suf_gap, in_gap, hook_gap1);
  out_f_cpu0 = gates::composite_eval_batch_with_postproc(0, cpu, c0_faith, kp_faith_cpu.k0, suf_faith, in_faith, hook_f0);
  out_f_cpu1 = gates::composite_eval_batch_with_postproc(1, cpu, c1_faith, kp_faith_cpu.k1, suf_faith, in_faith, hook_f1);

  auto check = [&](const gates::CompositeBatchOutput& a0,
                   const gates::CompositeBatchOutput& a1,
                   const gates::CompositeBatchOutput& b0,
                   const gates::CompositeBatchOutput& b1,
                   const char* label) {
    for (size_t i = 0; i < N; i++) {
      uint64_t rec_a = proto::sub_mod(a0.haty_share[i] + a1.haty_share[i],
                                      proto::add_mod(kp_gap.k0.r_out_share[0], kp_gap.k1.r_out_share[0]));
      uint64_t rec_b = proto::sub_mod(b0.haty_share[i] + b1.haty_share[i],
                                      proto::add_mod(kp_gap_cpu.k0.r_out_share[0], kp_gap_cpu.k1.r_out_share[0]));
      if (rec_a != rec_b) {
        std::cerr << label << " mismatch idx=" << i << " gpu=" << rec_a << " cpu=" << rec_b << "\n";
        return false;
      }
    }
    return true;
  };

  bool ok_gap = check(out_gap0, out_gap1, out_gap_cpu0, out_gap_cpu1, "GapARS");
  bool ok_f = check(out_f0, out_f1, out_f_cpu0, out_f_cpu1, "FaithfulTR");
  if (!ok_gap || !ok_f) return 1;
  std::cout << "CUDA GapARS/Faithful truncation tests passed.\n";
  return 0;
#endif
}
