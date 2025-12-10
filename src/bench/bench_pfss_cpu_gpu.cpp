#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <thread>

#include "gates/composite_fss.hpp"
#include <cuda_runtime.h>
#include "proto/backend_clear.hpp"
#include "proto/backend_gpu.hpp"
#include "gates/postproc_hooks.hpp"

namespace {

// Minimal in-process proto channel for two parties.
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

struct BenchResult {
  double avg_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
};

BenchResult bench_once(proto::PfssBackendBatch& be0,
                       proto::PfssBackendBatch& be1,
                       const gates::CompositeKeyPair& kp,
                       const suf::SUF<uint64_t>& suf_gap,
                       const std::vector<uint64_t>& hatx,
                       int frac_bits,
                       int reps) {
  gates::FaithfulTruncPostProc hook;
  hook.f = frac_bits;
  auto run = [&]() {
    ProtoLocalChan::Shared sh;
    ProtoLocalChan c0(&sh, true), c1(&sh, false);
    gates::CompositeBatchInput in{hatx.data(), hatx.size(), nullptr};
    gates::CompositeBatchOutput o0, o1;
    std::thread t([&]() {
      o1 = gates::composite_eval_batch_with_postproc(
          1, be1, c1, kp.k1, suf_gap, in, hook);
    });
    o0 = gates::composite_eval_batch_with_postproc(
        0, be0, c0, kp.k0, suf_gap, in, hook);
    t.join();
    return std::make_pair(o0, o1);
  };

  double min_ms = 1e9, max_ms = 0.0, sum_ms = 0.0;
  for (int i = 0; i < reps; i++) {
    auto start = std::chrono::steady_clock::now();
    run();
    auto end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    min_ms = std::min(min_ms, ms);
    max_ms = std::max(max_ms, ms);
    sum_ms += ms;
  }
  BenchResult br;
  br.avg_ms = sum_ms / reps;
  br.min_ms = min_ms;
  br.max_ms = max_ms;
  return br;
}

bool check_outputs(const gates::CompositeBatchOutput& o0,
                   const gates::CompositeBatchOutput& o1,
                   const gates::CompositeKeyPair& kp,
                   const std::vector<uint64_t>& expected) {
  for (size_t i = 0; i < expected.size(); i++) {
    uint64_t rec = proto::sub_mod(o0.haty_share[i] + o1.haty_share[i],
                                  proto::add_mod(kp.k0.r_out_share[0], kp.k1.r_out_share[0]));
    if (rec != expected[i]) {
      std::cerr << "mismatch idx=" << i << " got=" << rec << " expect=" << expected[i] << "\n";
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
#ifndef SUF_HAVE_CUDA
  std::cout << "SUF_HAVE_CUDA not defined; skipping GPU/CPU PFSS bench.\n";
  return 0;
#else
  std::cout << std::unitbuf;
  const int frac_bits = 8;
  const int reps = 10;
  std::vector<size_t> sizes = {256, 1024, 4096, 8192};

  proto::ClearBackend cpu;
  auto gpu0 = proto::make_real_gpu_backend();
  auto gpu1 = proto::make_real_gpu_backend();
  if (!gpu0 || !gpu1) {
    std::cout << "GPU backend unavailable; CPU only.\n";
  }

  for (size_t N : sizes) {
    std::cout << "\n[PFSS bench] N=" << N << " reps=" << reps << " frac_bits=" << frac_bits << "\n";
    std::vector<uint64_t> plain(N);
    std::mt19937_64 rng(42 + static_cast<uint64_t>(N));
    for (size_t i = 0; i < N; i++) {
      // Positive inputs to simplify expected truncation.
      uint64_t v = static_cast<uint64_t>(rng() % 65536);
      plain[i] = v;
    }

    std::cout << "[PFSS bench] keygen...\n";
    suf::SUF<uint64_t> suf_cpu, suf_gpu;
    std::mt19937_64 rng_k0(123 + static_cast<uint64_t>(N));
    auto kp_cpu = gates::composite_gen_trunc_gate(cpu, rng_k0, frac_bits,
                                                  compiler::GateKind::GapARS, static_cast<int>(N),
                                                  &suf_cpu);
    std::vector<uint64_t> hatx_cpu(N);
    for (size_t i = 0; i < N; i++) {
      hatx_cpu[i] = proto::add_mod(plain[i], kp_cpu.k0.compiled.r_in);
    }
    std::unique_ptr<gates::CompositeKeyPair> kp_gpu;
    std::vector<uint64_t> hatx_gpu;
    if (gpu0 && gpu1) {
      std::mt19937_64 rng_k1(321 + static_cast<uint64_t>(N));
      kp_gpu = std::make_unique<gates::CompositeKeyPair>(
          gates::composite_gen_trunc_gate(*gpu0, rng_k1, frac_bits,
                                          compiler::GateKind::GapARS, static_cast<int>(N),
                                          &suf_gpu));
      hatx_gpu.resize(N);
      for (size_t i = 0; i < N; i++) {
        hatx_gpu[i] = proto::add_mod(plain[i], kp_gpu->k0.compiled.r_in);
      }
    }

    // Correctness sanity: CPU baseline once and compare GPU once if present.
    std::cout << "[PFSS bench] CPU correctness run...\n";
    ProtoLocalChan::Shared sh_chk;
    ProtoLocalChan c0_chk(&sh_chk, true), c1_chk(&sh_chk, false);
    gates::FaithfulTruncPostProc hook_chk;
    hook_chk.f = frac_bits;
    hook_chk.r_in = kp_cpu.k0.compiled.r_in;
    hook_chk.r_hi_share = kp_cpu.k0.r_hi_share;
    gates::CompositeBatchInput in{hatx_cpu.data(), N, nullptr};
    std::exception_ptr cpu_exc;
    gates::CompositeBatchOutput out_cpu1;
    std::thread cpu_p1([&]() {
      try {
        out_cpu1 = gates::composite_eval_batch_with_postproc(1, cpu, c1_chk, kp_cpu.k1, suf_cpu, in, hook_chk);
      } catch (...) { cpu_exc = std::current_exception(); }
    });
    auto out_cpu0 = gates::composite_eval_batch_with_postproc(0, cpu, c0_chk, kp_cpu.k0, suf_cpu, in, hook_chk);
    cpu_p1.join();
    if (cpu_exc) std::rethrow_exception(cpu_exc);
    std::vector<uint64_t> expected(N);
    for (size_t i = 0; i < N; i++) {
      expected[i] = plain[i] >> frac_bits;
    }
    if (!check_outputs(out_cpu0, out_cpu1, kp_cpu, expected)) {
      std::cerr << "CPU self-check failed.\n";
      return 1;
    }

    if (kp_gpu) {
      std::cout << "[PFSS bench] GPU correctness run...\n";
      ProtoLocalChan::Shared sh_gpu;
      ProtoLocalChan c0g(&sh_gpu, true), c1g(&sh_gpu, false);
    gates::CompositeBatchInput in_gpu{hatx_gpu.data(), N, nullptr};
    std::exception_ptr gpu_exc;
    gates::CompositeBatchOutput out_gpu1;
    cudaEvent_t ev_start{}, ev_end{};
    cudaStream_t s = nullptr;
    if (auto* staged = dynamic_cast<proto::PfssGpuStagedEval*>(gpu0.get())) {
      s = reinterpret_cast<cudaStream_t>(staged->device_stream());
      if (s) {
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_end);
        cudaEventRecord(ev_start, s);
      }
    }
    std::thread gpu_p1([&]() {
      try {
        out_gpu1 = gates::composite_eval_batch_with_postproc(1, *gpu1, c1g, kp_gpu->k1, suf_gpu, in_gpu, hook_chk);
      } catch (...) { gpu_exc = std::current_exception(); }
    });
    auto out_gpu0 = gates::composite_eval_batch_with_postproc(0, *gpu0, c0g, kp_gpu->k0, suf_gpu, in_gpu, hook_chk);
    if (s && ev_end) cudaEventRecord(ev_end, s);
    gpu_p1.join();
    if (gpu_exc) std::rethrow_exception(gpu_exc);
    if (s && ev_start && ev_end) {
      cudaEventSynchronize(ev_end);
      float ms = 0.f;
      cudaEventElapsedTime(&ms, ev_start, ev_end);
      std::cout << "[PFSS bench] GPU device-time (events) ~" << ms << " ms (N=" << N << ")\n";
      cudaEventDestroy(ev_start);
      cudaEventDestroy(ev_end);
    }
    if (!check_outputs(out_gpu0, out_gpu1, *kp_gpu, expected)) {
      std::cerr << "GPU vs expectation failed.\n";
      return 1;
    }
  }

    auto cpu_br = bench_once(cpu, cpu, kp_cpu, suf_cpu, hatx_cpu, frac_bits, reps);
    std::cout << "[PFSS trunc GapARS] CPU: avg=" << cpu_br.avg_ms << "ms"
              << " min=" << cpu_br.min_ms << "ms max=" << cpu_br.max_ms << "ms over "
              << reps << " reps (N=" << N << ")\n";

    if (kp_gpu) {
      auto gpu_br = bench_once(*gpu0, *gpu1, *kp_gpu, suf_gpu, hatx_gpu, frac_bits, reps);
      std::cout << "[PFSS trunc GapARS] GPU: avg=" << gpu_br.avg_ms << "ms"
                << " min=" << gpu_br.min_ms << "ms max=" << gpu_br.max_ms << "ms over "
                << reps << " reps (N=" << N << ")\n";
    }
  }
  return 0;
#endif
}
