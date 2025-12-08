#include <cassert>
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

#include "gates/composite_fss.hpp"
#include "proto/backend_clear.hpp"
#include "proto/backend_gpu.hpp"
#include "proto/channel.hpp"
#include "gates/postproc_hooks.hpp"
#include "compiler/pfss_program_desc.hpp"

// Minimal in-process channel for tests (copied from test_truncation).
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
  std::cout << "SUF_HAVE_CUDA not defined; skipping GPU backend test.\n";
  return 0;
#else
  proto::ClearBackend cpu;
  auto gpu = proto::make_real_gpu_backend();

  // Simple DCF equivalence: alpha=5, payload=0xAA.
  int in_bits = 6;
  auto alpha_bits = cpu.u64_to_bits_msb(5, in_bits);
  std::vector<proto::u8> payload(1, 0xAA);
  auto kp_dcf_cpu = cpu.gen_dcf(in_bits, alpha_bits, payload);
  auto kp_dcf_gpu = gpu->gen_dcf(in_bits, alpha_bits, payload);
  auto to_bits = [&](uint64_t v) { return cpu.u64_to_bits_msb(v, in_bits); };
  for (uint64_t x : {3ull, 5ull, 7ull}) {
    auto out_cpu = cpu.eval_dcf(in_bits, kp_dcf_cpu.k0, to_bits(x));
    auto out_gpu = gpu->eval_dcf(in_bits, kp_dcf_gpu.k0, to_bits(x));
    bool lt = x < 5ull;
    proto::u8 expect = lt ? 0xAA : 0;
    if (out_cpu.size() != 1 || out_gpu.size() != 1 || out_cpu[0] != expect || out_gpu[0] != expect) {
      std::cerr << "DCF mismatch x=" << x << " expect=" << static_cast<int>(expect)
                << " cpu=" << (out_cpu.empty() ? -1 : static_cast<int>(out_cpu[0]))
                << " gpu=" << (out_gpu.empty() ? -1 : static_cast<int>(out_gpu[0])) << "\n";
      return 1;
    }
  }

  // Interval LUT equivalence (party0 share) if backend supports it.
  if (auto* gpu_lut = dynamic_cast<proto::PfssIntervalLutExt*>(gpu.get())) {
    auto* cpu_lut = dynamic_cast<proto::PfssIntervalLutExt*>(&cpu);
    proto::IntervalLutDesc il;
    il.in_bits = 6;
    il.out_words = 1;
    il.cutpoints = {0, 32, 64};
    il.payload_flat = {10, 20};
    auto il_gpu_keys = gpu_lut->gen_interval_lut(il);
    auto il_cpu_keys = cpu_lut->gen_interval_lut(il);
    std::vector<uint64_t> xs_lut = {0, 15, 40};
    size_t key_bytes = il_gpu_keys.k0.bytes.size();
    std::vector<uint8_t> keys_flat(xs_lut.size() * key_bytes);
    for (size_t i = 0; i < xs_lut.size(); i++) {
      std::memcpy(keys_flat.data() + i * key_bytes, il_gpu_keys.k0.bytes.data(), key_bytes);
    }
    std::vector<uint64_t> out_gpu(xs_lut.size() * static_cast<size_t>(il.out_words), 0),
        out_cpu(xs_lut.size() * static_cast<size_t>(il.out_words), 0);
    gpu_lut->eval_interval_lut_many_u64(key_bytes, keys_flat.data(), xs_lut, il.out_words, out_gpu.data());
    // Build CPU key stream separately.
    size_t key_bytes_cpu = il_cpu_keys.k0.bytes.size();
    std::vector<uint8_t> keys_flat_cpu(xs_lut.size() * key_bytes_cpu);
    for (size_t i = 0; i < xs_lut.size(); i++) {
      std::memcpy(keys_flat_cpu.data() + i * key_bytes_cpu, il_cpu_keys.k0.bytes.data(), key_bytes_cpu);
    }
    cpu_lut->eval_interval_lut_many_u64(key_bytes_cpu, keys_flat_cpu.data(), xs_lut, il.out_words, out_cpu.data());
    if (out_gpu != out_cpu) {
      std::cerr << "Interval LUT mismatch\n";
      return 1;
    }
  }

  // Composite truncation equivalence on a tiny batch.
  const size_t N = 4;
  int frac_bits = 4;
  std::mt19937_64 rng_cpu(12345);
  std::mt19937_64 rng_gpu(12345);
  std::vector<int64_t> plain = {16, -8, 3, -1};

  suf::SUF<uint64_t> trunc_suf_cpu, trunc_suf_gpu;
  auto trunc_cpu = gates::composite_gen_trunc_gate(cpu, rng_cpu, frac_bits, compiler::GateKind::FaithfulTR, N, &trunc_suf_cpu);
  auto trunc_gpu = gates::composite_gen_trunc_gate(*gpu, rng_gpu, frac_bits, compiler::GateKind::FaithfulTR, N, &trunc_suf_gpu);

  std::vector<uint64_t> hatx_cpu(N), hatx_gpu(N);
  for (size_t i = 0; i < N; i++) {
    hatx_cpu[i] = static_cast<uint64_t>(plain[i]) + trunc_cpu.k0.compiled.r_in;
    hatx_gpu[i] = static_cast<uint64_t>(plain[i]) + trunc_gpu.k0.compiled.r_in;
  }
  gates::CompositeBatchInput in_cpu{hatx_cpu.data(), N};
  gates::CompositeBatchInput in_gpu{hatx_gpu.data(), N};

  gates::FaithfulTruncPostProc hook_cpu0, hook_cpu1, hook_gpu0, hook_gpu1;
  hook_cpu0.f = hook_cpu1.f = hook_gpu0.f = hook_gpu1.f = frac_bits;
  ProtoLocalChan::Shared sh_cpu, sh_gpu;
  ProtoLocalChan c0_cpu(&sh_cpu, true), c1_cpu(&sh_cpu, false);
  ProtoLocalChan c0_gpu(&sh_gpu, true), c1_gpu(&sh_gpu, false);

  gates::CompositeBatchOutput out_cpu0, out_cpu1, out_gpu0, out_gpu1;
  std::exception_ptr thr_exc;
  std::thread t_cpu([&]() {
    try {
      out_cpu1 = gates::composite_eval_batch_with_postproc(1, cpu, c1_cpu, trunc_cpu.k1, trunc_suf_cpu, in_cpu, hook_cpu1);
    } catch (...) { thr_exc = std::current_exception(); }
  });
  out_cpu0 = gates::composite_eval_batch_with_postproc(0, cpu, c0_cpu, trunc_cpu.k0, trunc_suf_cpu, in_cpu, hook_cpu0);
  t_cpu.join();
  if (thr_exc) std::rethrow_exception(thr_exc);

  std::thread t_gpu([&]() {
    try {
      out_gpu1 = gates::composite_eval_batch_with_postproc(1, *gpu, c1_gpu, trunc_gpu.k1, trunc_suf_gpu, in_gpu, hook_gpu1);
    } catch (...) { thr_exc = std::current_exception(); }
  });
  out_gpu0 = gates::composite_eval_batch_with_postproc(0, *gpu, c0_gpu, trunc_gpu.k0, trunc_suf_gpu, in_gpu, hook_gpu0);
  t_gpu.join();
  if (thr_exc) std::rethrow_exception(thr_exc);

  auto mask_out_cpu = trunc_cpu.k0.r_out_share.empty() ? 0ull
                         : proto::add_mod(trunc_cpu.k0.r_out_share[0], trunc_cpu.k1.r_out_share[0]);
  auto mask_out_gpu = trunc_gpu.k0.r_out_share.empty() ? 0ull
                         : proto::add_mod(trunc_gpu.k0.r_out_share[0], trunc_gpu.k1.r_out_share[0]);

  for (size_t i = 0; i < N; i++) {
    uint64_t recon_cpu = out_cpu0.haty_share[i] + out_cpu1.haty_share[i];
    recon_cpu = proto::sub_mod(recon_cpu, mask_out_cpu);
    uint64_t recon_gpu = out_gpu0.haty_share[i] + out_gpu1.haty_share[i];
    recon_gpu = proto::sub_mod(recon_gpu, mask_out_gpu);
    if (recon_cpu != recon_gpu) {
      std::cerr << "Composite mismatch idx=" << i << " cpu=" << recon_cpu
                << " gpu=" << recon_gpu << "\n";
      return 1;
    }
    // Sanity check on one positive input.
    if (plain[i] >= 0) {
      int64_t expect = plain[i] >> frac_bits;
      if (recon_cpu != static_cast<uint64_t>(expect)) {
        std::cerr << "Composite mismatch vs expect idx=" << i << " expect=" << expect
                  << " got=" << recon_cpu << "\n";
        return 1;
      }
    }
  }

  std::cout << "GPU PFSS equivalence tests passed (using current CUDA backend).\n";
  return 0;
#endif
}
