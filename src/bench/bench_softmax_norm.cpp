#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "compiler/truncation_lowering.hpp"
#include "gates/nexp_composite.hpp"
#include "gates/reciprocal_composite.hpp"
#include "gates/reciprocal_gate.hpp"
#include "gates/rsqrt_gate.hpp"
#include "gates/tables/rsqrt_piecewise_affine_init.hpp"
#include "nn/softmax_block_task.hpp"
#include "proto/backend_gpu.hpp"
#include "proto/reference_backend.hpp"
#include "proto/channel.hpp"
#include "runtime/phase_executor.hpp"
#include "runtime/phase_tasks.hpp"
#include "runtime/pfss_phase_planner.hpp"
#include "runtime/pfss_gpu_staging.hpp"
#include "runtime/pfss_superbatch.hpp"

namespace {

struct LocalChan : net::Chan {
  struct Shared {
    std::mutex m;
    std::condition_variable cv;
    std::queue<uint64_t> q0to1, q1to0;
  };
  Shared* sh = nullptr;
  bool is0 = false;
  LocalChan(Shared* s, bool p) : sh(s), is0(p) {}
  void send_u64(uint64_t v) override {
    std::unique_lock<std::mutex> lk(sh->m);
    auto& q = is0 ? sh->q0to1 : sh->q1to0;
    q.push(v);
    sh->cv.notify_all();
  }
  uint64_t recv_u64() override {
    std::unique_lock<std::mutex> lk(sh->m);
    auto& q = is0 ? sh->q1to0 : sh->q0to1;
    sh->cv.wait(lk, [&] { return !q.empty(); });
    uint64_t v = q.front();
    q.pop();
    return v;
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

struct RowBroadcastTripleMaterial {
  int rows = 0;
  int cols = 0;
  std::vector<uint64_t> A0, A1;
  std::vector<uint64_t> B0, B1;
  std::vector<uint64_t> C0, C1;
};

RowBroadcastTripleMaterial make_row_broadcast_triples(int rows, int cols, std::mt19937_64& rng) {
  RowBroadcastTripleMaterial mat;
  mat.rows = rows;
  mat.cols = cols;
  size_t count = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  mat.A0.resize(count);
  mat.A1.resize(count);
  mat.B0.resize(static_cast<size_t>(rows));
  mat.B1.resize(static_cast<size_t>(rows));
  mat.C0.resize(count);
  mat.C1.resize(count);

  std::vector<uint64_t> B(rows);
  for (int r = 0; r < rows; ++r) {
    uint64_t b = rng();
    uint64_t b0 = rng();
    uint64_t b1 = b - b0;
    B[static_cast<size_t>(r)] = b;
    mat.B0[static_cast<size_t>(r)] = b0;
    mat.B1[static_cast<size_t>(r)] = b1;
  }
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      size_t idx = static_cast<size_t>(r * cols + c);
      uint64_t a = rng();
      uint64_t a0 = rng();
      uint64_t a1 = a - a0;
      uint64_t c_val = proto::mul_mod(a, B[static_cast<size_t>(r)]);
      uint64_t c0 = rng();
      uint64_t c1 = c_val - c0;
      mat.A0[idx] = a0;
      mat.A1[idx] = a1;
      mat.C0[idx] = c0;
      mat.C1[idx] = c1;
    }
  }
  return mat;
}

void ensure_mul_triples(std::vector<proto::BeaverTriple64Share>& t0,
                        std::vector<proto::BeaverTriple64Share>& t1,
                        size_t need,
                        std::mt19937_64& rng) {
  while (t0.size() < need || t1.size() < need) {
    uint64_t a = rng(), b = rng(), c = proto::mul_mod(a, b);
    uint64_t a0 = rng();
    uint64_t a1 = a - a0;
    uint64_t b0 = rng();
    uint64_t b1 = b - b0;
    uint64_t c0 = rng();
    uint64_t c1 = c - c0;
    t0.push_back({a0, b0, c0});
    t1.push_back({a1, b1, c1});
  }
}

class RowBroadcastTripleProviderImpl : public runtime::RowBroadcastTripleProvider {
 public:
  RowBroadcastTripleProviderImpl(const RowBroadcastTripleMaterial& mat, int party)
      : mat_(mat), party_(party) {}

  runtime::RowBroadcastTriple reserve_mul(int rows, int cols) override {
    if (rows != mat_.rows || cols != mat_.cols) {
      throw std::runtime_error("RowBroadcastTripleProviderImpl: shape mismatch");
    }
    size_t count = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    const auto& A = (party_ == 0) ? mat_.A0 : mat_.A1;
    const auto& B = (party_ == 0) ? mat_.B0 : mat_.B1;
    const auto& C = (party_ == 0) ? mat_.C0 : mat_.C1;
    return {std::span<const uint64_t>(A.data(), count),
            std::span<const uint64_t>(B.data(), static_cast<size_t>(rows)),
            std::span<const uint64_t>(C.data(), count)};
  }

 private:
  const RowBroadcastTripleMaterial& mat_;
  int party_ = 0;
};

struct SoftmaxResult {
  std::vector<uint64_t> probs;
};

SoftmaxResult run_softmax_party(int party,
                                int rows,
                                int cols,
                                int fb,
                                const std::vector<int>& valid,
                                const runtime::CubicPolyBundle& nexp,
                                const runtime::RecipTaskBundle& recip,
                                const runtime::TruncChoice& trunc_choice,
                                runtime::RowBroadcastTripleProvider* rb,
                                proto::PfssBackendBatch& backend,
                                LocalChan& net_ch,
                                std::span<const uint64_t> t_qf,
                                bool device_only = false) {
  if (std::getenv("SOFTMAX_BENCH_TRACE")) {
    std::fprintf(stderr, "[bench] party %d run_softmax_party rows=%d cols=%d device_only=%d\n",
                 party, rows, cols, device_only ? 1 : 0);
  }
  runtime::PhaseExecutor pe;
  if (device_only) {
    pe.set_device_pipeline(true);
    pe.set_device_pipeline_materialize(false);
  }
  runtime::PhaseResources R{};
  R.party = party;
  R.pfss_backend = &backend;
  runtime::ProtoChanFromNet pch(net_ch);
  R.pfss_chan = &pch;
  R.net_chan = &net_ch;
  R.pfss_coeff = &pe.pfss_coeff_batch();
  R.pfss_trunc = &pe.pfss_trunc_batch();
  R.opens = &pe.open_collector();
  if (device_only) {
    pe.pfss_coeff_batch().set_device_outputs(true);
    pe.pfss_trunc_batch().set_device_outputs(true);
  }

#ifdef SUF_HAVE_CUDA
  std::unique_ptr<runtime::CudaPfssStager> cuda_stager;
  if (device_only) {
    if (auto* staged = dynamic_cast<proto::PfssGpuStagedEval*>(&backend)) {
      cuda_stager = std::make_unique<runtime::CudaPfssStager>(staged->device_stream());
      pe.pfss_coeff_batch().set_gpu_stager(cuda_stager.get());
      pe.pfss_trunc_batch().set_gpu_stager(cuda_stager.get());
    }
  }
#endif

  nn::SoftmaxPlan plan;
  plan.frac_bits = fb;
  plan.rows = rows;
  plan.cols = cols;
  plan.valid_lens = valid;
  plan.device_only = device_only;
  plan.materialize_host = !device_only;
  plan.nexp = nexp;
  plan.recip = recip;
  plan.prob_trunc = trunc_choice;
  plan.row_triples = rb;
  compiler::RangeInterval pr;
  pr.is_signed = true;
  pr.lo = 0;
  pr.hi = static_cast<int64_t>(1ll << fb);
  plan.prob_range = pr;

  std::vector<uint64_t> out(static_cast<size_t>(rows * cols), 0);
  auto task = std::make_unique<nn::SoftmaxBlockTask>(
      plan,
      t_qf,
      std::span<uint64_t>(out.data(), out.size()));
  pe.begin_phase(runtime::PhaseExecutor::Phase::kSoftmax);
  pe.add_task(std::move(task));
  pe.run(R);
  if (std::getenv("SOFTMAX_BENCH_TRACE")) {
    std::fprintf(stderr, "[bench] party %d softmax finished\n", party);
  }

  SoftmaxResult res;
  res.probs = std::move(out);
  return res;
}

// Build a SUF that emits affine rsqrt init coefficients adjusted for fixed-point.
inline suf::SUF<uint64_t> build_rsqrt_affine_eval_suf(const gates::PiecewisePolySpec& spec) {
  suf::SUF<uint64_t> F;
  F.n_bits = 64;
  F.r_out = 2;
  F.l_out = 0;
  F.degree = 0;

  std::vector<gates::PiecewiseInterval> intervals = spec.intervals;
  std::sort(intervals.begin(), intervals.end(),
            [](const gates::PiecewiseInterval& a, const gates::PiecewiseInterval& b) {
              return a.start < b.start;
            });
  if (intervals.empty()) return F;
  F.alpha.clear();
  F.alpha.reserve(intervals.size() + 1);
  F.alpha.push_back(intervals.front().start);
  if (F.alpha.front() != 0) {
    suf::SufPiece<uint64_t> zero_piece;
    zero_piece.polys.resize(2);
    zero_piece.polys[0].coeffs = {0};
    zero_piece.polys[1].coeffs = {0};
    F.pieces.push_back(std::move(zero_piece));
    F.alpha.insert(F.alpha.begin(), 0ull);
  }
  auto clamp_to_ring = [](__int128 v) -> uint64_t {
    if (v > static_cast<__int128>(std::numeric_limits<int64_t>::max())) {
      v = static_cast<__int128>(std::numeric_limits<int64_t>::max());
    }
    if (v < static_cast<__int128>(std::numeric_limits<int64_t>::min())) {
      v = static_cast<__int128>(std::numeric_limits<int64_t>::min());
    }
    return static_cast<uint64_t>(static_cast<int64_t>(v));
  };
  for (const auto& iv : intervals) {
    int64_t a0 = (!iv.pack.coeffs.empty()) ? iv.pack.coeffs[0] : 0;
    int64_t a1 = (iv.pack.coeffs.size() > 1) ? iv.pack.coeffs[1] : 0;
    int64_t offset = iv.pack.offset;
    __int128 prod = static_cast<__int128>(a1) * static_cast<__int128>(offset);
    int64_t offset_term = static_cast<int64_t>(prod >> spec.frac_bits_in);
    int64_t adj = a0 - offset_term;

    suf::SufPiece<uint64_t> piece;
    piece.polys.resize(2);
    piece.polys[0].coeffs = {clamp_to_ring(adj)};
    piece.polys[1].coeffs = {clamp_to_ring(a1)};
    F.pieces.push_back(std::move(piece));
    F.alpha.push_back(iv.end);
  }
  return F;
}

struct RsqrtMaterial {
  suf::SUF<uint64_t> suf;
  gates::CompositeKeyPair keys;
  compiler::TruncationLoweringResult trunc_f;
  compiler::TruncationLoweringResult trunc_2f;
  gates::PiecewisePolySpec init_spec;
  int frac_bits = 0;
  int nr_iters = 1;
  runtime::RsqrtTaskBundle bundle() {
    runtime::RsqrtTaskBundle b{};
    b.suf = &suf;
    b.key0 = &keys.k0;
    b.key1 = &keys.k1;
    b.trunc_f = &trunc_f;
    b.trunc_2f = &trunc_2f;
    b.init_spec = &init_spec;
    b.frac_bits = frac_bits;
    b.nr_iters = nr_iters;
    return b;
  }
};

// Simple layernorm helper: uses RsqrtTask + LayerNormTask.
struct LayerNormBundle {
  runtime::LayerNormTaskBundle bundle;
  std::vector<proto::BeaverTriple64Share> mul_triples0;
  std::vector<proto::BeaverTriple64Share> mul_triples1;
  RowBroadcastTripleMaterial rb_mat;
  RsqrtMaterial rsqrt_mat;
  compiler::TruncationLoweringResult mean_trunc_res;
  compiler::TruncationLoweringResult var_trunc_res;
  compiler::TruncationLoweringResult norm_trunc_res;
};

RsqrtMaterial make_rsqrt_material(int frac_bits,
                                  int nr_iters,
                                  double eps,
                                  double vmax,
                                  proto::PfssBackendBatch& backend,
                                  std::mt19937_64& rng,
                                  int rows) {
  auto spec = gates::make_rsqrt_affine_init_spec(frac_bits, eps, vmax);
  auto suf_gate = build_rsqrt_affine_eval_suf(spec);
  std::vector<uint64_t> r_out(static_cast<size_t>(suf_gate.r_out), 0ull);
  auto kp = gates::composite_gen_backend_with_masks(
      suf_gate, backend, rng, rng(), r_out, static_cast<size_t>(rows), compiler::GateKind::Rsqrt);
  kp.k0.compiled.gate_kind = compiler::GateKind::Rsqrt;
  kp.k1.compiled.gate_kind = compiler::GateKind::Rsqrt;
  if (std::getenv("SOFTMAX_BENCH_TRACE")) {
    std::fprintf(stderr, "[bench rsqrt material] rows=%d r_in_vec0=%zu r_in_vec1=%zu r_in0=%llu r_in1=%llu\n",
                 rows,
                 kp.k0.r_in_share_vec.size(),
                 kp.k1.r_in_share_vec.size(),
                 static_cast<unsigned long long>(kp.k0.r_in_share),
                 static_cast<unsigned long long>(kp.k1.r_in_share));
  }

  compiler::GateParams p;
  p.kind = compiler::GateKind::FaithfulARS;
  p.frac_bits = frac_bits;
  p.per_element_masks = true;
  auto trunc_f = compiler::lower_truncation_gate(backend, rng, p, static_cast<size_t>(rows));
  p.frac_bits = 2 * frac_bits;
  p.kind = compiler::GateKind::GapARS;
  auto trunc_2f = compiler::lower_truncation_gate(backend, rng, p, static_cast<size_t>(rows));

  RsqrtMaterial mat;
  mat.suf = std::move(suf_gate);
  mat.keys = std::move(kp);
  mat.trunc_f = std::move(trunc_f);
  mat.trunc_2f = std::move(trunc_2f);
  mat.init_spec = spec;
  mat.frac_bits = frac_bits;
  mat.nr_iters = nr_iters;
  return mat;
}

LayerNormBundle build_layernorm_bundle(proto::PfssBackendBatch& backend,
                                       int rows,
                                       int cols,
                                       int fb) {
  LayerNormBundle out;
  // Mean/var/norm truncations (unsigned GapARS for variance, faithful for mean/norm).
  compiler::GateParams mean_p;
  mean_p.kind = compiler::GateKind::FaithfulTR;
  mean_p.frac_bits = fb;
  compiler::GateParams var_p = mean_p;
  var_p.kind = compiler::GateKind::GapARS;
  var_p.frac_bits = 2 * fb;
  compiler::GateParams norm_p = mean_p;

  std::mt19937_64 rng(1234);
  out.mean_trunc_res = compiler::lower_truncation_gate(backend, rng, mean_p, rows);
  out.var_trunc_res = compiler::lower_truncation_gate(backend, rng, var_p, rows);
  out.norm_trunc_res = compiler::lower_truncation_gate(backend, rng, norm_p, rows * cols);

  out.rsqrt_mat = make_rsqrt_material(fb, /*nr_iters=*/1, 1.0 / 1024.0, 16.0, backend, rng, rows);
  out.bundle.mean_trunc = {nullptr, &out.mean_trunc_res, fb, true};
  out.bundle.var_trunc = {nullptr, &out.var_trunc_res, 2 * fb, true};
  out.bundle.norm_trunc = {nullptr, &out.norm_trunc_res, fb, true};
  out.bundle.rsqrt = out.rsqrt_mat.bundle();
  out.bundle.inv_len_qf = static_cast<uint64_t>(std::llround((1.0 / cols) * std::ldexp(1.0, fb)));
  out.bundle.eps_qf = static_cast<uint64_t>(std::llround((1.0 / 1024.0) * std::ldexp(1.0, fb)));
  out.bundle.frac_bits = fb;
  out.mul_triples0 = out.rsqrt_mat.keys.k0.triples;
  out.mul_triples1 = out.rsqrt_mat.keys.k1.triples;
  // LayerNorm uses mul_triples for diff^2; ensure we have enough for rows*cols.
  size_t triple_need = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  ensure_mul_triples(out.mul_triples0, out.mul_triples1, triple_need, rng);
  out.bundle.mul_triples = std::span<const proto::BeaverTriple64Share>(out.mul_triples0);
  out.rb_mat = make_row_broadcast_triples(rows, cols, rng);
  return out;
}

struct BenchResult {
  double cpu_ms = 0.0;
  double gpu_ms = 0.0;
  double gpu_dev_ms = 0.0;
};

BenchResult bench_softmax(int rows, int cols, int fb, int reps, bool device_only) {
  std::mt19937_64 rng(7);
  std::vector<uint64_t> t_plain(static_cast<size_t>(rows * cols));
  for (auto& v : t_plain) v = static_cast<uint64_t>(rng() % (1 << fb));
  std::vector<uint64_t> t0 = t_plain, t1(t_plain.size(), 0ull);
  std::vector<int> valid(rows, cols);

  proto::ReferenceBackend ref_be;

  gates::NExpGateParams nexp_params{fb, 16};
  std::mt19937_64 rng_keys(99);
  auto nexp_mat_cpu = gates::dealer_make_nexp_task_material(ref_be, nexp_params, rng_keys, cols * rows, cols * rows);
  auto recip_mat_cpu = gates::dealer_make_recip_task_material(ref_be, fb, /*nr_iters=*/1, rng_keys, cols * rows);
  RowBroadcastTripleMaterial rb_mat = make_row_broadcast_triples(rows, cols, rng);
  RowBroadcastTripleProviderImpl rb0(rb_mat, 0), rb1(rb_mat, 1);

  auto run_pair = [&](proto::PfssBackendBatch& be0,
                      proto::PfssBackendBatch& be1,
                      const gates::NexpTaskMaterial& nexp_mat,
                      const gates::RecipTaskMaterial& recip_mat,
                      bool device_timing) -> std::pair<double, double> {
    if (std::getenv("SOFTMAX_BENCH_TRACE")) {
      std::fprintf(stderr, "[bench] run_pair start device_timing=%d device_only=%d\n",
                   device_timing ? 1 : 0, device_only ? 1 : 0);
    }
    LocalChan::Shared sh;
    LocalChan c0(&sh, true), c1(&sh, false);
    auto start = std::chrono::steady_clock::now();
    SoftmaxResult r0, r1;
    std::exception_ptr exc0;
    std::exception_ptr exc1;
    auto nexp_bundle0 = gates::make_nexp_cubic_bundle(nexp_mat, fb);
    auto recip_bundle0 = gates::make_recip_bundle(recip_mat);
    runtime::TruncChoice prob_choice;
    prob_choice.gapars = &recip_mat.trunc_fb;
    prob_choice.faithful = prob_choice.gapars;
    prob_choice.shift_bits = fb;
    prob_choice.signed_value = false;
    std::thread t([&]() {
      try {
        r1 = run_softmax_party(1, rows, cols, fb, valid, nexp_bundle0, recip_bundle0, prob_choice, &rb1, be1, c1, std::span<const uint64_t>(t1.data(), t1.size()), device_timing || device_only);
      } catch (...) { exc1 = std::current_exception(); }
    });
    float dev_ms = 0.0f;
    cudaEvent_t ev_start = nullptr, ev_end = nullptr;
    bool want_dev_time = device_timing || device_only;
    proto::PfssGpuStagedEval* staged = want_dev_time ? dynamic_cast<proto::PfssGpuStagedEval*>(&be0) : nullptr;
    cudaStream_t stream = staged ? reinterpret_cast<cudaStream_t>(staged->device_stream()) : nullptr;
    if (std::getenv("SOFTMAX_BENCH_TRACE")) {
      std::fprintf(stderr, "[bench] staged=%p stream=%p want_dev_time=%d\n",
                   static_cast<void*>(staged), static_cast<void*>(stream), want_dev_time ? 1 : 0);
    }
    if (staged) {
      if (stream == nullptr) stream = 0;
      cudaEventCreate(&ev_start);
      cudaEventCreate(&ev_end);
      cudaEventRecord(ev_start, stream);
    }
    try {
      r0 = run_softmax_party(0, rows, cols, fb, valid, nexp_bundle0, recip_bundle0, prob_choice, &rb0, be0, c0, std::span<const uint64_t>(t0.data(), t0.size()), device_timing || device_only);
    } catch (...) {
      exc0 = std::current_exception();
    }
    t.join();
    if (staged) {
      cudaEventRecord(ev_end, stream);
      cudaEventSynchronize(ev_end);
      cudaEventElapsedTime(&dev_ms, ev_start, ev_end);
      cudaEventDestroy(ev_start);
      cudaEventDestroy(ev_end);
    }
    if (exc0) std::rethrow_exception(exc0);
    if (exc1) std::rethrow_exception(exc1);
    auto end = std::chrono::steady_clock::now();
    if (std::getenv("SOFTMAX_BENCH_TRACE")) {
      std::fprintf(stderr, "[bench] run_pair finished host_ms=%.3f dev_ms=%.3f\n",
                   std::chrono::duration<double, std::milli>(end - start).count(),
                   static_cast<double>(dev_ms));
    }
    if (!device_only) {
      const char* dbg = std::getenv("SOFTMAX_BENCH_DEBUG");
      for (size_t i = 0; i < r0.probs.size(); ++i) {
        uint64_t a = r0.probs[i];
        uint64_t b = r1.probs[i];
        uint64_t rec = a + b;
        if (rec > (1ull << fb) + 4) {
          if (dbg) {
            std::cerr << "softmax recon mismatch idx=" << i
                      << " a=" << a << " b=" << b
                      << " rec=" << rec << " fb=" << fb
                      << " rows=" << rows << " cols=" << cols
                      << " party0_first=" << (r0.probs.empty() ? 0 : r0.probs[0])
                      << " party1_first=" << (r1.probs.empty() ? 0 : r1.probs[0])
                      << "\n";
          }
          throw std::runtime_error("softmax recon mismatch");
        }
      }
    }
    double host_ms = std::chrono::duration<double, std::milli>(end - start).count();
    if (device_only && dev_ms > 0.0f) host_ms = dev_ms;
    return {host_ms, static_cast<double>(dev_ms)};
  };

  BenchResult br{};
  double cpu_sum = 0.0, gpu_sum = 0.0;
  if (!device_only) {
    for (int i = 0; i < reps; ++i) cpu_sum += run_pair(ref_be, ref_be, nexp_mat_cpu, recip_mat_cpu, /*device_timing=*/false).first;
    br.cpu_ms = cpu_sum / reps;
  }
#ifdef SUF_HAVE_CUDA
  // Exercise CUDA kernels by default for the GPU run.
  setenv("SUF_PFSS_CACHE_KEYS", "1", 0);
  setenv("SUF_PFSS_CACHE_HATX", "1", 0);
  setenv("SUF_TRUNC_GPU", "1", 0);
  setenv("SUF_MUL_GPU", "1", 0);
  setenv("SUF_RECIP_GPU_KERNELS", "1", 0);
  setenv("SUF_HORNER_GPU", "1", 0);
  auto gpu_be0 = proto::make_real_gpu_backend();
  auto gpu_be1 = proto::make_real_gpu_backend();
  if (gpu_be0 && gpu_be1) {
    std::mt19937_64 rng_gpu(101);
    auto nexp_mat_gpu = gates::dealer_make_nexp_task_material(*gpu_be0, nexp_params, rng_gpu, cols * rows, cols * rows);
    auto recip_mat_gpu = gates::dealer_make_recip_task_material(*gpu_be0, fb, /*nr_iters=*/1, rng_gpu, cols * rows);
    double gpu_dev_sum = 0.0;
    bool dev_time = device_only || (std::getenv("SUF_BENCH_DEVICE_TIME") != nullptr);
    setenv("SUF_SOFTMAX_GPU", "1", 0);
    setenv("SUF_LN_GPU", "1", 0);
    for (int i = 0; i < reps; ++i) {
      auto [host_ms, dev_ms] = run_pair(*gpu_be0, *gpu_be1, nexp_mat_gpu, recip_mat_gpu, dev_time);
      gpu_sum += host_ms;
      gpu_dev_sum += dev_ms;
    }
    br.gpu_ms = gpu_sum / reps;
    br.gpu_dev_ms = dev_time ? (gpu_dev_sum / reps) : 0.0;
  }
#endif
  return br;
}

BenchResult bench_layernorm(int rows, int cols, int fb, int reps, bool device_only) {
  std::mt19937_64 rng(17);
  std::vector<uint64_t> x_plain(static_cast<size_t>(rows * cols));
  for (auto& v : x_plain) v = static_cast<uint64_t>(rng() % (1 << fb));
  std::vector<uint64_t> x0 = x_plain, x1(x_plain.size(), 0ull);

  proto::ReferenceBackend ref_be;

  auto bundle_cpu = build_layernorm_bundle(ref_be, rows, cols, fb);
  auto run_pair = [&](proto::PfssBackendBatch& be0, proto::PfssBackendBatch& be1,
                      const LayerNormBundle& b0, const LayerNormBundle& b1,
                      bool device_timing) -> std::pair<double, double> {
    LocalChan::Shared sh;
    LocalChan c0(&sh, true), c1(&sh, false);
    RowBroadcastTripleProviderImpl rb0(b0.rb_mat, 0), rb1(b1.rb_mat, 1);

    runtime::PhaseExecutor pe0, pe1;
    if (device_timing || device_only) {
      pe0.set_device_pipeline(true);
      pe1.set_device_pipeline(true);
      pe0.set_device_pipeline_materialize(false);
      pe1.set_device_pipeline_materialize(false);
    }
    runtime::PhaseResources R0, R1;
    R0.party = 0; R1.party = 1;
    runtime::ProtoChanFromNet pch0(c0), pch1(c1);
    R0.pfss_backend = &be0; R1.pfss_backend = &be1;
    R0.pfss_chan = &pch0; R1.pfss_chan = &pch1;
    R0.net_chan = &c0; R1.net_chan = &c1;
    R0.pfss_coeff = &pe0.pfss_coeff_batch(); R0.pfss_trunc = &pe0.pfss_trunc_batch();
    R1.pfss_coeff = &pe1.pfss_coeff_batch(); R1.pfss_trunc = &pe1.pfss_trunc_batch();
    R0.opens = &pe0.open_collector(); R1.opens = &pe1.open_collector();
    if (device_timing || device_only) {
      pe0.pfss_coeff_batch().set_device_outputs(true);
      pe0.pfss_trunc_batch().set_device_outputs(true);
      pe1.pfss_coeff_batch().set_device_outputs(true);
      pe1.pfss_trunc_batch().set_device_outputs(true);
    }
#ifdef SUF_HAVE_CUDA
    std::unique_ptr<runtime::CudaPfssStager> cuda_stager0;
    std::unique_ptr<runtime::CudaPfssStager> cuda_stager1;
    if (device_timing || device_only) {
      if (auto* staged0 = dynamic_cast<proto::PfssGpuStagedEval*>(&be0)) {
        cuda_stager0 = std::make_unique<runtime::CudaPfssStager>(staged0->device_stream());
        pe0.pfss_coeff_batch().set_gpu_stager(cuda_stager0.get());
        pe0.pfss_trunc_batch().set_gpu_stager(cuda_stager0.get());
      }
      if (auto* staged1 = dynamic_cast<proto::PfssGpuStagedEval*>(&be1)) {
        cuda_stager1 = std::make_unique<runtime::CudaPfssStager>(staged1->device_stream());
        pe1.pfss_coeff_batch().set_gpu_stager(cuda_stager1.get());
        pe1.pfss_trunc_batch().set_gpu_stager(cuda_stager1.get());
      }
    }
#endif

    std::vector<uint64_t> y0(x0.size(), 0ull), y1(x1.size(), 0ull);
    auto bundle0 = b0.bundle;
    auto bundle1 = b1.bundle;
    bundle0.row_triples = &rb0;
    bundle1.row_triples = &rb1;
    bundle0.mul_triples = std::span<const proto::BeaverTriple64Share>(b0.mul_triples0);
    bundle1.mul_triples = std::span<const proto::BeaverTriple64Share>(b1.mul_triples1);

    auto start = std::chrono::steady_clock::now();
    float dev_ms = 0.0f;
    cudaEvent_t ev_start = nullptr, ev_end = nullptr;
    bool want_dev_time = device_timing || device_only;
    proto::PfssGpuStagedEval* staged = want_dev_time ? dynamic_cast<proto::PfssGpuStagedEval*>(&be0) : nullptr;
    cudaStream_t stream = staged ? reinterpret_cast<cudaStream_t>(staged->device_stream()) : nullptr;
    if (staged) {
      if (stream == nullptr) stream = 0;
      cudaEventCreate(&ev_start);
      cudaEventCreate(&ev_end);
      cudaEventRecord(ev_start, stream);
    }
    std::exception_ptr exc0;
    std::exception_ptr exc1;
    std::thread t([&]() {
      try {
        auto task = std::make_unique<runtime::LayerNormTask>(
            bundle1,
            std::span<const uint64_t>(x1.data(), x1.size()),
            std::span<uint64_t>(y1.data(), y1.size()),
            rows,
            cols);
        pe1.add_task(std::move(task));
        pe1.run(R1);
      } catch (...) { exc1 = std::current_exception(); }
    });
    auto task0 = std::make_unique<runtime::LayerNormTask>(
        bundle0,
        std::span<const uint64_t>(x0.data(), x0.size()),
        std::span<uint64_t>(y0.data(), y0.size()),
        rows,
        cols);
    try {
      pe0.add_task(std::move(task0));
      pe0.run(R0);
    } catch (...) {
      exc0 = std::current_exception();
    }
    t.join();
    if (staged) {
      cudaEventRecord(ev_end, stream);
      cudaEventSynchronize(ev_end);
      cudaEventElapsedTime(&dev_ms, ev_start, ev_end);
      cudaEventDestroy(ev_start);
      cudaEventDestroy(ev_end);
    }
    if (exc0) std::rethrow_exception(exc0);
    if (exc1) std::rethrow_exception(exc1);
    auto end = std::chrono::steady_clock::now();
    if (!device_only) {
      for (size_t i = 0; i < y0.size(); ++i) {
        volatile uint64_t rec = y0[i] + y1[i];
        (void)rec;
      }
    }
    double host_ms = std::chrono::duration<double, std::milli>(end - start).count();
    if (device_only && dev_ms > 0.0f) host_ms = dev_ms;
    return {host_ms, static_cast<double>(dev_ms)};
  };

  BenchResult br{};
  double cpu_sum = 0.0, gpu_sum = 0.0;
  if (!device_only) {
    for (int i = 0; i < reps; ++i) cpu_sum += run_pair(ref_be, ref_be, bundle_cpu, bundle_cpu, /*device_timing=*/false).first;
    br.cpu_ms = cpu_sum / reps;
  }
#ifdef SUF_HAVE_CUDA
  setenv("SUF_PFSS_CACHE_KEYS", "1", 0);
  setenv("SUF_PFSS_CACHE_HATX", "1", 0);
  setenv("SUF_TRUNC_GPU", "1", 0);
  setenv("SUF_MUL_GPU", "1", 0);
  setenv("SUF_RECIP_GPU_KERNELS", "1", 0);
  setenv("SUF_HORNER_GPU", "1", 0);
  auto gpu_be0 = proto::make_real_gpu_backend();
  auto gpu_be1 = proto::make_real_gpu_backend();
  if (gpu_be0 && gpu_be1) {
    auto bundle_gpu = build_layernorm_bundle(*gpu_be0, rows, cols, fb);
    double gpu_dev_sum = 0.0;
    bool dev_time = device_only || (std::getenv("SUF_BENCH_DEVICE_TIME") != nullptr);
    for (int i = 0; i < reps; ++i) {
      auto [host_ms, dev_ms] = run_pair(*gpu_be0, *gpu_be1, bundle_gpu, bundle_gpu, dev_time);
      gpu_sum += host_ms;
      gpu_dev_sum += dev_ms;
    }
    br.gpu_ms = gpu_sum / reps;
    br.gpu_dev_ms = dev_time ? (gpu_dev_sum / reps) : 0.0;
  }
#endif
  return br;
}

}  // namespace

int main(int argc, char** argv) {
  int rows = 4, cols = 8, fb = 8, reps = 2;
  bool device_only = (std::getenv("SUF_BENCH_DEVICE_ONLY") != nullptr);
  int gpu_block = -1;
  std::vector<int> sweep_blocks;
  for (int i = 1; i < argc; ++i) {
    std::string a(argv[i]);
    if (a.rfind("--rows=", 0) == 0) rows = std::stoi(a.substr(7));
    else if (a.rfind("--cols=", 0) == 0) cols = std::stoi(a.substr(7));
    else if (a.rfind("--frac_bits=", 0) == 0) fb = std::stoi(a.substr(12));
    else if (a.rfind("--reps=", 0) == 0) reps = std::stoi(a.substr(7));
    else if (a.rfind("--preset=", 0) == 0) {
      std::string p = a.substr(9);
      if (p == "safe") { rows = 256; cols = 128; reps = 1; }
      else if (p == "tiny") { rows = 4; cols = 8; reps = 2; }
      else if (p == "large") { rows = 4096; cols = 512; reps = 2; }
    }
    else if (a == "--device-only") {
      device_only = true;
    } else if (a.rfind("--blocks=", 0) == 0) {
      gpu_block = std::stoi(a.substr(9));
    } else if (a.rfind("--sweep-blocks=", 0) == 0) {
      std::string list = a.substr(15);
      size_t pos = 0;
      while (pos < list.size()) {
        size_t comma = list.find(',', pos);
        std::string tok = list.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (!tok.empty()) sweep_blocks.push_back(std::stoi(tok));
        if (comma == std::string::npos) break;
        pos = comma + 1;
      }
    }
  }

  if (gpu_block > 0) {
    std::string b = std::to_string(gpu_block);
    setenv("SUF_PFSS_GPU_BLOCK", b.c_str(), 1);
  }

  std::cout << std::unitbuf;
  auto run_once = [&](int blk) {
    if (blk > 0) {
      std::string b = std::to_string(blk);
      setenv("SUF_PFSS_GPU_BLOCK", b.c_str(), 1);
    }
    std::cout << "Softmax bench rows=" << rows << " cols=" << cols << " fb=" << fb
              << " reps=" << reps;
    if (blk > 0) std::cout << " block=" << blk;
    std::cout << "\n";
    auto soft = bench_softmax(rows, cols, fb, reps, device_only);
    if (!device_only) std::cout << "Softmax CPU avg_ms=" << soft.cpu_ms << " ";
    std::cout << "GPU avg_ms=" << soft.gpu_ms;
    if (soft.gpu_dev_ms > 0.0) std::cout << " GPU_device_ms=" << soft.gpu_dev_ms;
    std::cout << "\n";

    std::cout << "LayerNorm bench rows=" << rows << " cols=" << cols << " fb=" << fb
              << " reps=" << reps;
    if (blk > 0) std::cout << " block=" << blk;
    std::cout << "\n";
    auto ln = bench_layernorm(rows, cols, fb, reps, device_only);
    if (!device_only) std::cout << "LayerNorm CPU avg_ms=" << ln.cpu_ms << " ";
    std::cout << "GPU avg_ms=" << ln.gpu_ms;
    if (ln.gpu_dev_ms > 0.0) std::cout << " GPU_device_ms=" << ln.gpu_dev_ms;
    std::cout << "\n";
  };

  if (!sweep_blocks.empty()) {
    try {
      for (int blk : sweep_blocks) run_once(blk);
    } catch (const std::exception& e) {
      std::cerr << "bench_softmax_norm failed: " << e.what() << "\n";
      return 1;
    } catch (...) {
      std::cerr << "bench_softmax_norm failed: unknown\n";
      return 1;
    }
  } else {
    try {
      run_once(gpu_block);
    } catch (const std::exception& e) {
      std::cerr << "bench_softmax_norm failed: " << e.what() << "\n";
      return 1;
    } catch (...) {
      std::cerr << "bench_softmax_norm failed: unknown\n";
      return 1;
    }
  }
  return 0;
}
