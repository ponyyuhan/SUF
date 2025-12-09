#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <thread>
#include <vector>
#include <cuda_runtime.h>

#include "gates/nexp_composite.hpp"
#include "gates/reciprocal_composite.hpp"
#include "nn/softmax_block_task.hpp"
#include "proto/backend_gpu.hpp"
#include "proto/reference_backend.hpp"
#include "runtime/pfss_phase_planner.hpp"
#include "runtime/pfss_superbatch.hpp"
#include "runtime/phase_executor.hpp"

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
  std::vector<uint64_t> exp_qf;
  std::vector<uint64_t> sum_qf;
  std::vector<uint64_t> inv_qf;
  std::vector<uint64_t> prod_q2f;
  std::vector<uint64_t> prob_qf;
};

std::vector<int64_t> ref_softmax(const std::vector<int64_t>& t_qf, int rows, int cols, int fb) {
  std::vector<int64_t> out(t_qf.size(), 0);
  auto spec = gates::make_nexp_spec(gates::NExpGateParams{fb, 16});
  auto recip_spec = gates::make_recip_affine_init_spec(fb, 1024.0);
  for (int r = 0; r < rows; ++r) {
    int row_start = r * cols;
    int64_t max_sc = t_qf[row_start];
    for (int c = 1; c < cols; ++c) max_sc = std::max<int64_t>(max_sc, t_qf[row_start + c]);
    std::vector<int64_t> expv(cols, 0);
    int64_t sum = 0;
    for (int c = 0; c < cols; ++c) {
      int64_t diff = max_sc - t_qf[row_start + c];
      expv[c] = gates::ref_nexp_fixed(spec, diff);
      sum += expv[c];
    }
    if (sum == 0) sum = 1;
    int64_t inv = gates::ref_reciprocal_fixed(recip_spec, sum, fb, 1);
    for (int c = 0; c < cols; ++c) {
      __int128 p = static_cast<__int128>(expv[c]) * static_cast<__int128>(inv);
      out[row_start + c] = static_cast<int64_t>(p >> fb);
    }
  }
  return out;
}

SoftmaxResult run_party(int party,
                        int rows,
                        int cols,
                        int fb,
                        const std::vector<uint64_t>& t_share,
                        const gates::NexpTaskMaterial& nexp_mat,
                        const gates::RecipTaskMaterial& recip_mat,
                        runtime::TruncChoice prob_choice,
                        runtime::RowBroadcastTripleProvider* rb,
                        proto::PfssBackendBatch& backend,
                        net::Chan& chan,
                        const std::vector<int>& valid_lens) {
  runtime::PhaseExecutor pe;
  runtime::PhaseResources R;
  R.party = party;
  runtime::ProtoChanFromNet pch(chan);
  R.pfss_backend = &backend;
  R.pfss_chan = &pch;
  R.net_chan = &chan;
  R.pfss_coeff = &pe.pfss_coeff_batch();
  R.pfss_trunc = &pe.pfss_trunc_batch();
  R.opens = &pe.open_collector();
  runtime::PfssPhasePlanner planner;
  planner.bind(R.pfss_coeff, R.pfss_trunc);
  R.pfss_planner = &planner;

  nn::SoftmaxPlan plan;
  plan.frac_bits = fb;
  plan.rows = rows;
  plan.cols = cols;
  plan.valid_lens = valid_lens;
  plan.nexp = gates::make_nexp_cubic_bundle(nexp_mat, fb);
  plan.recip = gates::make_recip_bundle(recip_mat);
  plan.prob_trunc = prob_choice;
  plan.row_triples = rb;
  compiler::RangeInterval prob_range;
  prob_range.lo = 0;
  prob_range.hi = static_cast<int64_t>(1) << fb;
  prob_range.is_signed = false;
  plan.prob_range = prob_range;

  std::vector<uint64_t> probs(plan.rows * plan.cols, 0);
  pe.begin_phase(runtime::PhaseExecutor::Phase::kSoftmax);
  auto task = std::make_unique<nn::SoftmaxBlockTask>(
      plan,
      std::span<const uint64_t>(t_share.data(), t_share.size()),
      std::span<uint64_t>(probs.data(), probs.size()));
  nn::SoftmaxBlockTask* task_raw = task.get();
  pe.add_task(std::move(task));
  pe.run(R);
  SoftmaxResult res;
  res.probs = std::move(probs);
  res.exp_qf = task_raw->exp_qf_debug();
  res.sum_qf = task_raw->sum_qf_debug();
  res.inv_qf = task_raw->inv_qf_debug();
  res.prod_q2f = task_raw->prod_q2f_debug();
  res.prob_qf = task_raw->prob_qf_debug();
  return res;
}

}  // namespace

int main() {
#ifndef SUF_HAVE_CUDA
  std::cout << "SUF_HAVE_CUDA not defined; skipping softmax GPU smoke.\n";
  return 0;
#else
  int devs = 0;
  if (cudaGetDeviceCount(&devs) != cudaSuccess || devs == 0) {
    std::cout << "No CUDA device; skipping softmax GPU smoke.\n";
    return 0;
  }
  auto gpu0 = proto::make_real_gpu_backend();
  auto gpu1 = proto::make_real_gpu_backend();
  if (!gpu0 || !gpu1) {
    std::cout << "GPU backend unavailable; skipping softmax GPU smoke.\n";
    return 0;
  }

  const int rows = 2;
  const int cols = 3;
  const int fb = 8;
  std::vector<int> valid = {3, 2};
  std::mt19937_64 rng(777);
  std::vector<int64_t> t_plain(rows * cols, 0);
  for (auto& v : t_plain) v = static_cast<int64_t>((rng() % 15) - 7);
  std::vector<uint64_t> t0(t_plain.size()), t1(t_plain.size());
  for (size_t i = 0; i < t_plain.size(); i++) {
    uint64_t s0 = rng();
    int64_t plain = t_plain[i];
    t0[i] = s0;
    t1[i] = static_cast<uint64_t>(plain) - s0;
  }

  gates::NExpGateParams nexp_params;
  nexp_params.frac_bits = fb;
  nexp_params.segments = 16;
  auto nexp_mat = gates::dealer_make_nexp_task_material(*gpu0, nexp_params, rng,
                                                        /*triple_need=*/3 * t0.size(),
                                                        t0.size());
  auto recip_mat = gates::dealer_make_recip_task_material(*gpu0, fb, /*nr_iters=*/2, rng, rows);
  auto fill_triples = [&](std::vector<proto::BeaverTriple64Share>& a,
                          std::vector<proto::BeaverTriple64Share>& b,
                          size_t need) {
    while (a.size() < need || b.size() < need) {
      uint64_t x = rng(), y = rng(), z = proto::mul_mod(x, y);
      uint64_t x0 = rng(), y0 = rng(), z0 = rng();
      a.push_back({x0, y0, z0});
      b.push_back({x - x0, y - y0, z - z0});
    }
  };
  fill_triples(nexp_mat.keys.k0.triples, nexp_mat.keys.k1.triples, 1024);
  fill_triples(recip_mat.keys.k0.triples, recip_mat.keys.k1.triples, 1024);

  compiler::GateParams prob_p;
  prob_p.kind = compiler::GateKind::GapARS;
  prob_p.frac_bits = fb;
  auto prob_trunc = compiler::lower_truncation_gate(*gpu0, rng, prob_p, t0.size());
  std::fill(prob_trunc.keys.k0.r_out_share.begin(), prob_trunc.keys.k0.r_out_share.end(), 0ull);
  std::fill(prob_trunc.keys.k1.r_out_share.begin(), prob_trunc.keys.k1.r_out_share.end(), 0ull);
  runtime::TruncChoice prob_choice;
  prob_choice.faithful = &prob_trunc;
  prob_choice.gapars = &prob_trunc;
  prob_choice.shift_bits = fb;
  prob_choice.signed_value = false;

  RowBroadcastTripleMaterial rb_mat = make_row_broadcast_triples(rows, cols, rng);
  RowBroadcastTripleProviderImpl rb0(rb_mat, 0);
  RowBroadcastTripleProviderImpl rb1(rb_mat, 1);

  LocalChan::Shared sh;
  LocalChan c0(&sh, true), c1(&sh, false);

  SoftmaxResult g0, g1, c0_res, c1_res;
  std::atomic<bool> fail{false};
  std::string fail_msg;
  std::thread t_a([&] {
    try {
      g0 = run_party(0, rows, cols, fb, t0, nexp_mat, recip_mat, prob_choice, &rb0, *gpu0, c0, valid);
    } catch (const std::exception& e) {
      fail = true;
      fail_msg = e.what();
    }
  });
  try {
    g1 = run_party(1, rows, cols, fb, t1, nexp_mat, recip_mat, prob_choice, &rb1, *gpu1, c1, valid);
  } catch (const std::exception& e) {
    fail = true;
    fail_msg = e.what();
  }
  t_a.join();
  if (fail) {
    std::cerr << "Softmax GPU smoke failed: " << fail_msg << "\n";
    return 1;
  }

  // CPU reference backend for stage comparison.
  proto::ReferenceBackend ref_backend;
  LocalChan::Shared sh_ref;
  LocalChan rc0(&sh_ref, true), rc1(&sh_ref, false);
  std::thread t_b([&] {
    try {
      c0_res = run_party(0, rows, cols, fb, t0, nexp_mat, recip_mat, prob_choice, &rb0, ref_backend, rc0, valid);
    } catch (const std::exception& e) {
      fail = true;
      fail_msg = e.what();
    }
  });
  try {
    c1_res = run_party(1, rows, cols, fb, t1, nexp_mat, recip_mat, prob_choice, &rb1, ref_backend, rc1, valid);
  } catch (const std::exception& e) {
    fail = true;
    fail_msg = e.what();
  }
  t_b.join();
  if (fail) {
    std::cerr << "Softmax reference run failed: " << fail_msg << "\n";
    return 1;
  }

  auto recon_stage = [](const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
    std::vector<uint64_t> out(a.size());
    for (size_t i = 0; i < a.size(); i++) out[i] = a[i] + b[i];
    return out;
  };

  auto recon = recon_stage(g0.probs, g1.probs);
  auto ref = ref_softmax(t_plain, rows, cols, fb);
  for (int r = 0; r < rows; ++r) {
    int L = valid[r];
    for (int c = 0; c < cols; ++c) {
      size_t idx = static_cast<size_t>(r * cols + c);
      if (c >= L) {
        if (recon[idx] != 0) {
          std::cerr << "Expected padded zero at (" << r << "," << c << ") got " << recon[idx] << "\n";
          return 1;
        }
        continue;
      }
      if (recon[idx] != ref[idx]) {
        std::cerr << "Softmax mismatch at (" << r << "," << c << ") gpu=" << recon[idx]
                  << " ref=" << ref[idx] << "\n";
        auto dump_diff = [&](const char* label,
                             const std::vector<uint64_t>& ga,
                             const std::vector<uint64_t>& gb,
                             const std::vector<uint64_t>& ca,
                             const std::vector<uint64_t>& cb) {
          auto gsum = recon_stage(ga, gb);
          auto csum = recon_stage(ca, cb);
          for (size_t i = 0; i < gsum.size(); i++) {
            if (gsum[i] != csum[i]) {
              std::cerr << label << " mismatch at idx " << i << " gpu=" << gsum[i]
                        << " ref=" << csum[i] << "\n";
              return;
            }
          }
        };
        dump_diff("exp", g0.exp_qf, g1.exp_qf, c0_res.exp_qf, c1_res.exp_qf);
        dump_diff("sum", g0.sum_qf, g1.sum_qf, c0_res.sum_qf, c1_res.sum_qf);
        dump_diff("inv", g0.inv_qf, g1.inv_qf, c0_res.inv_qf, c1_res.inv_qf);
        dump_diff("prod", g0.prod_q2f, g1.prod_q2f, c0_res.prod_q2f, c1_res.prod_q2f);
        dump_diff("prob_qf", g0.prob_qf, g1.prob_qf, c0_res.prob_qf, c1_res.prob_qf);
        return 1;
      }
    }
  }

  std::cout << "Softmax GPU smoke passed.\n";
  return 0;
#endif
}
