#include <cmath>
#include <cstdint>
#include <random>
#include <thread>
#include <vector>
#include <cassert>
#include <iostream>
#include <cstdlib>
#include <mutex>
#include <condition_variable>
#include <queue>

#include "mpc/net.hpp"
#include "proto/bit_ring_ops.hpp"
#include "proto/pfss_backend.hpp"
#include "proto/backend_clear.hpp"
#include "gates/nexp_composite.hpp"
#include "gates/reciprocal_composite.hpp"
#include "nn/softmax_block_task.hpp"
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

void ensure_triples(compiler::TruncationLoweringResult& bundle, size_t need, std::mt19937_64& rng) {
  auto fill = [&](std::vector<proto::BeaverTriple64Share>& dst0,
                  std::vector<proto::BeaverTriple64Share>& dst1) {
    while (dst0.size() < need || dst1.size() < need) {
      uint64_t a = rng();
      uint64_t b = rng();
      uint64_t c = proto::mul_mod(a, b);
      uint64_t a0 = rng();
      uint64_t a1 = a - a0;
      uint64_t b0 = rng();
      uint64_t b1 = b - b0;
      uint64_t c0 = rng();
      uint64_t c1 = c - c0;
      dst0.push_back({a0, b0, c0});
      dst1.push_back({a1, b1, c1});
    }
  };
  fill(bundle.keys.k0.triples, bundle.keys.k1.triples);
}

void ensure_triples(gates::CompositeKeyPair& kp, size_t need, std::mt19937_64& rng) {
  auto fill = [&](std::vector<proto::BeaverTriple64Share>& dst0,
                  std::vector<proto::BeaverTriple64Share>& dst1) {
    while (dst0.size() < need || dst1.size() < need) {
      uint64_t a = rng();
      uint64_t b = rng();
      uint64_t c = proto::mul_mod(a, b);
      uint64_t a0 = rng();
      uint64_t a1 = a - a0;
      uint64_t b0 = rng();
      uint64_t b1 = b - b0;
      uint64_t c0 = rng();
      uint64_t c1 = c - c0;
      dst0.push_back({a0, b0, c0});
      dst1.push_back({a1, b1, c1});
    }
  };
  fill(kp.k0.triples, kp.k1.triples);
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

struct PartyResult {
  std::vector<uint64_t> out;
  runtime::PhaseExecutor::Stats stats;
  std::vector<uint64_t> exp;
  std::vector<uint64_t> sum;
  std::vector<uint64_t> inv;
  std::vector<uint64_t> prod;
};

PartyResult run_party(int party,
                      nn::SoftmaxPlan plan,
                      std::span<const uint64_t> t_share,
                      net::Chan& ch,
                      proto::PfssBackendBatch& backend) {
  runtime::PhaseExecutor pe;
  pe.reset_stats();
  runtime::PhaseResources R;
  runtime::ProtoChanFromNet pch(ch);
  R.party = party;
  R.pfss_backend = &backend;
  R.pfss_chan = &pch;
  R.net_chan = &ch;
  R.pfss_coeff = &pe.pfss_coeff_batch();
  R.pfss_trunc = &pe.pfss_trunc_batch();
  R.opens = &pe.open_collector();

  std::vector<uint64_t> out(plan.rows * plan.cols, 0);
  pe.begin_phase(runtime::PhaseExecutor::Phase::kSoftmax);
  auto sm_task = std::make_unique<nn::SoftmaxBlockTask>(plan, t_share, std::span<uint64_t>(out.data(), out.size()));
  auto* sm_ptr = sm_task.get();
  pe.add_task(std::move(sm_task));
  pe.run(R);
  PartyResult res;
  res.out = std::move(out);
  res.stats = pe.stats();
  if (sm_ptr) {
    res.exp = sm_ptr->exp_qf_debug();
    res.sum = sm_ptr->sum_qf_debug();
    res.inv = sm_ptr->inv_qf_debug();
    res.prod = sm_ptr->prod_q2f_debug();
  }
  return res;
}

void test_correctness_and_flush() {
  const int fb = 16;
  const int rows = 2;
  const int cols = 3;
  // Inputs are already max-score deltas t >= 0 (no additional max step inside the task).
  std::vector<int64_t> plain(rows * cols);
  plain[0] = (2ll << fb);
  plain[1] = (1ll << fb);
  plain[2] = 0;
  plain[3] = 0;
  plain[4] = (1ll << fb);
  plain[5] = (2ll << fb);

  proto::ClearBackend backend;
  gates::NExpGateParams nexp_params;
  nexp_params.frac_bits = fb;
  nexp_params.segments = 16;
  auto nexp_spec = gates::make_nexp_spec(nexp_params);
  std::mt19937_64 rng(1234);
  auto nexp_mat = gates::dealer_make_nexp_task_material(backend, nexp_params, rng, /*triple_need=*/rows * cols * 8);
  ensure_triples(nexp_mat.keys, 4096, rng);
  ensure_triples(nexp_mat.trunc_f, 4096, rng);
  ensure_triples(nexp_mat.trunc_2f, 4096, rng);
  std::fill(nexp_mat.keys.k0.r_out_share.begin(), nexp_mat.keys.k0.r_out_share.end(), 0ull);
  std::fill(nexp_mat.keys.k1.r_out_share.begin(), nexp_mat.keys.k1.r_out_share.end(), 0ull);
  auto nexp_bundle = gates::make_nexp_cubic_bundle(nexp_mat, fb);
  auto recip_mat = gates::dealer_make_recip_task_material(backend, fb, /*nr_iters=*/1, rng);
  ensure_triples(recip_mat.keys, 4096, rng);
  ensure_triples(recip_mat.trunc_fb, 4096, rng);
  std::fill(recip_mat.keys.k0.r_out_share.begin(), recip_mat.keys.k0.r_out_share.end(), 0ull);
  std::fill(recip_mat.keys.k1.r_out_share.begin(), recip_mat.keys.k1.r_out_share.end(), 0ull);
  auto recip_bundle = gates::make_recip_bundle(recip_mat);

  compiler::GateParams gap_p;
  gap_p.kind = compiler::GateKind::GapARS;
  gap_p.frac_bits = fb;
  auto prob_gapars = compiler::lower_truncation_gate(backend, rng, gap_p);
  compiler::GateParams faithful_p;
  faithful_p.kind = compiler::GateKind::FaithfulTR;
  faithful_p.frac_bits = fb;
  auto prob_faithful = compiler::lower_truncation_gate(backend, rng, faithful_p);
  size_t triple_need = 4096;
  ensure_triples(prob_gapars.keys, triple_need, rng);
  ensure_triples(prob_faithful.keys, triple_need, rng);
  ensure_triples(prob_gapars, triple_need, rng);
  ensure_triples(prob_faithful, triple_need, rng);
  std::fill(prob_gapars.keys.k0.r_out_share.begin(), prob_gapars.keys.k0.r_out_share.end(), 0ull);
  std::fill(prob_gapars.keys.k1.r_out_share.begin(), prob_gapars.keys.k1.r_out_share.end(), 0ull);
  std::fill(prob_faithful.keys.k0.r_out_share.begin(), prob_faithful.keys.k0.r_out_share.end(), 0ull);
  std::fill(prob_faithful.keys.k1.r_out_share.begin(), prob_faithful.keys.k1.r_out_share.end(), 0ull);

  RowBroadcastTripleMaterial triple_mat = make_row_broadcast_triples(rows, cols, rng);
  RowBroadcastTripleProviderImpl triples0(triple_mat, 0);
  RowBroadcastTripleProviderImpl triples1(triple_mat, 1);

  nn::SoftmaxPlan plan0;
  plan0.frac_bits = fb;
  plan0.rows = rows;
  plan0.cols = cols;
  plan0.nexp = nexp_bundle;
  plan0.recip = recip_bundle;
  plan0.prob_trunc.gapars = &prob_gapars;
  plan0.prob_trunc.faithful = &prob_faithful;
  plan0.prob_trunc.shift_bits = fb;
  plan0.prob_trunc.signed_value = true;
  plan0.row_triples = &triples0;

  nn::SoftmaxPlan plan1 = plan0;
  plan1.row_triples = &triples1;

  std::vector<uint64_t> t0(plain.size()), t1(plain.size());
  for (size_t i = 0; i < plain.size(); ++i) {
    uint64_t r = rng();
    t0[i] = r;
    t1[i] = static_cast<uint64_t>(plain[i]) - r;
  }

  LocalChan::Shared sh;
  LocalChan c0(&sh, true), c1(&sh, false);
  PartyResult res0, res1;
  bool party_fail = false;
  std::string party_err;
  std::thread th1([&] {
    try {
      res1 = run_party(1, plan1, std::span<const uint64_t>(t1.data(), t1.size()), c1, backend);
    } catch (const std::exception& e) {
      party_fail = true;
      party_err = e.what();
    }
  });
  try {
    res0 = run_party(0, plan0, std::span<const uint64_t>(t0.data(), t0.size()), c0, backend);
  } catch (const std::exception& e) {
    party_fail = true;
    party_err = e.what();
  }
  th1.join();
  if (party_fail) {
    std::cerr << "party execution failed: " << party_err << "\n";
  }

  auto recip_spec = gates::make_recip_affine_init_spec(fb, /*nmax=*/1024.0);
  std::vector<int64_t> expected(plain.size());
  std::vector<int64_t> expected_hook(plain.size());
  std::vector<int64_t> expected_exp(plain.size());
  std::vector<uint64_t> expected_sum(static_cast<size_t>(rows), 0);
  std::vector<int64_t> expected_inv(static_cast<size_t>(rows), 0);
  for (int r = 0; r < rows; ++r) {
    uint64_t sum = 0;
    for (int c = 0; c < cols; ++c) {
      size_t idx = static_cast<size_t>(r * cols + c);
      int64_t exp_v = gates::ref_nexp_fixed(nexp_spec, plain[idx]);
      expected[idx] = exp_v;
      expected_exp[idx] = exp_v;
      sum = proto::add_mod(sum, static_cast<uint64_t>(exp_v));
    }
    expected_sum[static_cast<size_t>(r)] = sum;
    int64_t inv = gates::ref_reciprocal_fixed(recip_spec, static_cast<int64_t>(sum), fb, /*nr_iters=*/1);
    expected_inv[static_cast<size_t>(r)] = inv;
    for (int c = 0; c < cols; ++c) {
      size_t idx = static_cast<size_t>(r * cols + c);
      __int128 prod = static_cast<__int128>(expected[idx]) * static_cast<__int128>(inv);
      expected[idx] = static_cast<int64_t>(prod >> fb);
    }
  }

  // Also compute Horner-hook output directly using the composite + hook path for comparison.
  auto run_hook_exp = [&](int party, LocalChan& lc) {
    try {
      auto hook = std::make_unique<gates::HornerCubicHook>(fb, (party == 0) ? nexp_mat.keys.k0.r_in_share
                                                                             : nexp_mat.keys.k1.r_in_share);
      hook->backend = &backend;
      hook->trunc_frac_bits = fb;
      runtime::ProtoChanFromNet pch(lc);
      gates::CompositeBatchInput in;
      uint64_t r_in = proto::add_mod(nexp_mat.keys.k0.r_in_share, nexp_mat.keys.k1.r_in_share);
      std::vector<uint64_t> hatx(plain.size());
      for (size_t i = 0; i < plain.size(); ++i) {
        hatx[i] = proto::add_mod(static_cast<uint64_t>(plain[i]), r_in);
      }
      in.hatx = hatx.data();
      in.N = hatx.size();
      auto out = gates::composite_eval_batch_with_postproc(party,
                                                           backend,
                                                           pch,
                                                           (party == 0) ? nexp_mat.keys.k0 : nexp_mat.keys.k1,
                                                           nexp_mat.suf,
                                                           in,
                                                           *hook);
      return out.haty_share;
    } catch (const std::exception& e) {
      std::cerr << "run_hook_exp party " << party << " exception: " << e.what() << "\n";
      return std::vector<uint64_t>{};
    }
  };

  std::vector<uint64_t> hook0(plain.size(), 0), hook1(plain.size(), 0);
  bool hook_computed = false;

  auto recon_vec = [&](const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
    std::vector<uint64_t> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
      out[i] = proto::add_mod(a[i], b[i]);
    }
    return out;
  };
  {
    LocalChan::Shared hook_sh;
    LocalChan hc0(&hook_sh, true), hc1(&hook_sh, false);
    std::vector<uint64_t> h0, h1;
    std::thread ht([&] { h1 = run_hook_exp(1, hc1); });
    h0 = run_hook_exp(0, hc0);
    ht.join();
    if (!h0.empty() && h0.size() == h1.size()) {
      hook0 = std::move(h0);
      hook1 = std::move(h1);
      hook_computed = true;
    }
  }
  auto exp_recon = recon_vec(res0.exp, res1.exp);
  auto hook_recon = recon_vec(hook0, hook1);
  auto sum_recon = recon_vec(res0.sum, res1.sum);
  auto inv_recon = recon_vec(res0.inv, res1.inv);
  auto prod_recon = recon_vec(res0.prod, res1.prod);

  auto print_stage = [&](const char* label, size_t idx, int64_t got, int64_t want) {
    std::cerr << label << " mismatch idx " << idx << " got " << got << " expected " << want
              << " diff " << (got - want) << "\n";
  };

  bool stage_ok = true;
  for (size_t i = 0; i < exp_recon.size(); ++i) {
    int64_t got = static_cast<int64_t>(exp_recon[i]);
    if (std::llabs(got - expected_exp[i]) > 1) {
      stage_ok = false;
      print_stage("exp", i, got, expected_exp[i]);
    }
    if (hook_computed) {
      int64_t got_hook = static_cast<int64_t>(hook_recon[i]);
      if (std::llabs(got_hook - expected_exp[i]) > 1) {
        stage_ok = false;
        print_stage("exp_hook", i, got_hook, expected_exp[i]);
      }
    }
  }
  for (size_t r = 0; r < sum_recon.size(); ++r) {
    int64_t got = static_cast<int64_t>(sum_recon[r]);
    int64_t want = static_cast<int64_t>(expected_sum[r]);
    if (std::llabs(got - want) > 1) {
      stage_ok = false;
      print_stage("sum", r, got, want);
    }
  }
  for (size_t r = 0; r < inv_recon.size(); ++r) {
    int64_t got = static_cast<int64_t>(inv_recon[r]);
    int64_t want = expected_inv[r];
    if (std::llabs(got - want) > 1) {
      stage_ok = false;
      print_stage("inv", r, got, want);
    }
  }
  for (size_t i = 0; i < prod_recon.size(); ++i) {
    int64_t got = static_cast<int64_t>(prod_recon[i] >> fb);  // prod is Q2f; downscale to compare
    int64_t want = static_cast<int64_t>(static_cast<__int128>(expected_exp[i]) * expected_inv[i / cols] >> fb);
    if (std::llabs(got - want) > 1) {
      stage_ok = false;
      print_stage("prod", i, got, want);
    }
  }

  // Reconstruct and compare against reference softmax (same spline/NR approximations).
  bool all_close = true;
  for (size_t i = 0; i < plain.size(); ++i) {
    uint64_t recon = proto::add_mod(res0.out[i], res1.out[i]);
    int64_t recon_s = static_cast<int64_t>(recon);
    int64_t diff = recon_s - expected[i];
    if (std::llabs(diff) > 1) {
      all_close = false;
      std::cerr << "softmax mismatch idx " << i << ": share0 " << res0.out[i] << " share1 "
                << res1.out[i] << " recon " << recon_s << " expected "
                << expected[i] << " diff " << diff << "\n";
    }
  }
  if (!stage_ok) {
    std::cerr << "stage mismatch detected\n";
  }
  assert(all_close);  // within 1 LSB at Q16

  auto check_stats = [&](const runtime::PhaseExecutor::Stats& s) {
    size_t pfss_flushes = s.pfss_coeff_flushes + s.pfss_trunc_flushes;
    size_t pfss_jobs = s.pfss_coeff_jobs + s.pfss_trunc_jobs;
    // Expect a modest number of flushes for this tiny problem: a handful of waves.
    assert(pfss_flushes > 0 && pfss_flushes <= 8);
    assert(s.open_flushes > 0 && s.open_flushes <= 16);
    assert(pfss_jobs > 0);
  };
  check_stats(res0.stats);
  check_stats(res1.stats);

  // Flush counts should stay small for this tiny problem.
  assert(res0.stats.open_flushes > 0);
  assert(res0.stats.pfss_coeff_flushes + res0.stats.pfss_trunc_flushes > 0);
  assert(res0.stats.pfss_coeff_jobs + res0.stats.pfss_trunc_jobs > 0);
}

}  // namespace

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  test_correctness_and_flush();
  std::cout << "softmax task tests passed\n";
  return 0;
}
