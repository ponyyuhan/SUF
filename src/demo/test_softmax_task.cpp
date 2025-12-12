#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <random>
#include <thread>
#include <vector>

#include "gates/nexp_composite.hpp"
#include "gates/reciprocal_composite.hpp"
#include "proto/pfss_backend.hpp"
#include "proto/reference_backend.hpp"
#include "runtime/phase_tasks.hpp"
#include "runtime/pfss_superbatch.hpp"
#include "runtime/pfss_phase_planner.hpp"
#include "nn/softmax_block_task.hpp"
#include "runtime/staged_executor.hpp"

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

struct PartyResult {
  std::vector<uint64_t> probs;
  runtime::PhaseExecutor::Stats stats;
  runtime::PfssPhasePlanner::Stats planner_stats;
  std::vector<uint64_t> exp_qf;
  std::vector<uint64_t> sum_qf;
  std::vector<uint64_t> inv_qf;
  std::vector<uint64_t> prod_q2f;
  std::vector<uint64_t> prob_qf;
};

PartyResult run_party(int party,
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
  runtime::PhaseResources R{};
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
  auto* task_ptr = task.get();
  pe.add_task(std::move(task));
  pe.run(R);

  PartyResult out;
  out.probs = std::move(probs);
  out.stats = pe.stats();
  out.planner_stats = planner.stats();
  if (task_ptr) {
    out.exp_qf = task_ptr->exp_qf_debug();
    out.sum_qf = task_ptr->sum_qf_debug();
    out.inv_qf = task_ptr->inv_qf_debug();
    out.prod_q2f = task_ptr->prod_q2f_debug();
    out.prob_qf = task_ptr->prob_qf_debug();
  }
  return out;
}

std::vector<int64_t> ref_softmax(const std::vector<int64_t>& t_qf, int rows, int cols, int fb) {
  std::vector<int64_t> out(t_qf.size(), 0);
  auto spec = gates::make_nexp_spec(gates::NExpGateParams{fb, 16});
  auto recip_spec = gates::make_recip_affine_init_spec(fb, 1024.0);
  for (int r = 0; r < rows; ++r) {
    std::vector<int64_t> expv(cols, 0);
    int64_t sum = 0;
    for (int c = 0; c < cols; ++c) {
      int64_t v = t_qf[static_cast<size_t>(r * cols + c)];
      expv[c] = gates::ref_nexp_fixed(spec, v);
      sum += expv[c];
    }
    if (sum == 0) sum = 1;
    int64_t inv = gates::ref_reciprocal_fixed(recip_spec, sum, fb, 2);
    for (int c = 0; c < cols; ++c) {
      __int128 prod = static_cast<__int128>(expv[c]) * static_cast<__int128>(inv);
      out[static_cast<size_t>(r * cols + c)] = static_cast<int64_t>(prod >> fb);
    }
  }
  return out;
}

}  // namespace

int main() {
  try {
  const int fb = 16;
  const int rows = 2;
  const int cols = 3;

  // Party0 holds all shares; party1 holds zeros.
  std::vector<uint64_t> raw(rows * cols, 0);
  for (int i = 0; i < rows * cols; ++i) {
    raw[static_cast<size_t>(i)] = static_cast<uint64_t>((i + 1) << fb);
  }
  std::vector<uint64_t> t_share(rows * cols, 0);
  int64_t cap = static_cast<int64_t>(16ll << fb);
  for (int r = 0; r < rows; ++r) {
    uint64_t maxv = 0;
    for (int c = 0; c < cols; ++c) {
      uint64_t v = raw[static_cast<size_t>(r * cols + c)];
      if (v > maxv) maxv = v;
    }
    for (int c = 0; c < cols; ++c) {
      int64_t diff = static_cast<int64_t>(maxv) -
                     static_cast<int64_t>(raw[static_cast<size_t>(r * cols + c)]);
      if (diff < 0) diff = 0;
      if (diff > cap) diff = cap;
      t_share[static_cast<size_t>(r * cols + c)] = static_cast<uint64_t>(diff);
    }
  }

  proto::ReferenceBackend backend;
  std::mt19937_64 rng(1234);

  // Gate materials.
  gates::NExpGateParams nexp_params;
  nexp_params.frac_bits = fb;
  nexp_params.segments = 16;
  auto nexp_mat = gates::dealer_make_nexp_task_material(backend,
                                                        nexp_params,
                                                        rng,
                                                        /*triple_need=*/3 * t_share.size(),
                                                        t_share.size());

  auto recip_mat = gates::dealer_make_recip_task_material(backend,
                                                          fb,
                                                          /*nr_iters=*/2,
                                                          rng,
                                                          rows);
  auto fill_triples = [&](std::vector<proto::BeaverTriple64Share>& t0,
                          std::vector<proto::BeaverTriple64Share>& t1,
                          size_t need) {
    while (t0.size() < need || t1.size() < need) {
      uint64_t a = rng(), b = rng(), c = proto::mul_mod(a, b);
      uint64_t a0 = rng(), b0 = rng(), c0 = rng();
      t0.push_back({a0, b0, c0});
      t1.push_back({a - a0, b - b0, c - c0});
    }
  };
  fill_triples(nexp_mat.keys.k0.triples, nexp_mat.keys.k1.triples, 2048);
  fill_triples(recip_mat.keys.k0.triples, recip_mat.keys.k1.triples, 2048);

  compiler::GateParams prob_p;
  prob_p.kind = compiler::GateKind::GapARS;
  prob_p.frac_bits = fb;
  auto prob_trunc = compiler::lower_truncation_gate(backend, rng, prob_p, t_share.size());
  // Zero r_out so probabilities are unmasked in the test reconstruction.
  std::fill(prob_trunc.keys.k0.r_out_share.begin(), prob_trunc.keys.k0.r_out_share.end(), 0ull);
  std::fill(prob_trunc.keys.k1.r_out_share.begin(), prob_trunc.keys.k1.r_out_share.end(), 0ull);

  runtime::TruncChoice prob_choice;
  prob_choice.faithful = &prob_trunc;
  prob_choice.gapars = &prob_trunc;
  prob_choice.shift_bits = fb;
  prob_choice.signed_value = false;

  RowBroadcastTripleMaterial rb_mat = make_row_broadcast_triples(rows, cols, rng);
  RowBroadcastTripleProviderImpl rb_p0(rb_mat, 0);
  RowBroadcastTripleProviderImpl rb_p1(rb_mat, 1);

  LocalChan::Shared sh;
  LocalChan c0(&sh, true), c1(&sh, false);

  std::cerr << "nexp.r=" << nexp_mat.keys.k0.compiled.r
            << " coeff_out=" << nexp_mat.keys.k0.compiled.coeff.out_words
            << " degree=" << nexp_mat.keys.k0.compiled.degree
            << " r_out=" << nexp_mat.suf.r_out
            << "\n";

  PartyResult r0, r1;
  bool fail = false;
  std::string fail_msg;
  std::thread t0([&] {
    try {
      r0 = run_party(0, rows, cols, fb, t_share, nexp_mat, recip_mat, prob_choice, &rb_p0, backend, c0, {});
    } catch (const std::exception& e) {
      fail = true;
      fail_msg = e.what();
    }
  });
  std::thread t1([&] {
    try {
      // party1 shares are zero.
      std::vector<uint64_t> zero_share(t_share.size(), 0ull);
      r1 = run_party(1, rows, cols, fb, zero_share, nexp_mat, recip_mat, prob_choice, &rb_p1, backend, c1, {});
    } catch (const std::exception& e) {
      fail = true;
      fail_msg = e.what();
    }
  });
  t0.join();
  t1.join();
  if (fail) {
    std::cerr << "Softmax task test failed (thread): " << fail_msg << std::endl;
    return 1;
  }

  std::vector<int64_t> plain(r0.probs.size(), 0);
  for (size_t i = 0; i < plain.size(); ++i) {
    plain[i] = static_cast<int64_t>(r0.probs[i] + r1.probs[i]);
  }
  std::vector<int64_t> t_plain(t_share.size(), 0);
  for (size_t i = 0; i < t_share.size(); ++i) t_plain[i] = static_cast<int64_t>(t_share[i]);
  auto ref = ref_softmax(t_plain, rows, cols, fb);

  for (size_t i = 0; i < plain.size(); ++i) {
    if (std::llabs(static_cast<long long>(plain[i] - ref[i])) > 2) {
      std::cerr << "Mismatch at " << i << ": got " << plain[i] << " ref " << ref[i] << std::endl;
      if (!r0.exp_qf.empty()) {
        size_t idx = 0;
        auto exp_plain = r0.exp_qf[idx] + (r1.exp_qf.empty() ? 0ull : r1.exp_qf[idx]);
        auto sum_plain = (r0.sum_qf.empty() ? 0ull : r0.sum_qf[idx]) +
                         (r1.sum_qf.empty() ? 0ull : r1.sum_qf[idx]);
        auto inv_plain = (r0.inv_qf.empty() ? 0ull : r0.inv_qf[idx]) +
                         (r1.inv_qf.empty() ? 0ull : r1.inv_qf[idx]);
        auto prod_plain = (r0.prod_q2f.empty() ? 0ull : r0.prod_q2f[idx]) +
                          (r1.prod_q2f.empty() ? 0ull : r1.prod_q2f[idx]);
        auto prob_plain = (r0.prob_qf.empty() ? 0ull : r0.prob_qf[idx]) +
                          (r1.prob_qf.empty() ? 0ull : r1.prob_qf[idx]);
        std::cerr << "exp0=" << r0.exp_qf[idx]
                  << " exp1=" << (r1.exp_qf.empty() ? 0ull : r1.exp_qf[idx])
                  << " sum0=" << (r0.sum_qf.empty() ? 0ull : r0.sum_qf[idx])
                  << " sum1=" << (r1.sum_qf.empty() ? 0ull : r1.sum_qf[idx])
                  << " inv0=" << (r0.inv_qf.empty() ? 0ull : r0.inv_qf[idx])
                  << " inv1=" << (r1.inv_qf.empty() ? 0ull : r1.inv_qf[idx])
                  << " prod0=" << (r0.prod_q2f.empty() ? 0ull : r0.prod_q2f[idx])
                  << " prod1=" << (r1.prod_q2f.empty() ? 0ull : r1.prod_q2f[idx])
                  << " prob0=" << (r0.prob_qf.empty() ? 0ull : r0.prob_qf[idx])
                  << " prob1=" << (r1.prob_qf.empty() ? 0ull : r1.prob_qf[idx])
                  << " | plain exp=" << exp_plain
                  << " plain sum=" << sum_plain
                  << " plain inv=" << inv_plain
                  << " plain prod=" << prod_plain
                  << " plain prob=" << prob_plain
                  << " t0=" << (t_plain.empty() ? 0 : t_plain[0])
                  << " ref0=" << ref[0] << std::endl;
      }
      return 1;
    }
  }

  // Flush counts should be bounded (two PFSS flushes: nExp coeff + recip coeff; opens bounded).
  if (r0.stats.pfss_coeff_flushes > 4 || r0.stats.open_flushes > 4) {
    std::cerr << "Unexpected flush counts (pfss=" << r0.stats.pfss_coeff_flushes
              << ", open=" << r0.stats.open_flushes << ")\n";
    return 1;
  }

  if (r0.planner_stats.coeff_flushes > 1 || r0.planner_stats.trunc_flushes > 1) {
    std::cerr << "Unexpected planner flush counts (coeff=" << r0.planner_stats.coeff_flushes
              << ", trunc=" << r0.planner_stats.trunc_flushes << ")\n";
    return 1;
  }

  // Masked variant: cols=4 but only first 2 entries per row valid; PFSS jobs should shrink.
  const int cols_masked = 4;
  std::vector<uint64_t> t_mask(rows * cols_masked, 0);
  for (int i = 0; i < rows * cols_masked; ++i) {
    t_mask[static_cast<size_t>(i)] = static_cast<uint64_t>((i + 1) << fb);
  }
  std::vector<int> valid(rows, 2);
  std::vector<uint64_t> zero_mask(t_mask.size(), 0);
  RowBroadcastTripleMaterial rb_mat_mask = make_row_broadcast_triples(rows, cols_masked, rng);
  RowBroadcastTripleProviderImpl rb_mask_p0(rb_mat_mask, 0);
  RowBroadcastTripleProviderImpl rb_mask_p1(rb_mat_mask, 1);
  std::atomic<bool> fail_mask{false};
  std::string fail_mask_msg;
  PartyResult m0, m1;
  LocalChan::Shared sh_mask;
  LocalChan cm0(&sh_mask, true), cm1(&sh_mask, false);
  std::thread tm0([&] {
    try {
      m0 = run_party(0, rows, cols_masked, fb, t_mask, nexp_mat, recip_mat, prob_choice, &rb_mask_p0, backend, cm0, valid);
    } catch (const std::exception& e) {
      fail_mask = true;
      fail_mask_msg = e.what();
    }
  });
  std::thread tm1([&] {
    try {
      m1 = run_party(1, rows, cols_masked, fb, zero_mask, nexp_mat, recip_mat, prob_choice, &rb_mask_p1, backend, cm1, valid);
    } catch (const std::exception& e) {
      fail_mask = true;
      fail_mask_msg = e.what();
    }
  });
  tm0.join();
  tm1.join();
  if (fail_mask) {
    std::cerr << "Masked softmax task failed (thread): " << fail_mask_msg << std::endl;
    return 1;
  }
  if (m0.stats.pfss_coeff_jobs > r0.stats.pfss_coeff_jobs ||
      m0.stats.pfss_trunc_jobs > r0.stats.pfss_trunc_jobs) {
    std::cerr << "Masked PFSS jobs did not shrink (coeff " << m0.stats.pfss_coeff_jobs
              << " vs " << r0.stats.pfss_coeff_jobs << ", trunc " << m0.stats.pfss_trunc_jobs
              << " vs " << r0.stats.pfss_trunc_jobs << ")\n";
    return 1;
  }
  if (m0.stats.pfss_coeff_hatx_words >= r0.stats.pfss_coeff_hatx_words ||
      m0.stats.pfss_trunc_hatx_words >= r0.stats.pfss_trunc_hatx_words) {
    std::cerr << "Masked PFSS hatx did not shrink (coeff hatx " << m0.stats.pfss_coeff_hatx_words
              << " vs " << r0.stats.pfss_coeff_hatx_words << ", trunc hatx "
              << m0.stats.pfss_trunc_hatx_words << " vs " << r0.stats.pfss_trunc_hatx_words << ")\n";
    return 1;
  }
  if (m0.stats.pfss_coeff_hatx_bytes >= r0.stats.pfss_coeff_hatx_bytes ||
      m0.stats.pfss_trunc_hatx_bytes >= r0.stats.pfss_trunc_hatx_bytes) {
    std::cerr << "Masked PFSS hatx bytes did not shrink (coeff bytes " << m0.stats.pfss_coeff_hatx_bytes
              << " vs " << r0.stats.pfss_coeff_hatx_bytes << ", trunc bytes "
              << m0.stats.pfss_trunc_hatx_bytes << " vs " << r0.stats.pfss_trunc_hatx_bytes << ")\n";
    return 1;
  }
  size_t dense_hatx_bytes = r0.stats.pfss_coeff_hatx_bytes + r0.stats.pfss_trunc_hatx_bytes;
  size_t masked_hatx_bytes = m0.stats.pfss_coeff_hatx_bytes + m0.stats.pfss_trunc_hatx_bytes;
  int active = 0;
  for (int v : valid) active += v;
  double frac = (rows * cols_masked) > 0 ? static_cast<double>(active) / static_cast<double>(rows * cols_masked) : 1.0;
  double allowed = frac + 0.15;  // small overhead slack
  if (masked_hatx_bytes > static_cast<size_t>(dense_hatx_bytes * allowed + 1.0)) {
    std::cerr << "Masked PFSS hatx bytes exceed budget (masked=" << masked_hatx_bytes
              << ", dense=" << dense_hatx_bytes << ", frac=" << frac << ")\n";
    return 1;
  }
  if (m0.planner_stats.coeff_jobs > r0.planner_stats.coeff_jobs ||
      m0.planner_stats.trunc_jobs > r0.planner_stats.trunc_jobs) {
    std::cerr << "Planner stats did not shrink under valid_lens (coeff " << m0.planner_stats.coeff_jobs
              << " vs " << r0.planner_stats.coeff_jobs << ", trunc " << m0.planner_stats.trunc_jobs
              << " vs " << r0.planner_stats.trunc_jobs << ")\n";
    return 1;
  }
  if (m0.planner_stats.coeff_flushes > 1 || m0.planner_stats.trunc_flushes > 1) {
    std::cerr << "Masked softmax expected at most one planner flush, got coeff="
              << m0.planner_stats.coeff_flushes << " trunc=" << m0.planner_stats.trunc_flushes << "\n";
    return 1;
  }

  // Super-plan prototype: run two softmax tasks back-to-back with shared planner/flush.
  {
    std::vector<uint64_t> probs_a(rows * cols, 0);
    std::vector<uint64_t> probs_b(rows * cols, 0);
    runtime::PhaseExecutor pe;
    runtime::PhaseResources R{};
    R.party = 0;
    runtime::ProtoChanFromNet pch(c0);
    R.pfss_backend = &backend;
    R.pfss_chan = &pch;
    R.net_chan = &c0;
    R.pfss_coeff = &pe.pfss_coeff_batch();
    R.pfss_trunc = &pe.pfss_trunc_batch();
    R.opens = &pe.open_collector();
    runtime::PfssPhasePlanner planner;
    planner.bind(R.pfss_coeff, R.pfss_trunc);
    R.pfss_planner = &planner;
    nn::SoftmaxPlan plan = {};
    plan.frac_bits = fb;
    plan.rows = rows;
    plan.cols = cols;
    plan.nexp = gates::make_nexp_cubic_bundle(nexp_mat, fb);
    plan.recip = gates::make_recip_bundle(recip_mat);
    plan.prob_trunc = prob_choice;
    plan.row_triples = &rb_p0;
    compiler::RangeInterval prob_range2;
    prob_range2.lo = 0;
    prob_range2.hi = static_cast<int64_t>(1) << fb;
    prob_range2.is_signed = false;
    plan.prob_range = prob_range2;
    pe.begin_phase(runtime::PhaseExecutor::Phase::kSoftmax);
    pe.add_task(std::make_unique<nn::SoftmaxBlockTask>(
        plan,
        std::span<const uint64_t>(t_share.data(), t_share.size()),
        std::span<uint64_t>(probs_a.data(), probs_a.size())));
    // Append second logical phase without clearing tasks to force one PFSS flush.
    pe.begin_phase(runtime::PhaseExecutor::Phase::kSoftmax, /*clear_tasks=*/false);
    pe.add_task(std::make_unique<nn::SoftmaxBlockTask>(
        plan,
        std::span<const uint64_t>(t_share.data(), t_share.size()),
        std::span<uint64_t>(probs_b.data(), probs_b.size())));
    pe.run(R);
    if (planner.stats().coeff_flushes > 1 || planner.stats().trunc_flushes > 1) {
      std::cerr << "Super-plan expected single PFSS flush, got coeff=" << planner.stats().coeff_flushes
                << " trunc=" << planner.stats().trunc_flushes << "\n";
      return 1;
    }
  }

  // Staged executor demo: two trunc tasks share one PFSS flush.
  {
    proto::ReferenceBackend ref_backend;
    compiler::GateParams p;
    p.kind = compiler::GateKind::GapARS;
    p.frac_bits = fb;
    std::mt19937_64 rng(999);
    auto bundle = compiler::lower_truncation_gate(ref_backend, rng, p, rows * cols);
    std::vector<uint64_t> in_a(rows * cols, 0), in_b(rows * cols, 0);
    for (size_t i = 0; i < in_a.size(); ++i) {
      in_a[i] = t_share[i];
      in_b[i] = t_share[i];
    }
    std::vector<uint64_t> out_a(in_a.size(), 0), out_b(in_b.size(), 0);
    runtime::StagedExecutor se0, se1;
    runtime::PhaseResources RA, RB;
    RA.party = 0;
    RB.party = 1;
    runtime::ProtoChanFromNet pchA(c0), pchB(c1);
    RA.pfss_backend = &ref_backend;
    RB.pfss_backend = &ref_backend;
    RA.pfss_chan = &pchA;
    RB.pfss_chan = &pchB;
    RA.net_chan = &c0;
    RB.net_chan = &c1;
    runtime::PfssSuperBatch sb_coeffA, sb_coeffB;
    runtime::PfssSuperBatch sb_truncA, sb_truncB;
    RA.pfss_coeff = &sb_coeffA;
    RA.pfss_trunc = &sb_truncA;
    RB.pfss_coeff = &sb_coeffB;
    RB.pfss_trunc = &sb_truncB;
    runtime::OpenCollector opensA, opensB;
    RA.opens = &opensA;
    RB.opens = &opensB;

    se0.add_task(std::make_unique<runtime::TruncTask>(
        &bundle,
        std::span<const uint64_t>(in_a.data(), in_a.size()),
        std::span<uint64_t>(out_a.data(), out_a.size())));
    se0.add_task(std::make_unique<runtime::TruncTask>(
        &bundle,
        std::span<const uint64_t>(in_b.data(), in_b.size()),
        std::span<uint64_t>(out_b.data(), out_b.size())));
    se1.add_task(std::make_unique<runtime::TruncTask>(
        &bundle,
        std::span<const uint64_t>(in_a.data(), in_a.size()),
        std::span<uint64_t>(out_a.data(), out_a.size())));
    se1.add_task(std::make_unique<runtime::TruncTask>(
        &bundle,
        std::span<const uint64_t>(in_b.data(), in_b.size()),
        std::span<uint64_t>(out_b.data(), out_b.size())));

    se0.run(RA, ref_backend, pchA);
    se1.run(RB, ref_backend, pchB);
    for (size_t i = 0; i < out_a.size(); ++i) {
      uint64_t plain = out_a[i] + out_b[i];
      uint64_t expect = t_share[i] >> fb;
      if (plain != expect) {
        std::cerr << "Staged trunc mismatch at " << i << ": got " << plain << " expect " << expect << "\n";
        return 1;
      }
    }
  }

  std::cout << "Softmax task test passed\n";
  return 0;
  } catch (const std::exception& e) {
    std::cerr << "Softmax task test failed: " << e.what() << std::endl;
    return 1;
  }
}
