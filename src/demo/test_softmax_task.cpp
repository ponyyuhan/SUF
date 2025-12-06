#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <condition_variable>
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
#include "nn/softmax_block_task.hpp"

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
                      net::Chan& chan) {
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

  nn::SoftmaxPlan plan;
  plan.frac_bits = fb;
  plan.rows = rows;
  plan.cols = cols;
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
      r0 = run_party(0, rows, cols, fb, t_share, nexp_mat, recip_mat, prob_choice, &rb_p0, backend, c0);
    } catch (const std::exception& e) {
      fail = true;
      fail_msg = e.what();
    }
  });
  std::thread t1([&] {
    try {
      // party1 shares are zero.
      std::vector<uint64_t> zero_share(t_share.size(), 0ull);
      r1 = run_party(1, rows, cols, fb, zero_share, nexp_mat, recip_mat, prob_choice, &rb_p1, backend, c1);
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

  std::cout << "Softmax task test passed\n";
  return 0;
  } catch (const std::exception& e) {
    std::cerr << "Softmax task test failed: " << e.what() << std::endl;
    return 1;
  }
}
