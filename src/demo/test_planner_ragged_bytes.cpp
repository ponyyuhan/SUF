#include <cassert>
#include <cstdint>
#include <random>
#include <thread>
#include <atomic>
#include <string>
#include <sstream>
#include <vector>

#include "proto/reference_backend.hpp"
#include "runtime/phase_executor.hpp"
#include "runtime/pfss_phase_planner.hpp"
#include "runtime/pfss_superbatch.hpp"
#include "nn/softmax_block_task.hpp"
#include "gates/nexp_composite.hpp"
#include "gates/reciprocal_composite.hpp"
#include "mpc/local_chan.hpp"

namespace {

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

  std::vector<uint64_t> B(static_cast<size_t>(rows));
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

}  // namespace

// Regression: ragged (valid_lens) softmax packing should reduce PFSS hatx bytes
// enough to satisfy tight per-layer budgets.
int main() {
  const int rows = 8;
  const int cols = 16;
  const int fb = 8;

  // Ragged: active_elems = sum(valid_lens) < rows*cols.
  const std::vector<int> valid_lens = {16, 12, 8, 4, 16, 2, 1, 0};
  size_t active_elems = 0;
  for (int L : valid_lens) active_elems += static_cast<size_t>(L);
  assert(active_elems < static_cast<size_t>(rows * cols));

  // Tight budgets: should pass with packing, but would fail if using dense rows*cols hatx.
  runtime::PfssLayerPlanner::Limits lim;
  lim.max_phases = 8;
  lim.max_coeff_jobs = 1ull << 16;
  lim.max_trunc_jobs = 1ull << 16;
  lim.max_coeff_hatx_words = 100;
  lim.max_trunc_hatx_words = 256;
  lim.max_coeff_hatx_bytes = lim.max_coeff_hatx_words * sizeof(uint64_t);
  lim.max_trunc_hatx_bytes = lim.max_trunc_hatx_words * sizeof(uint64_t);
  lim.max_coeff_flushes = 1ull << 10;
  lim.max_trunc_flushes = 1ull << 10;

  std::mt19937_64 rng(123);
  std::vector<uint64_t> t_plain(static_cast<size_t>(rows * cols));
  for (auto& v : t_plain) v = static_cast<uint64_t>(rng() % (1ull << fb));
  std::vector<uint64_t> t0 = t_plain;
  std::vector<uint64_t> t1(t_plain.size(), 0ull);

  proto::ReferenceBackend keygen_be;
  gates::NExpGateParams nexp_params{fb, 16};
  auto nexp_mat = gates::dealer_make_nexp_task_material(keygen_be,
                                                        nexp_params,
                                                        rng,
                                                        /*triple_need=*/active_elems * 8 + 64,
                                                        /*batch_N=*/active_elems);
  auto recip_mat = gates::dealer_make_recip_task_material(keygen_be, fb, /*nr_iters=*/1, rng, rows);

  auto nexp_bundle = gates::make_nexp_cubic_bundle(nexp_mat, fb);
  auto recip_bundle = gates::make_recip_bundle(recip_mat);
  runtime::TruncChoice prob_choice;
  prob_choice.gapars = &recip_mat.trunc_fb;
  prob_choice.faithful = prob_choice.gapars;
  prob_choice.shift_bits = fb;
  prob_choice.signed_value = false;

  RowBroadcastTripleMaterial rb_mat = make_row_broadcast_triples(rows, cols, rng);
  RowBroadcastTripleProviderImpl rb0(rb_mat, 0), rb1(rb_mat, 1);

  mpc::net::LocalChan::Shared sh;
  mpc::net::LocalChan c0(&sh, /*is_party0=*/true);
  mpc::net::LocalChan c1(&sh, /*is_party0=*/false);

  std::atomic<bool> fail{false};
  std::string err;
  auto record_err = [&](const std::exception& e, const runtime::PfssLayerPlanner& p0, const runtime::PfssLayerPlanner& p1) {
    fail.store(true, std::memory_order_relaxed);
    std::ostringstream oss;
    oss << e.what();
    const auto t0 = p0.totals();
    const auto t1 = p1.totals();
    oss << " | p0 coeff_hatx=" << t0.coeff_hatx_words << "/" << lim.max_coeff_hatx_words
        << " trunc_hatx=" << t0.trunc_hatx_words << "/" << lim.max_trunc_hatx_words
        << " coeff_bytes=" << t0.coeff_hatx_bytes << "/" << lim.max_coeff_hatx_bytes
        << " trunc_bytes=" << t0.trunc_hatx_bytes << "/" << lim.max_trunc_hatx_bytes
        << "; p1 coeff_hatx=" << t1.coeff_hatx_words << "/" << lim.max_coeff_hatx_words
        << " trunc_hatx=" << t1.trunc_hatx_words << "/" << lim.max_trunc_hatx_words
        << " coeff_bytes=" << t1.coeff_hatx_bytes << "/" << lim.max_coeff_hatx_bytes
        << " trunc_bytes=" << t1.trunc_hatx_bytes << "/" << lim.max_trunc_hatx_bytes;
    err = oss.str();
  };

  auto run_party = [&](int party,
                       mpc::net::LocalChan& net,
                       runtime::PfssLayerPlanner& planner,
                       proto::ReferenceBackend& be,
                       runtime::RowBroadcastTripleProvider* rb,
                       const std::vector<uint64_t>& t_in) {
    runtime::PhaseExecutor pe;
    pe.set_lazy_mode(false);
    pe.set_keep_batches(true);
    pe.begin_phase(runtime::PhaseExecutor::Phase::kSoftmax);

    runtime::PhaseResources R{};
    R.party = party;
    R.net_chan = &net;
    runtime::ProtoChanFromNet pch(net);
    R.pfss_backend = &be;
    R.pfss_chan = &pch;
    R.pfss_coeff = &pe.pfss_coeff_batch();
    R.pfss_trunc = &pe.pfss_trunc_batch();
    R.opens = &pe.open_collector();

    planner.set_limits(lim);
    planner.begin_layer();
    planner.enter_phase();

    nn::SoftmaxPlan plan;
    plan.frac_bits = fb;
    plan.rows = rows;
    plan.cols = cols;
    plan.valid_lens = valid_lens;
    plan.nexp = nexp_bundle;
    plan.recip = recip_bundle;
    plan.prob_trunc = prob_choice;
    plan.row_triples = rb;

    std::vector<uint64_t> out(static_cast<size_t>(rows * cols), 0);
    pe.add_task(std::make_unique<nn::SoftmaxBlockTask>(
        plan,
        std::span<const uint64_t>(t_in.data(), t_in.size()),
        std::span<uint64_t>(out.data(), out.size())));

    pe.run(R);

    // Record remaining PFSS usage (if any) and enforce budgets.
    planner.finalize_layer(party,
                           be,
                           pe.pfss_coeff_batch(),
                           pe.pfss_trunc_batch(),
                           pch);
  };

  runtime::PfssLayerPlanner planner0, planner1;
  proto::ReferenceBackend be0, be1;
  std::thread th([&] {
    try {
      run_party(1, c1, planner1, be1, &rb1, t1);
    } catch (const std::exception& e) {
      record_err(e, planner0, planner1);
    }
  });
  try {
    run_party(0, c0, planner0, be0, &rb0, t0);
  } catch (const std::exception& e) {
    record_err(e, planner0, planner1);
  }
  th.join();

  if (fail.load()) {
    throw std::runtime_error("planner ragged bytes regression: " + err);
  }

  // Basic sanity: both parties should have stayed within budgets.
  assert(planner0.totals().coeff_hatx_bytes <= lim.max_coeff_hatx_bytes);
  assert(planner0.totals().trunc_hatx_bytes <= lim.max_trunc_hatx_bytes);
  assert(planner1.totals().coeff_hatx_bytes <= lim.max_coeff_hatx_bytes);
  assert(planner1.totals().trunc_hatx_bytes <= lim.max_trunc_hatx_bytes);
  return 0;
}

