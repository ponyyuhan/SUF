#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <thread>
#include <vector>

#include "gates/nexp_composite.hpp"
#include "gates/reciprocal_composite.hpp"
#include "runtime/staged_executor.hpp"
#include "runtime/pfss_superbatch.hpp"
#include "runtime/open_collector.hpp"
#include "proto/pfss_backend.hpp"
#include "proto/reference_backend.hpp"
#include "nn/softmax_block_task_staged.hpp"
#include "nn/softmax_block_task.hpp"
#include "mpc/net.hpp"

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

}  // namespace

// Staged softmax: prepare two tasks, flush once, finalize both.
int main() {
  try {
    const int fb = 16;
    const int rows = 2;
    const int cols = 3;

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
    gates::NExpGateParams nexp_params;
    nexp_params.frac_bits = fb;
    nexp_params.segments = 16;
    auto nexp_mat = gates::dealer_make_nexp_task_material(backend,
                                                          nexp_params,
                                                          rng,
                                                          /*triple_need=*/3 * t_share.size(),
                                                          t_share.size());
    auto recip_mat = gates::dealer_make_recip_task_material(backend, fb, 2, rng, static_cast<size_t>(rows));
    runtime::TruncChoice prob_choice;
    compiler::GateParams gap_p;
    gap_p.kind = compiler::GateKind::GapARS;
    gap_p.frac_bits = fb;
    compiler::GateParams faithful_p = gap_p;
    faithful_p.kind = compiler::GateKind::FaithfulTR;
    prob_choice.gapars = new compiler::TruncationLoweringResult(
        compiler::lower_truncation_gate(backend, rng, gap_p, static_cast<size_t>(rows * cols)));
    prob_choice.faithful = new compiler::TruncationLoweringResult(
        compiler::lower_truncation_gate(backend, rng, faithful_p, static_cast<size_t>(rows * cols)));
    prob_choice.shift_bits = fb;
    prob_choice.signed_value = false;
    auto zero_r_out = [](const compiler::TruncationLoweringResult* b) {
      auto* m = const_cast<compiler::TruncationLoweringResult*>(b);
      if (!m) return;
      std::fill(m->keys.k0.r_out_share.begin(), m->keys.k0.r_out_share.end(), 0ull);
      std::fill(m->keys.k1.r_out_share.begin(), m->keys.k1.r_out_share.end(), 0ull);
    };
    zero_r_out(prob_choice.gapars);
    zero_r_out(prob_choice.faithful);

    nn::SoftmaxPlan plan0;
    plan0.frac_bits = fb;
    plan0.rows = rows;
    plan0.cols = cols;
    plan0.nexp = gates::make_nexp_cubic_bundle(nexp_mat, fb);
    plan0.recip = gates::make_recip_bundle(recip_mat);
    plan0.prob_trunc = prob_choice;
    nn::SoftmaxPlan plan1 = plan0;
    std::mt19937_64 rng_triples(777);
    auto mat = make_row_broadcast_triples(rows, cols, rng_triples);
    RowBroadcastTripleProviderImpl rb0(mat, 0);
    RowBroadcastTripleProviderImpl rb1(mat, 1);
    plan0.row_triples = &rb0;
    plan1.row_triples = &rb1;

    std::vector<uint64_t> out_a(rows * cols, 0), out_b(rows * cols, 0);
    LocalChan::Shared sh;
    LocalChan c0(&sh, true), c1(&sh, false);
    auto run_party = [&](int party,
                         nn::SoftmaxPlan plan,
                         std::span<const uint64_t> t_in,
                         std::vector<uint64_t>& out0,
                         runtime::PfssSuperBatch& coeff,
                         runtime::PfssSuperBatch& trunc,
                         runtime::OpenCollector& opens,
                         LocalChan& net) {
      runtime::StagedExecutor se;
      runtime::PhaseResources R;
      R.party = party;
      runtime::ProtoChanFromNet pch(net);
      R.pfss_backend = &backend;
      R.pfss_chan = &pch;
      R.net_chan = &net;
      R.pfss_coeff = &coeff;
      R.pfss_trunc = &trunc;
      R.opens = &opens;
      se.add_task(std::make_unique<nn::StagedSoftmaxTask>(
          plan,
          t_in,
          std::span<uint64_t>(out0.data(), out0.size())));
      se.run(R, backend, pch);
    };

    runtime::PfssSuperBatch coeffA, coeffB, truncA, truncB;
    runtime::OpenCollector opensA, opensB;
    std::vector<uint64_t> zero_share(t_share.size(), 0ull);
    std::thread t0([&] {
      run_party(0,
                plan0,
                std::span<const uint64_t>(t_share.data(), t_share.size()),
                out_a,
                coeffA,
                truncA,
                opensA,
                c0);
    });
    std::thread t1([&] {
      run_party(1,
                plan1,
                std::span<const uint64_t>(zero_share.data(), zero_share.size()),
                out_b,
                coeffB,
                truncB,
                opensB,
                c1);
    });
    t0.join();
    t1.join();

    // Reconstruct and compare to reference softmax.
    std::vector<int64_t> t_plain(t_share.size(), 0);
    for (size_t i = 0; i < t_share.size(); ++i) t_plain[i] = static_cast<int64_t>(t_share[i]);
    auto spec = gates::make_nexp_spec(gates::NExpGateParams{fb, 16});
    auto recip_spec = gates::make_recip_affine_init_spec(fb, 1024.0);
    std::vector<int64_t> ref(out_a.size(), 0);
    for (int r = 0; r < rows; ++r) {
      std::vector<int64_t> expv(cols, 0);
      int64_t sum = 0;
      for (int c = 0; c < cols; ++c) {
        int64_t v = t_plain[static_cast<size_t>(r * cols + c)];
        expv[c] = gates::ref_nexp_fixed(spec, v);
        sum += expv[c];
      }
      if (sum == 0) sum = 1;
      int64_t inv = gates::ref_reciprocal_fixed(recip_spec, sum, fb, 2);
      for (int c = 0; c < cols; ++c) {
        __int128 prod = static_cast<__int128>(expv[c]) * static_cast<__int128>(inv);
        ref[static_cast<size_t>(r * cols + c)] = static_cast<int64_t>(prod >> fb);
      }
    }

    for (size_t i = 0; i < out_a.size(); ++i) {
      uint64_t plain = out_a[i] + out_b[i];
      if (std::llabs(static_cast<long long>(plain) - ref[i]) > 2) {
        std::cerr << "Staged softmax mismatch at " << i << " got " << plain << " expect " << ref[i] << "\n";
        return 1;
      }
    }
    if (truncA.stats().flushes > 1 || truncB.stats().flushes > 1) {
      std::cerr << "Expected single trunc flush per party, got " << truncA.stats().flushes
                << " and " << truncB.stats().flushes << "\n";
      return 1;
    }

    std::cout << "Staged softmax test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Staged softmax test failed: " << e.what() << std::endl;
    return 1;
  }
}
#include "runtime/phase_tasks.hpp"
