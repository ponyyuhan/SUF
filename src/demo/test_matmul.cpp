#include <cassert>
#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <thread>
#include <vector>

#include "compiler/layer_graph.hpp"
#include "compiler/matmul_truncation.hpp"
#include "compiler/range_analysis.hpp"
#include "compiler/truncation_pass_runner.hpp"
#include "nn/matmul_beaver.hpp"
#include "nn/matmul_publicW.hpp"
#include "mpc/net.hpp"
#include "proto/reference_backend.hpp"
#include "runtime/pfss_superbatch.hpp"

using namespace nn;

struct LocalChan : net::Chan {
  struct Shared {
    std::mutex m;
    std::condition_variable cv;
    std::queue<uint64_t> q0to1, q1to0;
  };
  Shared* s = nullptr;
  bool is0 = false;
  LocalChan() = default;
  LocalChan(Shared* sh, bool p) : s(sh), is0(p) {}
  void send_u64(uint64_t v) override {
    std::unique_lock<std::mutex> lk(s->m);
    auto& q = is0 ? s->q0to1 : s->q1to0;
    q.push(v);
    s->cv.notify_all();
  }
  uint64_t recv_u64() override {
    std::unique_lock<std::mutex> lk(s->m);
    auto& q = is0 ? s->q1to0 : s->q0to1;
    s->cv.wait(lk, [&]{ return !q.empty(); });
    uint64_t v = q.front(); q.pop(); return v;
  }
};

static inline int64_t to_signed(uint64_t v) { return static_cast<int64_t>(v); }
static inline uint64_t to_ring(int64_t v) { return static_cast<uint64_t>(v); }

static std::vector<int64_t> matmul_ref(const std::vector<int64_t>& X,
                                       const std::vector<int64_t>& W,
                                       size_t B,
                                       size_t M,
                                       size_t K,
                                       size_t N,
                                       int frac_bits,
                                       bool w_transposed) {
  std::vector<int64_t> out(B * M * N, 0);
  for (size_t b = 0; b < B; ++b) {
    for (size_t m = 0; m < M; ++m) {
      for (size_t n = 0; n < N; ++n) {
        __int128 acc = 0;
        for (size_t k = 0; k < K; ++k) {
          size_t xidx = (b * M + m) * K + k;
          size_t widx = w_transposed ? (n * K + k) : (k * N + n);
          acc += static_cast<__int128>(X[xidx]) * static_cast<__int128>(W[widx]);
        }
        out[(b * M + m) * N + n] = static_cast<int64_t>(acc >> frac_bits);
      }
    }
  }
  return out;
}

static void test_matmul_public_basic() {
  std::mt19937_64 rng(1);
  const int frac_bits = 8;
  const uint64_t mask = (uint64_t(1) << frac_bits) - 1;
  size_t B = 2, M = 3, K = 4, N = 5;
  std::vector<int64_t> X(B * M * K), W(K * N);
  for (auto& v : X) v = static_cast<int64_t>(static_cast<int16_t>(rng()));
  for (auto& v : W) v = static_cast<int64_t>(static_cast<int16_t>(rng()));

  auto ref = matmul_ref(X, W, B, M, K, N, frac_bits, /*w_transposed=*/false);

  auto split = [&](const std::vector<int64_t>& src) {
    std::vector<uint64_t> a(src.size()), b(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
      uint64_t r = rng() & ~mask;  // avoid fractional-bit carry between shares
      a[i] = r;
      b[i] = to_ring(src[i] - static_cast<int64_t>(r));
    }
    return std::make_pair(std::move(a), std::move(b));
  };
  auto [X0, X1] = split(X);

  MatmulParams mp{frac_bits, false, nullptr};
  std::vector<uint64_t> Y0(B * M * N), Y1(B * M * N);
  matmul_publicW(view3(X0.data(), B, M, K),
                 view2(W.data(), K, N),
                 view3(Y0.data(), B, M, N),
                 mp);
  matmul_publicW(view3(X1.data(), B, M, K),
                 view2(W.data(), K, N),
                 view3(Y1.data(), B, M, N),
                 mp);

  const int64_t tol = 1;
  int64_t max_diff = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    int64_t got = to_signed(Y0[i]) + to_signed(Y1[i]);
    int64_t diff = std::llabs(got - ref[i]);
    if (diff > max_diff) max_diff = diff;
  }
  std::cout << "beaver max diff " << max_diff << std::endl;
  assert(max_diff <= tol);
}

static void test_matmul_beaver_case(size_t B,
                                    size_t M,
                                    size_t K,
                                    size_t N,
                                    bool w_transposed,
                                    int frac_bits,
                                    uint64_t seed,
                                    bool use_truncation = false) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int64_t> dist(-200, 200);
  const uint64_t mask = (uint64_t(1) << frac_bits) - 1;
  std::vector<int64_t> X(B * M * K), W(K * N);
  if (w_transposed) W.resize(N * K);
  for (auto& v : X) v = dist(rng);
  for (auto& v : W) v = dist(rng);

  auto ref = matmul_ref(X, W, B, M, K, N, frac_bits, w_transposed);

  auto split = [&](const std::vector<int64_t>& src) {
    std::vector<uint64_t> a(src.size()), b(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
      uint64_t r = rng() & ~mask;
      a[i] = r;
      b[i] = to_ring(src[i] - static_cast<int64_t>(r));
    }
    return std::make_pair(std::move(a), std::move(b));
  };
  auto [X0, X1] = split(X);
  auto [W0, W1] = split(W);

  proto::TapeWriter tw0, tw1;
  for (size_t b = 0; b < B; ++b) {
    auto [t0, t1] = dealer_gen_matmul_triple(M, K, N, frac_bits, rng, w_transposed);
    write_matmul_triple(tw0, t0);
    write_matmul_triple(tw1, t1);
  }
  proto::TapeReader tr0(tw0.data()), tr1(tw1.data());

  LocalChan::Shared sh;
  LocalChan c0(&sh, true), c1(&sh, false);
  MatmulBeaverParams params;
  params.frac_bits = frac_bits;
  params.w_transposed = w_transposed;
  proto::ReferenceBackend trunc_backend;
  compiler::TruncationPassContext pass_ctx(trunc_backend, seed + 12345);
  if (use_truncation) {
    auto range_of = [](const std::vector<int64_t>& vec) {
      int64_t lo = vec.empty() ? 0 : vec[0];
      int64_t hi = vec.empty() ? 0 : vec[0];
      for (auto v : vec) {
        lo = std::min(lo, v);
        hi = std::max(hi, v);
      }
      compiler::RangeInterval r;
      r.lo = lo;
      r.hi = hi;
      r.is_signed = true;
      return r;
    };
    auto xr = range_of(X);
    auto wr = range_of(W);
    params.trunc_backend = &trunc_backend;
    compiler::Scale input_scale{64, frac_bits, true};
    compiler::LayerGraph g;
    int x_id = g.add_tensor(input_scale, xr);
    compiler::MatmulAttrs matmul_attrs;
    matmul_attrs.params = &params;
    matmul_attrs.M = M;
    matmul_attrs.K = K;
    matmul_attrs.N = N;
    matmul_attrs.w_transposed = w_transposed;
    matmul_attrs.frac_bits = frac_bits;
    matmul_attrs.x_range = xr;
    matmul_attrs.w_range = wr;
    compiler::Scale accum_scale{64, 2 * frac_bits, true};
    int matmul_out = g.add_matmul_beaver(x_id, matmul_attrs, accum_scale);
    compiler::RescaleAttrs rattrs;
    rattrs.matmul_op = g.ops().size() - 1;
    rattrs.from_frac = accum_scale.frac_bits;
    rattrs.to_frac = frac_bits;
    compiler::Scale out_scale{64, frac_bits, true};
    g.add_rescale(matmul_out, rattrs, out_scale);
    g.lower_truncations(pass_ctx);
  }

  MatmulBeaverParams params0 = params;
  MatmulBeaverParams params1 = params;
  runtime::PfssSuperBatch batch0, batch1;
  if (use_truncation) {
    params0.pfss_batch = &batch0;
    params1.pfss_batch = &batch1;
    params0.defer_trunc_finalize = true;
    params1.defer_trunc_finalize = true;
    params0.require_truncation = true;
    params1.require_truncation = true;
  }

  std::vector<uint64_t> Y0(B * M * N), Y1(B * M * N);
  std::thread th1([&] {
    matmul_beaver(params1,
                  1,
                  c1,
                  view3(X1.data(), B, M, K),
                  w_transposed ? view2(W1.data(), N, K) : view2(W1.data(), K, N),
                  view3(Y1.data(), B, M, N),
                  tr1);
  });
  matmul_beaver(params0,
                0,
                c0,
                view3(X0.data(), B, M, K),
                w_transposed ? view2(W0.data(), N, K) : view2(W0.data(), K, N),
                view3(Y0.data(), B, M, N),
                tr0);
  th1.join();

  if (use_truncation) {
    runtime::ProtoChanFromNet pch0(c0), pch1(c1);
    std::thread flush1([&]{ batch1.flush_and_finalize(1, trunc_backend, pch1); });
    batch0.flush_and_finalize(0, trunc_backend, pch0);
    flush1.join();
  }

  const int64_t tol = 1;
  int64_t max_diff = 0;
  int64_t worst = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    int64_t got = to_signed(Y0[i]) + to_signed(Y1[i]);
    int64_t diff = std::llabs(got - ref[i]);
    if (diff > max_diff) max_diff = diff;
    if (diff > worst) worst = diff;
  }
  std::cout << "beaver max diff " << max_diff << "\n";
  if (worst > tol) {
    std::cerr << "worst diff " << worst << " exceeds tol " << tol << "\n";
  }
  assert(worst <= tol);
}

int main() {
  test_matmul_public_basic();
  test_matmul_beaver_case(/*B=*/1, /*M=*/2, /*K=*/3, /*N=*/4, /*w_transposed=*/false, /*frac_bits=*/8, /*seed=*/7);
  test_matmul_beaver_case(/*B=*/2, /*M=*/2, /*K=*/2, /*N=*/3, /*w_transposed=*/true, /*frac_bits=*/6, /*seed=*/9);
  test_matmul_beaver_case(/*B=*/1, /*M=*/2, /*K=*/3, /*N=*/4, /*w_transposed=*/false, /*frac_bits=*/8, /*seed=*/13, /*use_truncation=*/true);
  std::cout << "Matmul public/private tests passed\n";
  return 0;
}
