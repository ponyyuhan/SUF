#include "nn/matmul_beaver.hpp"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>

#include "gates/composite_fss.hpp"
#include "runtime/pfss_superbatch.hpp"

namespace nn {

namespace {

inline int64_t to_signed(uint64_t v) { return proto::to_signed(v); }
inline uint64_t to_ring(int64_t v) { return proto::from_signed(v); }
inline size_t b_offset(size_t k, size_t n, size_t K, size_t N, bool w_transposed) {
  return w_transposed ? (n * K + k) : (k * N + n);
}

}  // namespace

static runtime::OpenHandle enqueue_open(runtime::OpenCollector* collector,
                                        const std::vector<uint64_t>& diff,
                                        std::vector<int64_t>& opened) {
  if (!collector) {
    opened.resize(diff.size());
    return {};
  }
  return collector->enqueue(diff);
}

static void materialize_open(int party,
                             net::Chan& ch,
                             runtime::OpenCollector* collector,
                             const runtime::OpenHandle& handle,
                             const std::vector<uint64_t>& diff,
                             std::vector<int64_t>& opened) {
  size_t n = diff.size();
  opened.resize(n);
  if (collector) {
    auto v = collector->view(handle);
    if (v.size() != n) throw std::runtime_error("OpenCollector: length mismatch");
    for (size_t i = 0; i < n; ++i) opened[i] = v[i];
    return;
  }
  if (party == 0) {
    for (auto v : diff) ch.send_u64(v);
    for (size_t i = 0; i < n; ++i) opened[i] = to_signed(diff[i] + ch.recv_u64());
  } else {
    for (size_t i = 0; i < n; ++i) opened[i] = to_signed(diff[i] + ch.recv_u64());
    for (auto v : diff) ch.send_u64(v);
  }
}

static void open_public_hatx(int party,
                             net::Chan& ch,
                             const std::vector<uint64_t>& hatx_share,
                             std::vector<uint64_t>& hatx_public) {
  size_t n = hatx_share.size();
  hatx_public.resize(n);
  if (party == 0) {
    for (auto v : hatx_share) ch.send_u64(v);
    for (size_t i = 0; i < n; ++i) {
      uint64_t other = ch.recv_u64();
      hatx_public[i] = proto::add_mod(hatx_share[i], other);
    }
  } else {
    for (size_t i = 0; i < n; ++i) {
      uint64_t other = ch.recv_u64();
      hatx_public[i] = proto::add_mod(hatx_share[i], other);
    }
    for (auto v : hatx_share) ch.send_u64(v);
  }
}

PreparedMatmulBeaver matmul_beaver_prepare(const MatmulBeaverParams& params,
                                           int party,
                                           net::Chan& ch,
                                           const TensorView<uint64_t>& X_share,
                                           const TensorView<uint64_t>& W_share,
                                           TensorView<uint64_t> Y_share,
                                           proto::TapeReader& triple_reader) {
  PreparedMatmulBeaver prep;
  prep.params = params;
  prep.X_share = X_share;
  prep.W_share = W_share;
  prep.Y_share = Y_share;
  if (X_share.dims != 2) {
    throw std::runtime_error("matmul_beaver_prepare: only 2D supported in prepare path");
  }
  prep.M = X_share.shape[0];
  prep.K = X_share.shape[1];
  prep.N = params.w_transposed ? W_share.shape[0] : W_share.shape[1];
  prep.triple = read_matmul_triple(triple_reader);
  if (prep.triple.M != prep.M || prep.triple.K != prep.K || prep.triple.N != prep.N ||
      prep.triple.w_transposed != params.w_transposed) {
    throw std::runtime_error("matmul_beaver_prepare: triple shape mismatch");
  }

  prep.diff_X.resize(prep.M * prep.K);
  prep.diff_W.resize(prep.K * prep.N);
  for (size_t i = 0; i < prep.M * prep.K; ++i) {
    prep.diff_X[i] = to_ring(to_signed(X_share.data[i]) - to_signed(prep.triple.A_share[i]));
  }
  for (size_t idx = 0; idx < prep.K * prep.N; ++idx) {
    prep.diff_W[idx] = to_ring(to_signed(W_share.data[idx]) - to_signed(prep.triple.B_share[idx]));
  }

  prep.hE = enqueue_open(params.open_collector, prep.diff_X, prep.opened_E);
  prep.hF = enqueue_open(params.open_collector, prep.diff_W, prep.opened_F);
  if (!params.open_collector) {
    materialize_open(party, ch, nullptr, prep.hE, prep.diff_X, prep.opened_E);
    materialize_open(party, ch, nullptr, prep.hF, prep.diff_W, prep.opened_F);
    prep.opened_immediate = true;
  }
  return prep;
}

void matmul_beaver_finalize(PreparedMatmulBeaver& prep,
                            int party,
                            net::Chan& ch) {
  auto params = prep.params;  // copy for convenience
  if (prep.X_share.dims != 2) {
    throw std::runtime_error("matmul_beaver_finalize: only 2D supported");
  }
  if (params.open_collector && !prep.opened_immediate) {
    materialize_open(party, ch, params.open_collector, prep.hE, prep.diff_X, prep.opened_E);
    materialize_open(party, ch, params.open_collector, prep.hF, prep.diff_W, prep.opened_F);
  }
  const auto& E = prep.opened_E;
  const auto& F = prep.opened_F;

  size_t M = prep.M;
  size_t K = prep.K;
  size_t N = prep.N;

  // Compute accumulator shares once.
  size_t total = M * N;
  std::vector<uint64_t> acc_share(total);
  {
    size_t idx = 0;
    for (size_t m = 0; m < M; ++m) {
      for (size_t n = 0; n < N; ++n) {
        __int128 acc = static_cast<__int128>(to_signed(prep.triple.C_share[m * N + n]));
        for (size_t k = 0; k < K; ++k) {
          size_t bidx = b_offset(k, n, K, N, params.w_transposed);
          acc += static_cast<__int128>(E[m * K + k]) *
                 static_cast<__int128>(to_signed(prep.triple.B_share[bidx]));
          acc += static_cast<__int128>(to_signed(prep.triple.A_share[m * K + k])) *
                 static_cast<__int128>(F[bidx]);
          if (party == 0) {
            acc += static_cast<__int128>(E[m * K + k]) *
                   static_cast<__int128>(F[bidx]);
          }
        }
        acc_share[idx++] = to_ring(static_cast<int64_t>(acc));
      }
    }
  }

  bool trunc_requested = (params.trunc_backend != nullptr) ||
                         (params.trunc_bundle != nullptr) ||
                         (params.trunc_plan != nullptr);
  const compiler::TruncationLoweringResult* bundle_ptr = params.trunc_bundle;
  const compiler::MatmulTruncationPlan* plan_ptr = params.trunc_plan;
  if (trunc_requested && !bundle_ptr && plan_ptr) {
    bundle_ptr = &plan_ptr->bundle;
  }

  if (trunc_requested && !params.trunc_backend) {
    throw std::runtime_error("matmul_beaver_finalize: truncation requested but backend null");
  }

  if (trunc_requested && !bundle_ptr) {
    throw std::runtime_error("matmul_beaver_finalize: truncation backend set but no plan provided");
  }

  if (!trunc_requested || params.require_truncation) {
    if (!trunc_requested) {
      throw std::runtime_error("matmul_beaver_finalize: truncation required but no backend/plan provided");
    }
    if (!bundle_ptr) {
      throw std::runtime_error("matmul_beaver_finalize: truncation bundle missing");
    }
  }

  if (plan_ptr && plan_ptr->batch != total) {
    throw std::runtime_error("matmul_beaver_finalize: truncation plan batch mismatch");
  }
  const auto& bundle = *bundle_ptr;
  const auto& key = (party == 0) ? bundle.keys.k0 : bundle.keys.k1;

  // Build masked hatx shares and open them to both parties.
  std::vector<uint64_t> hatx_share(total);
  for (size_t i = 0; i < total; ++i) {
    uint64_t rin = (key.r_in_share_vec.size() > i) ? key.r_in_share_vec[i] : key.r_in_share;
    hatx_share[i] = proto::add_mod(acc_share[i], rin);
  }
  std::vector<uint64_t> hatx_public;
  open_public_hatx(party, ch, hatx_share, hatx_public);

  gates::PostProcHook* hook = (party == 0) ? bundle.hook0.get() : bundle.hook1.get();
  if (!hook) throw std::runtime_error("matmul_beaver_finalize: truncation hook missing");

  runtime::ProtoChanFromNet pch(ch);
  if (params.pfss_batch) {
    params.pfss_batch->enqueue_truncation(bundle, key, *hook, std::move(hatx_public), prep.Y_share);
    if (!params.defer_trunc_finalize) {
      params.pfss_batch->flush_and_finalize(party, *params.trunc_backend, pch);
    }
  } else {
    gates::CompositeBatchInput in{hatx_public.data(), total, nullptr};
    auto out = gates::composite_eval_batch_with_postproc(
        party, *params.trunc_backend, pch, key, bundle.suf, in, *hook);
    uint64_t r_out_share = key.r_out_share.empty() ? 0ull : key.r_out_share[0];
    for (size_t i = 0; i < total; ++i) {
      prep.Y_share.data[i] = proto::sub_mod(out.haty_share[i], r_out_share);
    }
  }
}

std::pair<MatmulBeaverTriple, MatmulBeaverTriple> dealer_gen_matmul_triple(
    size_t M,
    size_t K,
    size_t N,
    int frac_bits,
    std::mt19937_64& rng,
    bool w_transposed) {
  std::uniform_int_distribution<int64_t> dist(static_cast<int64_t>(-512), static_cast<int64_t>(512));
  std::vector<int64_t> A(M * K), B(K * N), C(M * N);

  for (auto& v : A) v = dist(rng);
  for (size_t n = 0; n < N; ++n) {
    for (size_t k = 0; k < K; ++k) {
      B[b_offset(k, n, K, N, w_transposed)] = dist(rng);
    }
  }

  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      __int128 acc = 0;
      for (size_t k = 0; k < K; ++k) {
        size_t bidx = b_offset(k, n, K, N, w_transposed);
        acc += static_cast<__int128>(A[m * K + k]) * static_cast<__int128>(B[bidx]);
      }
      C[m * N + n] = static_cast<int64_t>(acc);
    }
  }

  auto split_vec = [&](const std::vector<int64_t>& src) {
    std::vector<uint64_t> s0(src.size()), s1(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
      uint64_t r = to_ring(dist(rng));
      uint64_t src_ring = to_ring(src[i]);
      s0[i] = r;
      s1[i] = proto::sub_mod(src_ring, r);
    }
    return std::make_pair(std::move(s0), std::move(s1));
  };

  auto [A0, A1] = split_vec(A);
  auto [B0, B1] = split_vec(B);
  auto [C0, C1] = split_vec(C);

  MatmulBeaverTriple t0, t1;
  t0.M = t1.M = M;
  t0.K = t1.K = K;
  t0.N = t1.N = N;
  t0.w_transposed = t1.w_transposed = w_transposed;
  t0.A_share = std::move(A0);
  t1.A_share = std::move(A1);
  t0.B_share = std::move(B0);
  t1.B_share = std::move(B1);
  t0.C_share = std::move(C0);
  t1.C_share = std::move(C1);
  return {std::move(t0), std::move(t1)};
}

void write_matmul_triple(proto::TapeWriter& w, const MatmulBeaverTriple& t) {
  w.write_u64(static_cast<uint64_t>(t.M));
  w.write_u64(static_cast<uint64_t>(t.K));
  w.write_u64(static_cast<uint64_t>(t.N));
  w.write_u64(static_cast<uint64_t>(t.w_transposed ? 1 : 0));
  w.write_u64_vec(t.A_share);
  w.write_u64_vec(t.B_share);
  w.write_u64_vec(t.C_share);
}

MatmulBeaverTriple read_matmul_triple(proto::TapeReader& r) {
  MatmulBeaverTriple t;
  t.M = static_cast<size_t>(r.read_u64());
  t.K = static_cast<size_t>(r.read_u64());
  t.N = static_cast<size_t>(r.read_u64());
  t.w_transposed = r.read_u64() != 0;
  t.A_share = r.read_u64_vec();
  t.B_share = r.read_u64_vec();
  t.C_share = r.read_u64_vec();
  return t;
}

static void matmul_beaver2d(const MatmulBeaverParams& params,
                            int party,
                            net::Chan& ch,
                            const TensorView<uint64_t>& X_share,
                            const TensorView<uint64_t>& W_share,
                            TensorView<uint64_t> Y_share,
                            MatmulBeaverTriple& t) {
  size_t M = X_share.shape[0];
  size_t K = X_share.shape[1];
  size_t N = params.w_transposed ? W_share.shape[0] : W_share.shape[1];
  assert(t.M == M && t.K == K && t.N == N);
  assert(t.w_transposed == params.w_transposed);

  const size_t expected_W0 = params.w_transposed ? N : K;
  const size_t expected_W1 = params.w_transposed ? K : N;
  assert(W_share.shape[0] == expected_W0 && W_share.shape[1] == expected_W1);

  std::vector<uint64_t> diff_X(M * K), diff_W(K * N);
  for (size_t i = 0; i < M * K; ++i) {
    diff_X[i] = to_ring(to_signed(X_share.data[i]) - to_signed(t.A_share[i]));
  }
  for (size_t idx = 0; idx < K * N; ++idx) {
    diff_W[idx] = to_ring(to_signed(W_share.data[idx]) - to_signed(t.B_share[idx]));
  }

  std::vector<int64_t> E, F;
  runtime::OpenHandle hE = enqueue_open(params.open_collector, diff_X, E);
  runtime::OpenHandle hF = enqueue_open(params.open_collector, diff_W, F);
  if (params.open_collector) {
    params.open_collector->flush(party, ch);
  }
  materialize_open(party, ch, params.open_collector, hE, diff_X, E);
  materialize_open(party, ch, params.open_collector, hF, diff_W, F);

  // Compute accumulator shares once.
  size_t total = M * N;
  std::vector<uint64_t> acc_share(total);
  {
    size_t idx = 0;
    for (size_t m = 0; m < M; ++m) {
      for (size_t n = 0; n < N; ++n) {
        __int128 acc = static_cast<__int128>(to_signed(t.C_share[m * N + n]));
        for (size_t k = 0; k < K; ++k) {
          size_t bidx = b_offset(k, n, K, N, params.w_transposed);
          acc += static_cast<__int128>(E[m * K + k]) *
                 static_cast<__int128>(to_signed(t.B_share[bidx]));
          acc += static_cast<__int128>(to_signed(t.A_share[m * K + k])) *
                 static_cast<__int128>(F[bidx]);
          if (party == 0) {
            acc += static_cast<__int128>(E[m * K + k]) *
                   static_cast<__int128>(F[bidx]);
          }
        }
        acc_share[idx++] = to_ring(static_cast<int64_t>(acc));
      }
    }
  }

  bool trunc_requested = (params.trunc_backend != nullptr) ||
                         (params.trunc_bundle != nullptr) ||
                         (params.trunc_plan != nullptr);
  const compiler::TruncationLoweringResult* bundle_ptr = params.trunc_bundle;
  const compiler::MatmulTruncationPlan* plan_ptr = params.trunc_plan;
  if (trunc_requested && !bundle_ptr && plan_ptr) {
    bundle_ptr = &plan_ptr->bundle;
  }

  if (trunc_requested && !params.trunc_backend) {
    throw std::runtime_error("matmul_beaver: truncation requested but backend null");
  }

  if (trunc_requested && !bundle_ptr) {
    throw std::runtime_error("matmul_beaver: truncation backend set but no plan provided");
  }

  if (params.require_truncation && !trunc_requested) {
    throw std::runtime_error("matmul_beaver: truncation required but no backend/plan provided");
  }

  if (!trunc_requested) {
    if (!params.allow_local_shift) {
      throw std::runtime_error("matmul_beaver: local shift fallback disallowed");
    }
    for (size_t i = 0; i < total; ++i) {
      Y_share.data[i] = to_ring(static_cast<int64_t>(to_signed(acc_share[i]) >> params.frac_bits));
    }
    return;
  }

  if (plan_ptr && plan_ptr->batch != total) {
    throw std::runtime_error("matmul_beaver: truncation plan batch mismatch");
  }
  if (!bundle_ptr) {
    throw std::runtime_error("matmul_beaver: truncation bundle missing");
  }
  const auto& bundle = *bundle_ptr;
  const auto& key = (party == 0) ? bundle.keys.k0 : bundle.keys.k1;

  // Build masked hatx shares and open them to both parties.
  std::vector<uint64_t> hatx_share(total);
  for (size_t i = 0; i < total; ++i) {
    hatx_share[i] = proto::add_mod(acc_share[i], key.r_in_share);
  }
  std::vector<uint64_t> hatx_public;
  open_public_hatx(party, ch, hatx_share, hatx_public);

  gates::PostProcHook* hook = (party == 0) ? bundle.hook0.get() : bundle.hook1.get();
  if (!hook) throw std::runtime_error("matmul_beaver: truncation hook missing");

  runtime::ProtoChanFromNet pch(ch);
  if (params.pfss_batch) {
    params.pfss_batch->enqueue_truncation(bundle, key, *hook, std::move(hatx_public), Y_share);
    if (!params.defer_trunc_finalize) {
      params.pfss_batch->flush_and_finalize(party, *params.trunc_backend, pch);
    }
  } else {
    gates::CompositeBatchInput in{hatx_public.data(), total, nullptr};
    auto out = gates::composite_eval_batch_with_postproc(
        party, *params.trunc_backend, pch, key, bundle.suf, in, *hook);
    uint64_t r_out_share = key.r_out_share.empty() ? 0ull : key.r_out_share[0];
    for (size_t i = 0; i < total; ++i) {
      Y_share.data[i] = proto::sub_mod(out.haty_share[i], r_out_share);
    }
  }
}

void matmul_beaver(const MatmulBeaverParams& params,
                   int party,
                   net::Chan& ch,
                   const TensorView<uint64_t>& X_share,
                   const TensorView<uint64_t>& W_share,
                   TensorView<uint64_t> Y_share,
                   proto::TapeReader& triple_reader) {
  if (X_share.dims == 3) {
    size_t B = X_share.shape[0];
    size_t M = X_share.shape[1];
    size_t K = X_share.shape[2];
    size_t N = params.w_transposed ? W_share.shape[0] : W_share.shape[1];
    for (size_t b = 0; b < B; ++b) {
      TensorView<uint64_t> Xb = view2(const_cast<uint64_t*>(X_share.data + b * M * K), M, K);
      TensorView<uint64_t> Yb = view2(Y_share.data + b * M * N, M, N);
      TensorView<uint64_t> Wv =
          view2(const_cast<uint64_t*>(W_share.data), W_share.shape[0], W_share.shape[1]);
      auto prep = matmul_beaver_prepare(params, party, ch, Xb, Wv, Yb, triple_reader);
      if (params.open_collector && !params.defer_open_flush) params.open_collector->flush(party, ch);
      matmul_beaver_finalize(prep, party, ch);
    }
    return;
  }

  auto prep = matmul_beaver_prepare(params, party, ch, X_share, W_share, Y_share, triple_reader);
  if (params.open_collector && !params.defer_open_flush) params.open_collector->flush(party, ch);
  matmul_beaver_finalize(prep, party, ch);
}

}  // namespace nn
