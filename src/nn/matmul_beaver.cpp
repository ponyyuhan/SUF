#include "nn/matmul_beaver.hpp"

#include <cassert>
#include <cstdint>

namespace nn {

namespace {

inline int64_t to_signed(uint64_t v) { return static_cast<int64_t>(v); }
inline uint64_t to_ring(int64_t v) { return static_cast<uint64_t>(v); }

inline size_t b_offset(size_t k, size_t n, size_t K, size_t N, bool w_transposed) {
  return w_transposed ? (n * K + k) : (k * N + n);
}

}  // namespace

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
      uint64_t r = static_cast<uint64_t>(dist(rng));
      s0[i] = r;
      s1[i] = to_ring(src[i] - static_cast<int64_t>(r));
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

static void open_vector(int party,
                        net::Chan& ch,
                        const std::vector<uint64_t>& diff,
                        std::vector<int64_t>& opened) {
  size_t n = diff.size();
  opened.resize(n);
  if (party == 0) {
    for (auto v : diff) ch.send_u64(v);
    for (size_t i = 0; i < n; ++i) opened[i] = to_signed(diff[i] + ch.recv_u64());
  } else {
    for (size_t i = 0; i < n; ++i) opened[i] = to_signed(diff[i] + ch.recv_u64());
    for (auto v : diff) ch.send_u64(v);
  }
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
  open_vector(party, ch, diff_X, E);
  open_vector(party, ch, diff_W, F);

  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      __int128 acc = static_cast<__int128>(to_signed(t.C_share[m * N + n]));
      for (size_t k = 0; k < K; ++k) {
        size_t bidx = b_offset(k, n, K, N, params.w_transposed);
        acc += static_cast<__int128>(E[m * K + k]) * static_cast<__int128>(to_signed(t.B_share[bidx]));
        acc += static_cast<__int128>(to_signed(t.A_share[m * K + k])) * static_cast<__int128>(F[bidx]);
        if (party == 0) {
          acc += static_cast<__int128>(E[m * K + k]) * static_cast<__int128>(F[bidx]);
        }
      }
      Y_share.data[m * N + n] =
          to_ring(static_cast<int64_t>(acc >> params.frac_bits));
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
      MatmulBeaverTriple t = read_matmul_triple(triple_reader);
      TensorView<uint64_t> Xb = view2(const_cast<uint64_t*>(X_share.data + b * M * K), M, K);
      TensorView<uint64_t> Yb = view2(Y_share.data + b * M * N, M, N);
      TensorView<uint64_t> Wv =
          view2(const_cast<uint64_t*>(W_share.data), W_share.shape[0], W_share.shape[1]);
      matmul_beaver2d(params, party, ch, Xb, Wv, Yb, t);
    }
    return;
  }

  MatmulBeaverTriple t = read_matmul_triple(triple_reader);
  matmul_beaver2d(params, party, ch, X_share, W_share, Y_share, t);
}

}  // namespace nn
