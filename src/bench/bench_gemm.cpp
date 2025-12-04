#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "nn/matmul_publicW.hpp"
#include "nn/matmul_beaver.hpp"
#include "mpc/net.hpp"

using namespace nn;

struct NullChan : net::Chan {
  void send_u64(uint64_t) override {}
  uint64_t recv_u64() override { return 0; }
};

static int64_t to_signed(uint64_t v) { return static_cast<int64_t>(v); }
static uint64_t to_ring(int64_t v) { return static_cast<uint64_t>(v); }

int main(int argc, char** argv) {
  size_t M = 64, K = 64, N = 64;
  bool use_beaver = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--M=", 0) == 0) M = std::strtoull(arg.c_str() + 4, nullptr, 10);
    else if (arg.rfind("--K=", 0) == 0) K = std::strtoull(arg.c_str() + 4, nullptr, 10);
    else if (arg.rfind("--N=", 0) == 0) N = std::strtoull(arg.c_str() + 4, nullptr, 10);
    else if (arg == "--beaver") use_beaver = true;
  }
  int frac_bits = 8;
  std::mt19937_64 rng(42);
  std::vector<int64_t> W(K * N);
  for (auto& w : W) w = static_cast<int64_t>(rng() % 16);

  size_t batches = 16;
  std::vector<uint64_t> X0(M * K), X1(M * K), Y0(M * N), Y1(M * N);
  for (auto& x : X0) x = rng();
  for (size_t i = 0; i < M * K; ++i) X1[i] = to_ring(-to_signed(X0[i]));

  auto start = std::chrono::steady_clock::now();
  if (!use_beaver) {
    MatmulParams p{frac_bits, false, nullptr};
    for (size_t b = 0; b < batches; ++b) {
      matmul_publicW(view2(X0.data(), M, K), view2(W.data(), K, N), view2(Y0.data(), M, N), p);
      matmul_publicW(view2(X1.data(), M, K), view2(W.data(), K, N), view2(Y1.data(), M, N), p);
    }
  } else {
    std::vector<uint64_t> W0(W.size()), W1(W.size());
    for (size_t i = 0; i < W.size(); ++i) {
      uint64_t r = rng();
      W0[i] = r;
      W1[i] = to_ring(W[i] - static_cast<int64_t>(r));
    }
    auto [t0, t1] = dealer_gen_matmul_triple(M, K, N, frac_bits, rng);
    proto::TapeWriter w0, w1;
    write_matmul_triple(w0, t0);
    write_matmul_triple(w1, t1);
    proto::TapeReader r0(w0.data()), r1(w1.data());
    NullChan ch;
    MatmulBeaverParams p{frac_bits};
    for (size_t b = 0; b < batches; ++b) {
      matmul_beaver(p, 0, ch, view2(X0.data(), M, K), view2(W0.data(), K, N), view2(Y0.data(), M, N), r0);
      matmul_beaver(p, 0, ch, view2(X1.data(), M, K), view2(W1.data(), K, N), view2(Y1.data(), M, N), r0);
    }
  }
  auto end = std::chrono::steady_clock::now();
  double secs = std::chrono::duration<double>(end - start).count();
  std::cout << "GEMM " << (use_beaver ? "beaver" : "publicW")
            << " M=" << M << " K=" << K << " N=" << N
            << " batches=" << batches
            << " time=" << secs << "s\n";
  return 0;
}
