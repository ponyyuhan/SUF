#pragma once

#include <cstdint>
#include <utility>
#include <vector>
#include "core/ring.hpp"
#include "mpc/beaver.hpp"
#include "mpc/net.hpp"
#include "nn/tensor_view.hpp"
#include "proto/tape.hpp"

namespace nn {

struct MatmulBeaverTriple {
  size_t M = 0, K = 0, N = 0;
  bool w_transposed = false;
  std::vector<uint64_t> A_share;
  std::vector<uint64_t> B_share;
  std::vector<uint64_t> C_share;
};

struct MatmulBeaverParams {
  int frac_bits = 0;
  bool w_transposed = false;
};

std::pair<MatmulBeaverTriple, MatmulBeaverTriple> dealer_gen_matmul_triple(
    size_t M,
    size_t K,
    size_t N,
    int frac_bits,
    std::mt19937_64& rng,
    bool w_transposed = false);

void write_matmul_triple(proto::TapeWriter& w, const MatmulBeaverTriple& t);
MatmulBeaverTriple read_matmul_triple(proto::TapeReader& r);

void matmul_beaver(const MatmulBeaverParams& params,
                   int party,
                   net::Chan& ch,
                   const TensorView<uint64_t>& X_share,
                   const TensorView<uint64_t>& W_share,
                   TensorView<uint64_t> Y_share,
                   proto::TapeReader& triple_reader);

}  // namespace nn

