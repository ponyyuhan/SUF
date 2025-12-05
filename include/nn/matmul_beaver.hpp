#pragma once

#include <cstdint>
#include <utility>
#include <vector>
#include <random>
#include "core/ring.hpp"
#include "compiler/range_analysis.hpp"
#include "compiler/matmul_truncation.hpp"
#include "compiler/truncation_lowering.hpp"
#include "proto/pfss_backend_batch.hpp"
#include "mpc/beaver.hpp"
#include "mpc/net.hpp"
#include "nn/tensor_view.hpp"
#include "proto/tape.hpp"

namespace runtime {
class PfssSuperBatch;
}

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
  // Optional: use composite truncation instead of local shift.
  proto::PfssBackendBatch* trunc_backend = nullptr;
  // Either a prebuilt plan or a raw bundle; plan carries GateKind/range metadata.
  const compiler::MatmulTruncationPlan* trunc_plan = nullptr;
  const compiler::TruncationLoweringResult* trunc_bundle = nullptr;
  bool require_truncation = false;  // forbid local-shift fallback when true
  // Optional range hints to allow GapARS selection during auto-plan.
  compiler::RangeInterval x_range = compiler::RangeInterval::whole(true);
  compiler::RangeInterval w_range = compiler::RangeInterval::whole(true);
  // Optional PFSS batching surface; when set we enqueue truncation instead of
  // evaluating immediately so callers can flush across a phase.
  runtime::PfssSuperBatch* pfss_batch = nullptr;
  bool defer_trunc_finalize = false;  // true: leave results enqueued for caller to flush.
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
