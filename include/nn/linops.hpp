#pragma once

#include <cstdint>
#include <vector>
#include "core/ring.hpp"
#include "mpc/arithmetic_mpc.hpp"
#include "mpc/beaver.hpp"
#include "mpc/net.hpp"
#include "nn/tensor_view.hpp"
#include "compiler/layer_graph.hpp"
#include "nn/layer_context.hpp"

namespace nn {

struct LinOpsContext {
  int party = 0;
  net::Chan* ch = nullptr;
  compiler::LayerGraph* graph = nullptr;  // optional builder
  int frac_bits = 0;                      // global model frac bits (e.g., 16)
};

void add(const TensorView<uint64_t>& x,
         const TensorView<uint64_t>& y,
         TensorView<uint64_t> out);

void sub(const TensorView<uint64_t>& x,
         const TensorView<uint64_t>& y,
         TensorView<uint64_t> out);

void mul_const(const TensorView<uint64_t>& x,
               int64_t c,
               int frac_bits,
               TensorView<uint64_t> out,
               LayerContext* ctx = nullptr);

void axpy(const TensorView<uint64_t>& x,
          const TensorView<uint64_t>& y,
          int64_t a,
          int frac_bits,
          TensorView<uint64_t> out,
          LayerContext* ctx = nullptr);

void hadamard(const LinOpsContext& ctx,
              const TensorView<uint64_t>& x,
              const TensorView<uint64_t>& y,
              TensorView<uint64_t> out,
              const std::vector<mpc::BeaverTripleA<core::Z2n<64>>>& triples,
              size_t triple_offset = 0,
              int frac_bits = 0);

void sum_lastdim(const LinOpsContext& ctx,
                 const TensorView<uint64_t>& x,
                 TensorView<uint64_t> out);

void max_lastdim(const LinOpsContext& ctx,
                 const TensorView<uint64_t>& x,
                 TensorView<uint64_t> out);

}  // namespace nn
