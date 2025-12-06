#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "compiler/range_propagation.hpp"
#include "compiler/rescale_pass.hpp"
#include "compiler/truncation_pass_runner.hpp"
#include "nn/matmul_beaver.hpp"

namespace runtime {
class PfssSuperBatch;
}

namespace compiler {

// Lightweight IR for a single layer that makes rescale explicit so the
// truncation pass can plan TR/ARS/GapARS gates. This is intentionally minimal
// and only covers ops we need for truncation planning.

struct Scale {
  int n_bits = 64;
  int frac_bits = 0;
  bool is_signed = true;
};

struct TensorFacts {
  Scale scale;
  RangeInterval range = RangeInterval::whole(true);  // conservative ring-range
  bool gap_cert = false;  // true if range provably satisfies GapARS condition
};

enum class OpKind {
  kMatmulBeaver,
  kAdd,
  kSub,
  kAxpy,
  kMulConst,
  kHadamard,
  kRescale,
  kMean,
  kVar,
  kRsqrt,
  kAffine,
  kBiasAdd  // add public bias vector (range known from weights)
};

struct MatmulAttrs {
  nn::MatmulBeaverParams* params = nullptr;  // patched when lowering
  size_t M = 0;
  size_t K = 0;
  size_t N = 0;
  bool w_transposed = false;
  int64_t row_l1_max = 0;  // optional bound on column L1 norm (public weights)
  RangeInterval x_range = RangeInterval::whole(true);
  RangeInterval w_range = RangeInterval::whole(true);
  int frac_bits = 0;  // target downscale (params->frac_bits)
};

struct RescaleAttrs {
  // Optional link back to the matmul op that produced this value.
  size_t matmul_op = static_cast<size_t>(-1);
  bool signed_ars = true;
  bool prefer_gapars = false;
  int from_frac = 0;
  int to_frac = 0;
};

struct OpNode {
  OpKind kind = OpKind::kAdd;
  std::vector<int> inputs;
  std::vector<int> outputs;
  MatmulAttrs matmul;
  RescaleAttrs rescale;
  int64_t scalar = 0;      // for MulConst / Axpy
  int frac_bits = 0;       // for MulConst / Axpy / Hadamard rescale
  int length = 0;          // for LN mean/var length
  std::vector<int64_t> bias;  // for BiasAdd (public, Qf)
};

class LayerGraph {
 public:
  // Graph builder convenience: create an initial tensor with known scale/range.
  int add_tensor(const Scale& scale, const RangeInterval& r);

  // Matmul that produces an unscaled accumulator. Returns output tensor id.
  int add_matmul_beaver(int x_tensor,
                        const MatmulAttrs& attrs,
                        const Scale& out_scale = Scale{},
                        const RangeInterval& out_range = RangeInterval::whole(true));

  // Insert a rescale op (typically after matmul). Returns output tensor id.
  int add_rescale(int input_tensor,
                  const RescaleAttrs& attrs,
                  const Scale& out_scale,
                  const RangeInterval& out_range = RangeInterval::whole(true));

  // Simple elementwise ops to keep range propagation from collapsing.
  int add_add(int a, int b, const Scale& out_scale);
  int add_sub(int a, int b, const Scale& out_scale);
  int add_mul_const(int x, int64_t c, int frac_bits, const Scale& out_scale);
  int add_axpy(int x, int y, int64_t a, int frac_bits, const Scale& out_scale);
  int add_hadamard(int x, int y, int frac_bits, const Scale& out_scale);
  // LayerNorm primitives
  int add_mean(int x, int length, const Scale& out_scale);
  int add_var(int x, int mean_tensor, int length, int frac_bits, const Scale& out_scale);
  int add_rsqrt(int x, int frac_bits, const Scale& out_scale);
  int add_affine(int x, int gamma, int beta, int frac_bits, const Scale& out_scale);
  int add_bias(int x, const std::vector<int64_t>& bias_qf, const Scale& out_scale);

  // Run forward range propagation.
  void propagate_ranges();

  // Lower all rescale nodes that reference matmul ops into truncation plans.
  // When provided, pfss_batch is threaded into MatmulBeaverParams so runtime
  // truncation can be enqueued rather than executed inline.
  TruncationPassResult lower_truncations(TruncationPassContext& ctx,
                                         ::runtime::PfssSuperBatch* pfss_batch = nullptr);

  const std::vector<TensorFacts>& tensors() const { return tensors_; }
  const std::vector<OpNode>& ops() const { return ops_; }

  // Current op count (useful to tag producer indices).
  size_t current_op_index() const { return ops_.size(); }

  // Conservative hoisting/merging of back-to-back rescale ops to reduce gate count.
  void hoist_rescales();

 private:
  std::vector<TensorFacts> tensors_;
  std::vector<OpNode> ops_;
};

}  // namespace compiler
