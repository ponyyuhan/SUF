#include "nn/layernorm_graph.hpp"

#include <stdexcept>

namespace nn {

SecretTensor build_layernorm_graph(LayerContext& ctx,
                                   const SecretTensor& x,
                                   const TensorView<int64_t>* gamma_public,
                                   const TensorView<int64_t>* beta_public,
                                   int length,
                                   int frac_bits) {
  if (!x.valid()) {
    throw std::runtime_error("build_layernorm_graph: input tensor invalid");
  }
  compiler::Scale qf = make_scale(frac_bits, true);
  // Mean
  int mean_tid = ctx.graph.add_mean(x.tid, length, qf);
  // Variance
  int var_tid = ctx.graph.add_var(x.tid, mean_tid, length, frac_bits, make_scale(2 * frac_bits, true));
  // Rsqrt (1/sqrt(var+eps)): treated as external task; scale stays Qf.
  int rsqrt_tid = ctx.graph.add_rsqrt(var_tid, frac_bits, qf);
  // Affine: (x - mu) * r (rsqrt) with optional gamma/beta (public).
  int gamma_tid = -1;
  int beta_tid = -1;
  compiler::RangeInterval gamma_range = compiler::RangeInterval::whole(true);
  compiler::RangeInterval beta_range = compiler::RangeInterval::whole(true);
  if (gamma_public) {
    gamma_range = range_from_public_weights(*gamma_public);
    SecretTensor g = make_secret_tensor(&ctx,
                                        TensorView<uint64_t>(nullptr, gamma_public->shape[0]),
                                        qf,
                                        gamma_range);
    gamma_tid = g.tid;
  }
  if (beta_public) {
    beta_range = range_from_public_weights(*beta_public);
    SecretTensor b = make_secret_tensor(&ctx,
                                        TensorView<uint64_t>(nullptr, beta_public->shape[0]),
                                        qf,
                                        beta_range);
    beta_tid = b.tid;
  }
  int affine_tid = ctx.graph.add_affine(rsqrt_tid, gamma_tid, beta_tid, frac_bits, qf);
  SecretTensor out;
  out.tid = affine_tid;
  out.scale = qf;
  out.range = compiler::RangeInterval::whole(true);
  out.ctx = &ctx;
  return out;
}

}  // namespace nn
