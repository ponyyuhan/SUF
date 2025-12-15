#include <cassert>
#include <cstdint>
#include <vector>

#include "compiler/layer_graph.hpp"

int main() {
  compiler::LayerGraph g;

  compiler::Scale q16;
  q16.n_bits = 64;
  q16.frac_bits = 16;
  q16.is_signed = true;

  // Input tensor (unknown at entry), then clamp to a proof-grade range.
  auto x0 = g.add_tensor(q16, compiler::RangeInterval::whole(true));
  compiler::RangeInterval clamp_r;
  clamp_r.is_signed = true;
  clamp_r.lo = -(1ll << q16.frac_bits);  // -1.0 in Q16
  clamp_r.hi = (1ll << q16.frac_bits);   // +1.0 in Q16
  auto x = g.add_clamp(x0, clamp_r, q16);

  // Matmul with public weights: the weight bound should be treated as Proof.
  compiler::MatmulAttrs mm;
  mm.K = 4;
  mm.w_range.is_signed = true;
  mm.w_range.lo = -(1ll << 10);
  mm.w_range.hi = (1ll << 10);
  mm.row_l1_max = 0;  // force the w_range path
  auto acc = g.add_matmul_beaver(x, mm, q16, compiler::RangeInterval::whole(true));

  // Add a public bias vector.
  std::vector<int64_t> bias_q16 = {
      static_cast<int64_t>(0),
      static_cast<int64_t>(1ll << q16.frac_bits),
      static_cast<int64_t>(-(1ll << q16.frac_bits)),
      static_cast<int64_t>(0),
  };
  auto biased = g.add_bias(acc, bias_q16, q16);

  // LayerNorm-style proof chain: mean/var/rsqrt/affine with public gamma/beta.
  int len = 4;
  auto mean = g.add_mean(biased, len, q16);
  auto var = g.add_var(biased, mean, len, /*frac_bits=*/q16.frac_bits, q16);
  auto inv_std = g.add_rsqrt(var, /*frac_bits=*/q16.frac_bits, q16);

  compiler::RangeInterval one_q16;
  one_q16.is_signed = true;
  one_q16.lo = one_q16.hi = (1ll << q16.frac_bits);
  auto gamma = g.add_public_tensor(q16, one_q16);

  compiler::RangeInterval zero_q16;
  zero_q16.is_signed = true;
  zero_q16.lo = zero_q16.hi = 0;
  auto beta = g.add_public_tensor(q16, zero_q16);

  auto affine = g.add_affine(inv_std, gamma, beta, /*frac_bits=*/q16.frac_bits, q16);

  g.propagate_ranges();

  auto expect_proof = [&](int tid, bool need_gap) {
    const auto& t = g.tensors()[static_cast<size_t>(tid)];
    assert(t.abs.kind == compiler::RangeKind::Proof);
    if (need_gap) assert(t.gap.has_value());
    assert(t.mask_abs != 0);
  };

  expect_proof(x, /*need_gap=*/true);
  expect_proof(acc, /*need_gap=*/true);
  expect_proof(biased, /*need_gap=*/true);
  expect_proof(mean, /*need_gap=*/true);
  expect_proof(var, /*need_gap=*/true);
  expect_proof(inv_std, /*need_gap=*/true);
  expect_proof(gamma, /*need_gap=*/false);  // public tensors do not track gap certs
  expect_proof(beta, /*need_gap=*/false);
  expect_proof(affine, /*need_gap=*/true);

  // The gap cert should hold for these small bounds.
  assert(compiler::can_gapars(*g.tensors()[static_cast<size_t>(affine)].gap));

  return 0;
}
