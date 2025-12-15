#include <cassert>
#include <cstdint>

#include "compiler/layer_graph.hpp"

int main() {
  compiler::LayerGraph g;
  compiler::Scale q8;
  q8.n_bits = 64;
  q8.frac_bits = 8;
  q8.is_signed = true;

  auto t0 = g.add_tensor(q8, compiler::RangeInterval::whole(true));
  auto t1 = g.add_tensor(q8, compiler::RangeInterval::whole(true));

  auto t_add = g.add_add(t0, t1, q8);
  auto t_sub = g.add_sub(t0, t1, q8);
  auto t_mul3 = g.add_mul_const(t0, /*c=*/3, /*frac_bits=*/0, q8);
  auto t_mulneg3_div2 = g.add_mul_const(t0, /*c=*/-3, /*frac_bits=*/1, q8);
  auto t_axpy = g.add_axpy(t0, t1, /*a=*/3, /*frac_bits=*/1, q8);

  compiler::RescaleAttrs r;
  r.from_frac = 8;
  r.to_frac = 4;
  compiler::Scale q4 = q8;
  q4.frac_bits = 4;
  auto t_rescale = g.add_rescale(t0, r, q4);

  g.propagate_ranges();

  constexpr uint64_t kMask0 = 1ull << 8;
  assert(g.tensors()[static_cast<size_t>(t0)].mask_abs == kMask0);
  assert(g.tensors()[static_cast<size_t>(t1)].mask_abs == kMask0);

  assert(g.tensors()[static_cast<size_t>(t_add)].mask_abs == 2 * kMask0);
  assert(g.tensors()[static_cast<size_t>(t_sub)].mask_abs == 2 * kMask0);
  assert(g.tensors()[static_cast<size_t>(t_mul3)].mask_abs == 3 * kMask0);
  assert(g.tensors()[static_cast<size_t>(t_mulneg3_div2)].mask_abs == (3 * kMask0 + 1) / 2);
  assert(g.tensors()[static_cast<size_t>(t_axpy)].mask_abs == kMask0 + (3 * kMask0 + 1) / 2);

  // Rescale Q8 -> Q4: ceil(2^8 / 2^4) == 2^4, and we enforce >= default_mask_bound(4).
  assert(g.tensors()[static_cast<size_t>(t_rescale)].mask_abs == (1ull << 4));

  return 0;
}

