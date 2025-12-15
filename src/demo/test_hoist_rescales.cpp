#include <cassert>
#include <cstddef>

#include "compiler/layer_graph.hpp"

static size_t count_rescales(const compiler::LayerGraph& g) {
  size_t n = 0;
  for (const auto& op : g.ops()) {
    if (op.kind == compiler::OpKind::kRescale) n++;
  }
  return n;
}

int main() {
  compiler::LayerGraph g;

  compiler::Scale hi;
  hi.n_bits = 64;
  hi.frac_bits = 16;
  hi.is_signed = true;

  compiler::Scale lo = hi;
  lo.frac_bits = 8;

  auto x = g.add_tensor(hi, compiler::RangeInterval::whole(true));
  auto y = g.add_tensor(hi, compiler::RangeInterval::whole(true));

  compiler::RescaleAttrs r;
  r.from_frac = hi.frac_bits;
  r.to_frac = lo.frac_bits;
  r.signed_ars = true;

  auto x_lo = g.add_rescale(x, r, lo);
  auto y_lo = g.add_rescale(y, r, lo);
  (void)g.add_add(x_lo, y_lo, lo);

  g.propagate_ranges();
  const size_t before = count_rescales(g);
  g.hoist_rescales();
  g.propagate_ranges();
  const size_t after = count_rescales(g);

  // The constructed pattern should trigger an add/sub hoist and insert a new
  // trailing rescale (in addition to the original per-input rescales).
  assert(after > before);

  // Every rescale op's output tensor must reflect its target frac_bits.
  const auto& ops = g.ops();
  const auto& tensors = g.tensors();
  for (const auto& op : ops) {
    if (op.kind != compiler::OpKind::kRescale) continue;
    assert(op.rescale.to_frac != 0);
    assert(!op.outputs.empty());
    int out_tid = op.outputs[0];
    assert(out_tid >= 0 && static_cast<size_t>(out_tid) < tensors.size());
    assert(tensors[static_cast<size_t>(out_tid)].scale.frac_bits == op.rescale.to_frac);
  }

  return 0;
}

