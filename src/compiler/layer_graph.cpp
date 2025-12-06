#include "compiler/layer_graph.hpp"

#include <stdexcept>

namespace compiler {

int LayerGraph::add_tensor(const Scale& scale, const RangeInterval& r) {
  TensorFacts t;
  t.scale = scale;
  t.range = r;
  tensors_.push_back(t);
  return static_cast<int>(tensors_.size() - 1);
}

int LayerGraph::add_matmul_beaver(int x_tensor,
                                  const MatmulAttrs& attrs,
                                  const Scale& out_scale,
                                  const RangeInterval& out_range) {
  if (x_tensor < 0 || static_cast<size_t>(x_tensor) >= tensors_.size()) {
    throw std::runtime_error("add_matmul_beaver: input tensor id out of range");
  }
  OpNode op;
  op.kind = OpKind::kMatmulBeaver;
  op.inputs = {x_tensor};
  op.outputs = {add_tensor(out_scale, out_range)};
  op.matmul = attrs;
  op.matmul.frac_bits = attrs.frac_bits;
  ops_.push_back(op);
  return op.outputs[0];
}

int LayerGraph::add_rescale(int input_tensor,
                            const RescaleAttrs& attrs,
                            const Scale& out_scale,
                            const RangeInterval& out_range) {
  if (input_tensor < 0 || static_cast<size_t>(input_tensor) >= tensors_.size()) {
    throw std::runtime_error("add_rescale: input tensor id out of range");
  }
  OpNode op;
  op.kind = OpKind::kRescale;
  op.inputs = {input_tensor};
  op.outputs = {add_tensor(out_scale, out_range)};
  op.rescale = attrs;
  ops_.push_back(op);
  return op.outputs[0];
}

int LayerGraph::add_add(int a, int b, const Scale& out_scale) {
  OpNode op;
  op.kind = OpKind::kAdd;
  op.inputs = {a, b};
  op.outputs = {add_tensor(out_scale, RangeInterval::whole(out_scale.is_signed))};
  ops_.push_back(op);
  return op.outputs[0];
}

int LayerGraph::add_sub(int a, int b, const Scale& out_scale) {
  OpNode op;
  op.kind = OpKind::kSub;
  op.inputs = {a, b};
  op.outputs = {add_tensor(out_scale, RangeInterval::whole(out_scale.is_signed))};
  ops_.push_back(op);
  return op.outputs[0];
}

int LayerGraph::add_mul_const(int x, int64_t c, int frac_bits, const Scale& out_scale) {
  OpNode op;
  op.kind = OpKind::kMulConst;
  op.inputs = {x};
  op.outputs = {add_tensor(out_scale, RangeInterval::whole(out_scale.is_signed))};
  op.scalar = c;
  op.frac_bits = frac_bits;
  ops_.push_back(op);
  return op.outputs[0];
}

int LayerGraph::add_axpy(int x, int y, int64_t a, int frac_bits, const Scale& out_scale) {
  OpNode op;
  op.kind = OpKind::kAxpy;
  op.inputs = {x, y};
  op.outputs = {add_tensor(out_scale, RangeInterval::whole(out_scale.is_signed))};
  op.scalar = a;
  op.frac_bits = frac_bits;
  ops_.push_back(op);
  return op.outputs[0];
}

int LayerGraph::add_hadamard(int x, int y, int frac_bits, const Scale& out_scale) {
  OpNode op;
  op.kind = OpKind::kHadamard;
  op.inputs = {x, y};
  op.outputs = {add_tensor(out_scale, RangeInterval::whole(out_scale.is_signed))};
  op.frac_bits = frac_bits;
  ops_.push_back(op);
  return op.outputs[0];
}

int LayerGraph::add_mean(int x, int length, const Scale& out_scale) {
  OpNode op;
  op.kind = OpKind::kMean;
  op.inputs = {x};
  op.outputs = {add_tensor(out_scale, RangeInterval::whole(out_scale.is_signed))};
  op.length = length;
  ops_.push_back(op);
  return op.outputs[0];
}

int LayerGraph::add_var(int x, int mean_tensor, int length, int frac_bits, const Scale& out_scale) {
  OpNode op;
  op.kind = OpKind::kVar;
  op.inputs = {x, mean_tensor};
  op.outputs = {add_tensor(out_scale, RangeInterval::whole(out_scale.is_signed))};
  op.length = length;
  op.frac_bits = frac_bits;
  ops_.push_back(op);
  return op.outputs[0];
}

int LayerGraph::add_rsqrt(int x, int frac_bits, const Scale& out_scale) {
  OpNode op;
  op.kind = OpKind::kRsqrt;
  op.inputs = {x};
  op.outputs = {add_tensor(out_scale, RangeInterval::whole(out_scale.is_signed))};
  op.frac_bits = frac_bits;
  ops_.push_back(op);
  return op.outputs[0];
}

int LayerGraph::add_affine(int x, int gamma, int beta, int frac_bits, const Scale& out_scale) {
  OpNode op;
  op.kind = OpKind::kAffine;
  op.inputs = {x, gamma, beta};
  op.outputs = {add_tensor(out_scale, RangeInterval::whole(out_scale.is_signed))};
  op.frac_bits = frac_bits;
  ops_.push_back(op);
  return op.outputs[0];
}

void LayerGraph::propagate_ranges() {
  for (const auto& op : ops_) {
    auto get_range = [&](int tid) -> const RangeInterval& {
      if (tid < 0 || static_cast<size_t>(tid) >= tensors_.size()) {
        throw std::runtime_error("propagate_ranges: tensor id out of range");
      }
      return tensors_[static_cast<size_t>(tid)].range;
    };
    switch (op.kind) {
      case OpKind::kMatmulBeaver: {
        const auto& attrs = op.matmul;
        const auto& xr = get_range(op.inputs[0]);
        RangeInterval acc;
        if (attrs.row_l1_max > 0) {
          acc = propagate_matmul_accum_rowl1(xr, attrs.row_l1_max);
        } else {
          acc = propagate_matmul_accum(xr, attrs.w_range, attrs.K);
        }
        tensors_[static_cast<size_t>(op.outputs[0])].range = acc;
        break;
      }
      case OpKind::kAdd: {
        tensors_[static_cast<size_t>(op.outputs[0])].range =
            propagate_add(get_range(op.inputs[0]), get_range(op.inputs[1]));
        break;
      }
      case OpKind::kSub: {
        tensors_[static_cast<size_t>(op.outputs[0])].range =
            propagate_sub(get_range(op.inputs[0]), get_range(op.inputs[1]));
        break;
      }
      case OpKind::kMulConst: {
        tensors_[static_cast<size_t>(op.outputs[0])].range =
            propagate_mul_const(get_range(op.inputs[0]), op.scalar, op.frac_bits);
        break;
      }
      case OpKind::kAxpy: {
        tensors_[static_cast<size_t>(op.outputs[0])].range =
            propagate_axpy(get_range(op.inputs[0]), get_range(op.inputs[1]), op.scalar,
                           op.frac_bits);
        break;
      }
      case OpKind::kHadamard: {
        RangeInterval prod = mul_range(get_range(op.inputs[0]), get_range(op.inputs[1]));
        tensors_[static_cast<size_t>(op.outputs[0])].range = shift_down(prod, op.frac_bits);
        break;
      }
      case OpKind::kRescale: {
        // Conservatively shift-down by the producing matmul's frac bits if present,
        // otherwise assume frac_bits stored on input scale.
        RangeInterval r = get_range(op.inputs[0]);
        int shift = (op.rescale.from_frac - op.rescale.to_frac);
        tensors_[static_cast<size_t>(op.outputs[0])].range = shift_down(r, shift);
        break;
      }
      case OpKind::kMean: {
        tensors_[static_cast<size_t>(op.outputs[0])].range = get_range(op.inputs[0]);
        break;
      }
      case OpKind::kVar: {
        RangeInterval xr = get_range(op.inputs[0]);
        RangeInterval mr = get_range(op.inputs[1]);
        RangeInterval diff = propagate_sub(xr, mr);
        RangeInterval sq = propagate_mul(diff, diff, op.frac_bits);
        tensors_[static_cast<size_t>(op.outputs[0])].range = sq;
        break;
      }
      case OpKind::kRsqrt: {
        tensors_[static_cast<size_t>(op.outputs[0])].range = RangeInterval::whole(true);
        break;
      }
      case OpKind::kAffine: {
        tensors_[static_cast<size_t>(op.outputs[0])].range = get_range(op.inputs[0]);
        break;
      }
      default:
        break;
    }
  }
}

void LayerGraph::hoist_rescales() {
  // Map tensor id -> producing op index for quick lookups.
  std::vector<int> producer(tensors_.size(), -1);
  for (size_t i = 0; i < ops_.size(); ++i) {
    for (int t : ops_[i].outputs) {
      if (t >= 0 && static_cast<size_t>(t) < producer.size()) {
        producer[static_cast<size_t>(t)] = static_cast<int>(i);
      }
    }
  }

  for (auto& op : ops_) {
    if (op.kind != OpKind::kRescale || op.inputs.empty()) continue;
    int in_tid = op.inputs[0];
    if (in_tid < 0 || static_cast<size_t>(in_tid) >= producer.size()) continue;
    int parent_idx = producer[static_cast<size_t>(in_tid)];
    if (parent_idx < 0 || static_cast<size_t>(parent_idx) >= ops_.size()) continue;
    auto& parent = ops_[static_cast<size_t>(parent_idx)];
    if (parent.kind != OpKind::kRescale) continue;

    // Fuse parent -> child into a single rescale by updating parent target frac.
    int from_frac = parent.rescale.from_frac ? parent.rescale.from_frac
                                             : tensors_[static_cast<size_t>(parent.inputs[0])].scale.frac_bits;
    int to_frac = op.rescale.to_frac ? op.rescale.to_frac
                                     : tensors_[static_cast<size_t>(op.outputs[0])].scale.frac_bits;
    parent.rescale.to_frac = to_frac;
    parent.rescale.prefer_gapars = parent.rescale.prefer_gapars || op.rescale.prefer_gapars;

    // Update parent output scale/range to reflect the fused target.
    auto& parent_out = tensors_[static_cast<size_t>(parent.outputs[0])];
    parent_out.scale.frac_bits = to_frac;
    parent_out.scale.is_signed = tensors_[static_cast<size_t>(op.outputs[0])].scale.is_signed;
    RangeInterval in_range = tensors_[static_cast<size_t>(parent.inputs[0])].range;
    parent_out.range = shift_down(in_range, from_frac - to_frac);

    // Mark child as a no-op so lowering will skip it.
    op.rescale.from_frac = op.rescale.to_frac;
  }

  // Hoist identical rescales over add/sub when safe: if both inputs are rescale
  // outputs with the same from/to frac and signedness, replace with one rescale
  // after the add/sub.
  for (auto& op : ops_) {
    if (op.kind != OpKind::kAdd && op.kind != OpKind::kSub) continue;
    if (op.inputs.size() != 2) continue;
    int a_tid = op.inputs[0];
    int b_tid = op.inputs[1];
    if (a_tid < 0 || b_tid < 0) continue;
    if (static_cast<size_t>(a_tid) >= producer.size() ||
        static_cast<size_t>(b_tid) >= producer.size()) continue;
    int a_prod_idx = producer[static_cast<size_t>(a_tid)];
    int b_prod_idx = producer[static_cast<size_t>(b_tid)];
    if (a_prod_idx < 0 || b_prod_idx < 0) continue;
    auto& a_prod = ops_[static_cast<size_t>(a_prod_idx)];
    auto& b_prod = ops_[static_cast<size_t>(b_prod_idx)];
    if (a_prod.kind != OpKind::kRescale || b_prod.kind != OpKind::kRescale) continue;
    const auto& ar = a_prod.rescale;
    const auto& br = b_prod.rescale;
    bool same_shift = (ar.from_frac == br.from_frac) && (ar.to_frac == br.to_frac);
    bool same_sign = (tensors_[static_cast<size_t>(a_prod.outputs[0])].scale.is_signed ==
                      tensors_[static_cast<size_t>(b_prod.outputs[0])].scale.is_signed);
    if (!same_shift || !same_sign) continue;

    // Fuse: treat inputs as pre-rescale values and rescale the add/sub result instead.
    int from_frac = ar.from_frac ? ar.from_frac
                                 : tensors_[static_cast<size_t>(a_prod.inputs[0])].scale.frac_bits;
    int to_frac = ar.to_frac ? ar.to_frac
                             : tensors_[static_cast<size_t>(op.outputs[0])].scale.frac_bits;
    op.inputs[0] = a_prod.inputs[0];
    op.inputs[1] = b_prod.inputs[0];
    // Update output scale/range to high-precision before rescale.
    tensors_[static_cast<size_t>(op.outputs[0])].scale.frac_bits = from_frac;
    tensors_[static_cast<size_t>(op.outputs[0])].scale.is_signed =
        tensors_[static_cast<size_t>(a_prod.inputs[0])].scale.is_signed ||
        tensors_[static_cast<size_t>(b_prod.inputs[0])].scale.is_signed;
    RangeInterval ra = tensors_[static_cast<size_t>(op.inputs[0])].range;
    RangeInterval rb = tensors_[static_cast<size_t>(op.inputs[1])].range;
    RangeInterval sum_range = (op.kind == OpKind::kAdd) ? add_range(ra, rb) : sub_range(ra, rb);
    tensors_[static_cast<size_t>(op.outputs[0])].range = sum_range;

    // Insert a new rescale op after the add/sub to restore expected frac_bits.
    OpNode new_rescale;
    new_rescale.kind = OpKind::kRescale;
    new_rescale.inputs = {op.outputs[0]};
    new_rescale.outputs = {add_tensor(tensors_[static_cast<size_t>(op.outputs[0])].scale,
                                      shift_down(sum_range, from_frac - to_frac))};
    new_rescale.rescale.from_frac = from_frac;
    new_rescale.rescale.to_frac = to_frac;
    new_rescale.rescale.signed_ars = ar.signed_ars;
    new_rescale.rescale.prefer_gapars = ar.prefer_gapars || br.prefer_gapars;
    ops_.push_back(new_rescale);
    // Redirect the original consumer output to the new rescale output.
    op.outputs[0] = new_rescale.outputs[0];
  }
}

TruncationPassResult LayerGraph::lower_truncations(TruncationPassContext& ctx,
                                                   ::runtime::PfssSuperBatch* pfss_batch) {
  propagate_ranges();
  TruncationPassConfig cfg;
  for (size_t i = 0; i < ops_.size(); ++i) {
    const auto& op = ops_[i];
    if (op.kind != OpKind::kRescale) continue;
    if (op.rescale.matmul_op == static_cast<size_t>(-1)) continue;
    if (op.rescale.matmul_op >= ops_.size()) continue;
    const auto& matmul = ops_[op.rescale.matmul_op].matmul;
    if (!matmul.params) continue;
    if (pfss_batch) {
      matmul.params->pfss_batch = pfss_batch;
      matmul.params->defer_trunc_finalize = true;
    }
    int src_frac = op.rescale.from_frac
                       ? op.rescale.from_frac
                       : (op.rescale.matmul_op != static_cast<size_t>(-1)
                              ? ops_[op.rescale.matmul_op].matmul.frac_bits * 2
                              : tensors_[static_cast<size_t>(op.inputs[0])].scale.frac_bits);
    int dst_frac = op.rescale.to_frac
                       ? op.rescale.to_frac
                       : tensors_[static_cast<size_t>(op.outputs[0])].scale.frac_bits;
    if (src_frac == dst_frac) continue;
    MatmulRescaleSite site;
    site.params = matmul.params;
    site.M = matmul.M;
    site.K = matmul.K;
    site.N = matmul.N;
    site.x_range = tensors_[static_cast<size_t>(ops_[op.rescale.matmul_op].inputs[0])].range;
    site.w_range = matmul.w_range;
    cfg.matmuls.push_back(site);
  }
  return run_truncation_pass(cfg, ctx);
}

}  // namespace compiler
