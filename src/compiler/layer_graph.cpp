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
