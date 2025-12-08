#include "compiler/layer_graph.hpp"

#include <cmath>
#include <stdexcept>
#include <limits>

namespace compiler {

namespace {

inline uint64_t ceil_div_u64(uint64_t a, uint64_t b) {
  if (b == 0) return std::numeric_limits<uint64_t>::max();
  return (a + b - 1) / b;
}

}  // namespace

int LayerGraph::add_tensor(const Scale& scale, const RangeInterval& r) {
  TensorFacts t;
  t.scale = scale;
  t.range = r;
  t.abs = abs_from_range(r, scale.is_signed);
  t.mask_abs = default_mask_bound(scale.frac_bits);
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

int LayerGraph::add_bias(int x, const std::vector<int64_t>& bias_qf, const Scale& out_scale) {
  OpNode op;
  op.kind = OpKind::kBiasAdd;
  op.inputs = {x};
  op.outputs = {add_tensor(out_scale, RangeInterval::whole(out_scale.is_signed))};
  op.bias = bias_qf;
  ops_.push_back(op);
  return op.outputs[0];
}

int LayerGraph::add_clamp(int x, const RangeInterval& r, const Scale& out_scale) {
  OpNode op;
  op.kind = OpKind::kClamp;
  op.inputs = {x};
  op.outputs = {add_tensor(out_scale, r)};
  op.clamp_range = r;
  ops_.push_back(op);
  return op.outputs[0];
}

void LayerGraph::propagate_ranges() {
  for (const auto& op : ops_) {
    auto get_tensor = [&](int tid) -> const TensorFacts& {
      if (tid < 0 || static_cast<size_t>(tid) >= tensors_.size()) {
        throw std::runtime_error("propagate_ranges: tensor id out of range");
      }
      return tensors_[static_cast<size_t>(tid)];
    };
    auto set_tensor = [&](int tid,
                          const RangeInterval& r,
                          const AbsBound& ab,
                          std::optional<GapCert> gap = std::nullopt,
                          uint64_t mask_abs = 0) {
      if (tid < 0 || static_cast<size_t>(tid) >= tensors_.size()) {
        throw std::runtime_error("propagate_ranges: tensor id out of range");
      }
      auto& t = tensors_[static_cast<size_t>(tid)];
      t.range = r;
      t.abs = ab;
       if (mask_abs == 0) mask_abs = default_mask_bound(t.scale.frac_bits);
       t.mask_abs = mask_abs;
      if (gap && gap->kind == RangeKind::Proof && gap->is_signed) {
        t.gap = gap;
      } else if (ab.kind == RangeKind::Proof && ab.is_signed) {
        t.gap = gap_from_abs(ab, t.scale.frac_bits, t.mask_abs);
      } else {
        t.gap.reset();
      }
    };

    switch (op.kind) {
      case OpKind::kMatmulBeaver: {
        const auto& attrs = op.matmul;
        const auto& tx = get_tensor(op.inputs[0]);
        RangeInterval acc;
        AbsBound acc_abs;
        if (attrs.row_l1_max > 0) {
          acc = propagate_matmul_accum_rowl1(tx.range, attrs.row_l1_max);
          acc_abs = matmul_rowl1_abs(tx.abs, attrs.row_l1_max);
        } else {
          acc = propagate_matmul_accum(tx.range, attrs.w_range, attrs.K);
          AbsBound w_abs = abs_from_range(attrs.w_range, true);
          acc_abs = matmul_accum_abs(tx.abs, w_abs, attrs.K);
        }
        set_tensor(op.outputs[0], acc, acc_abs);
        break;
      }
      case OpKind::kAdd: {
        auto ra = propagate_add(get_tensor(op.inputs[0]).range, get_tensor(op.inputs[1]).range);
        auto ab = add_abs(get_tensor(op.inputs[0]).abs, get_tensor(op.inputs[1]).abs);
        uint64_t mask_abs = std::max(get_tensor(op.inputs[0]).mask_abs,
                                     get_tensor(op.inputs[1]).mask_abs);
        set_tensor(op.outputs[0], ra, ab, gap_from_abs(ab, tensors_[static_cast<size_t>(op.outputs[0])].scale.frac_bits, mask_abs), mask_abs);
        break;
      }
      case OpKind::kSub: {
        auto ra = propagate_sub(get_tensor(op.inputs[0]).range, get_tensor(op.inputs[1]).range);
        auto ab = sub_abs(get_tensor(op.inputs[0]).abs, get_tensor(op.inputs[1]).abs);
        uint64_t mask_abs = std::max(get_tensor(op.inputs[0]).mask_abs,
                                     get_tensor(op.inputs[1]).mask_abs);
        set_tensor(op.outputs[0], ra, ab, gap_from_abs(ab, tensors_[static_cast<size_t>(op.outputs[0])].scale.frac_bits, mask_abs), mask_abs);
        break;
      }
      case OpKind::kMulConst: {
        auto ra = propagate_mul_const(get_tensor(op.inputs[0]).range, op.scalar, op.frac_bits);
        auto ab = mul_const_abs(get_tensor(op.inputs[0]).abs, op.scalar, op.frac_bits);
        uint64_t mask_abs = get_tensor(op.inputs[0]).mask_abs;
        set_tensor(op.outputs[0], ra, ab, gap_from_abs(ab, tensors_[static_cast<size_t>(op.outputs[0])].scale.frac_bits, mask_abs), mask_abs);
        break;
      }
      case OpKind::kAxpy: {
        auto ra = propagate_axpy(get_tensor(op.inputs[0]).range,
                                 get_tensor(op.inputs[1]).range,
                                  op.scalar,
                                  op.frac_bits);
        auto ab = axpy_abs(get_tensor(op.inputs[0]).abs,
                           get_tensor(op.inputs[1]).abs,
                           op.scalar,
                           op.frac_bits);
        uint64_t mask_abs = std::max(get_tensor(op.inputs[0]).mask_abs,
                                     get_tensor(op.inputs[1]).mask_abs);
        set_tensor(op.outputs[0], ra, ab, gap_from_abs(ab, tensors_[static_cast<size_t>(op.outputs[0])].scale.frac_bits, mask_abs), mask_abs);
        break;
      }
      case OpKind::kHadamard: {
        RangeInterval prod = mul_range(get_tensor(op.inputs[0]).range, get_tensor(op.inputs[1]).range);
        RangeInterval ra = shift_down(prod, op.frac_bits);
        auto ab = hadamard_abs(get_tensor(op.inputs[0]).abs, get_tensor(op.inputs[1]).abs, op.frac_bits);
        uint64_t mask_abs = std::max(get_tensor(op.inputs[0]).mask_abs,
                                     get_tensor(op.inputs[1]).mask_abs);
        set_tensor(op.outputs[0], ra, ab, gap_from_abs(ab, tensors_[static_cast<size_t>(op.outputs[0])].scale.frac_bits, mask_abs), mask_abs);
        break;
      }
      case OpKind::kRescale: {
        RangeInterval r = get_tensor(op.inputs[0]).range;
        int shift = (op.rescale.from_frac - op.rescale.to_frac);
        RangeInterval ra = shift_down(r, shift);
        AbsBound ab = shift_down_abs(get_tensor(op.inputs[0]).abs, shift);
        uint64_t mask_abs = default_mask_bound(tensors_[static_cast<size_t>(op.outputs[0])].scale.frac_bits);
        set_tensor(op.outputs[0], ra, ab, gap_from_abs(ab, tensors_[static_cast<size_t>(op.outputs[0])].scale.frac_bits, mask_abs), mask_abs);
        break;
      }
      case OpKind::kMean: {
        // Exact bound: |mean| <= |x|max. Preserve proof if input is proof.
        const auto& x_t = get_tensor(op.inputs[0]);
        RangeInterval ra = x_t.range;
        int len = std::max(op.length, 1);
        // Tighten range by dividing by len in fixed-point.
        if (len > 1) {
          ra.lo /= len;
          ra.hi /= len;
        }
        AbsBound ab = x_t.abs;
        if (ab.kind == RangeKind::Proof) {
          ab.kind = RangeKind::Proof;
        } else {
          ab.kind = RangeKind::Hint;
        }
        uint64_t mask_abs = x_t.mask_abs;
        set_tensor(op.outputs[0], ra, ab, gap_from_abs(ab, x_t.scale.frac_bits, mask_abs), mask_abs);
        break;
      }
      case OpKind::kVar: {
        const auto& x_t = get_tensor(op.inputs[0]);
        const auto& m_t = get_tensor(op.inputs[1]);
        RangeInterval diff = propagate_sub(x_t.range, m_t.range);
        RangeInterval sq = propagate_mul(diff, diff, op.frac_bits);
        int len = std::max(op.length, 1);
        RangeInterval ra = sq;
        if (len > 1) {
          ra.lo /= len;
          ra.hi /= len;
        }

        AbsBound ax = x_t.abs;
        AbsBound am = m_t.abs;
        AbsBound diff_abs = add_abs(ax, am);
        // |diff| <= |x| + |m|. var <= E[diff^2]/2^f.
        uint64_t sq_abs = sat_mul_u64(diff_abs.max_abs, diff_abs.max_abs);
        uint64_t var_abs = ceil_div_u64(ceil_div_pow2(sq_abs, op.frac_bits), static_cast<uint64_t>(len));
        AbsBound ab{true, var_abs,
                    (diff_abs.kind == RangeKind::Proof) ? RangeKind::Proof : RangeKind::Hint};
        uint64_t mask_abs = std::max(x_t.mask_abs, m_t.mask_abs);
        set_tensor(op.outputs[0], ra, ab, gap_from_abs(ab, op.frac_bits, mask_abs), mask_abs);
        break;
      }
      case OpKind::kRsqrt: {
        RangeInterval r;
        r.is_signed = true;
        r.lo = 0;
        r.hi = static_cast<int64_t>(1ll << op.frac_bits);
        AbsBound ab;
        ab.is_signed = true;
        ab.max_abs = static_cast<uint64_t>(1ull << op.frac_bits);
        ab.kind = RangeKind::Proof;
        uint64_t mask_abs = default_mask_bound(op.frac_bits);
        set_tensor(op.outputs[0], r, ab, gap_from_abs(ab, op.frac_bits, mask_abs), mask_abs);
        break;
      }
      case OpKind::kAffine: {
        // If gamma/beta are present, propagate exact affine bounds; otherwise
        // fall back to a conservative clamp.
        const bool has_gamma = (op.inputs.size() > 1 && op.inputs[1] >= 0);
        const bool has_beta = (op.inputs.size() > 2 && op.inputs[2] >= 0);
        if (has_gamma || has_beta) {
          const auto& x = get_tensor(op.inputs[0]);
          RangeInterval g_range = has_gamma ? get_tensor(op.inputs[1]).range
                                            : RangeInterval{true, 1, 1};
          RangeInterval b_range = has_beta ? get_tensor(op.inputs[2]).range
                                            : RangeInterval{true, 0, 0};
          RangeInterval prod = mul_range(x.range, g_range);
          prod = shift_down(prod, op.frac_bits);
          RangeInterval ra = add_range(prod, b_range);

          AbsBound ag = has_gamma ? get_tensor(op.inputs[1]).abs : AbsBound{true, 1, RangeKind::Proof};
          AbsBound abeta = has_beta ? get_tensor(op.inputs[2]).abs : AbsBound{true, 0, RangeKind::Proof};
          AbsBound axg = hadamard_abs(x.abs, ag, op.frac_bits);
          AbsBound ab = add_abs(axg, abeta);
          uint64_t mask_abs = std::max<uint64_t>(x.mask_abs,
                                                 std::max(has_gamma ? get_tensor(op.inputs[1]).mask_abs : 0ull,
                                                          has_beta ? get_tensor(op.inputs[2]).mask_abs : 0ull));
          set_tensor(op.outputs[0], ra, ab, gap_from_abs(ab, op.frac_bits, mask_abs), mask_abs);
        } else {
          int64_t bound = static_cast<int64_t>(8ll << op.frac_bits);
          RangeInterval clamp_r;
          clamp_r.is_signed = true;
          clamp_r.lo = -bound;
          clamp_r.hi = bound;
          RangeInterval ra =
              clamp_range(get_tensor(op.inputs[0]).range, clamp_r.lo, clamp_r.hi, clamp_r.is_signed);
          AbsBound ab;
          ab.is_signed = true;
          ab.max_abs = static_cast<uint64_t>(bound);
          ab.kind = RangeKind::Proof;
          uint64_t mask_abs = default_mask_bound(op.frac_bits);
          set_tensor(op.outputs[0], ra, ab, gap_from_abs(ab, op.frac_bits, mask_abs), mask_abs);
        }
        break;
      }
      case OpKind::kBiasAdd: {
        RangeInterval xr = get_tensor(op.inputs[0]).range;
        RangeInterval br;
        br.is_signed = true;
        if (!op.bias.empty()) {
          int64_t lo = op.bias[0];
          int64_t hi = op.bias[0];
          for (size_t i = 1; i < op.bias.size(); ++i) {
            int64_t v = op.bias[i];
            if (v < lo) lo = v;
            if (v > hi) hi = v;
          }
          br.lo = lo;
          br.hi = hi;
        } else {
          br.lo = br.hi = 0;
        }
        RangeInterval ra = add_range(xr, br);
        uint64_t bias_abs = static_cast<uint64_t>(std::max(std::abs(br.lo), std::abs(br.hi)));
        AbsBound bias_b{true, bias_abs, RangeKind::Proof};
        AbsBound ab = add_abs(get_tensor(op.inputs[0]).abs, bias_b);
        uint64_t mask_abs = get_tensor(op.inputs[0]).mask_abs;
        set_tensor(op.outputs[0], ra, ab, gap_from_abs(ab, tensors_[static_cast<size_t>(op.outputs[0])].scale.frac_bits, mask_abs), mask_abs);
        break;
      }
      case OpKind::kClamp: {
        RangeInterval ra =
            clamp_range(get_tensor(op.inputs[0]).range, op.clamp_range.lo, op.clamp_range.hi,
                        op.clamp_range.is_signed);
        uint64_t max_abs = static_cast<uint64_t>(std::max(std::abs(op.clamp_range.lo),
                                                          std::abs(op.clamp_range.hi)));
        AbsBound ab;
        ab.is_signed = op.clamp_range.is_signed;
        ab.max_abs = max_abs;
        ab.kind = RangeKind::Proof;
        uint64_t mask_abs = default_mask_bound(op.outputs.empty()
                                                   ? 0
                                                   : tensors_[static_cast<size_t>(op.outputs[0])].scale.frac_bits);
        set_tensor(op.outputs[0], ra, ab, gap_from_abs(ab, op.outputs.empty()
                                                           ? 0
                                                           : tensors_[static_cast<size_t>(op.outputs[0])].scale.frac_bits, mask_abs), mask_abs);
        break;
      }
      default:
        break;
    }
  }
}

void LayerGraph::hoist_rescales() {
  // Map tensor id -> producing op index for quick lookups.
  auto rebuild_producer = [&]() {
    std::vector<int> prod(tensors_.size(), -1);
    for (size_t i = 0; i < ops_.size(); ++i) {
      for (int t : ops_[i].outputs) {
        if (t >= 0 && static_cast<size_t>(t) < prod.size()) {
          prod[static_cast<size_t>(t)] = static_cast<int>(i);
        }
      }
    }
    return prod;
  };
  auto has_gap = [&](int tid) -> bool {
    if (tid < 0 || static_cast<size_t>(tid) >= tensors_.size()) return false;
    const auto& t = tensors_[static_cast<size_t>(tid)];
    if (t.gap && can_gapars(*t.gap)) return true;
    auto g = gap_from_abs(t.abs, t.scale.frac_bits, t.mask_abs);
    return g && can_gapars(*g);
  };
  auto proof_ok = [&](int tid) -> bool {
    if (tid < 0 || static_cast<size_t>(tid) >= tensors_.size()) return false;
    const auto& t = tensors_[static_cast<size_t>(tid)];
    return t.abs.kind == RangeKind::Proof;
  };
  auto safe_sum_bound = [&](const AbsBound& a, int len) -> bool {
    if (len <= 0) len = 1;
    __int128 sum = static_cast<__int128>(a.max_abs) * static_cast<__int128>(len);
    return sum <= static_cast<__int128>(std::numeric_limits<int64_t>::max());
  };
  std::vector<int> producer = rebuild_producer();

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
  // after the add/sub. Extend to BiasAdd and Hadamard when the sides match.
  for (auto& op : ops_) {
    bool is_addlike = (op.kind == OpKind::kAdd || op.kind == OpKind::kSub || op.kind == OpKind::kBiasAdd ||
                       op.kind == OpKind::kAxpy);
    if (!is_addlike) continue;
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
    bool gap_ok = has_gap(a_prod.outputs[0]) && has_gap(b_prod.outputs[0]);
    bool same_sign = (tensors_[static_cast<size_t>(a_prod.outputs[0])].scale.is_signed ==
                      tensors_[static_cast<size_t>(b_prod.outputs[0])].scale.is_signed);
    if (!same_shift || (!same_sign && !gap_ok)) continue;

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
        tensors_[static_cast<size_t>(b_prod.inputs[0])].scale.is_signed || gap_ok;
    RangeInterval ra = tensors_[static_cast<size_t>(op.inputs[0])].range;
    RangeInterval rb = tensors_[static_cast<size_t>(op.inputs[1])].range;
    RangeInterval sum_range = (op.kind == OpKind::kAdd) ? add_range(ra, rb)
                          : (op.kind == OpKind::kSub) ? sub_range(ra, rb)
                          : (op.kind == OpKind::kAxpy) ? axpy_range(ra, rb, op.scalar, op.frac_bits)
                                                       : add_range(ra, rb);
    tensors_[static_cast<size_t>(op.outputs[0])].range = sum_range;
    // Refresh abs bound after rewriting inputs.
    if (op.kind == OpKind::kAdd) {
      tensors_[static_cast<size_t>(op.outputs[0])].abs =
          add_abs(tensors_[static_cast<size_t>(op.inputs[0])].abs,
                  tensors_[static_cast<size_t>(op.inputs[1])].abs);
    } else if (op.kind == OpKind::kSub) {
      tensors_[static_cast<size_t>(op.outputs[0])].abs =
          sub_abs(tensors_[static_cast<size_t>(op.inputs[0])].abs,
                  tensors_[static_cast<size_t>(op.inputs[1])].abs);
    } else if (op.kind == OpKind::kAxpy) {
      tensors_[static_cast<size_t>(op.outputs[0])].abs =
          axpy_abs(tensors_[static_cast<size_t>(op.inputs[0])].abs,
                   tensors_[static_cast<size_t>(op.inputs[1])].abs,
                   op.scalar,
                   op.frac_bits);
    } else {
      tensors_[static_cast<size_t>(op.outputs[0])].abs =
          abs_from_range(sum_range, tensors_[static_cast<size_t>(op.outputs[0])].scale.is_signed);
    }

    // Insert a new rescale op after the add/sub to restore expected frac_bits.
    OpNode new_rescale;
    new_rescale.kind = OpKind::kRescale;
    new_rescale.inputs = {op.outputs[0]};
    new_rescale.outputs = {add_tensor(tensors_[static_cast<size_t>(op.outputs[0])].scale,
                                      shift_down(sum_range, from_frac - to_frac))};
    new_rescale.rescale.from_frac = from_frac;
    new_rescale.rescale.to_frac = to_frac;
    new_rescale.rescale.signed_ars = ar.signed_ars;
    new_rescale.rescale.prefer_gapars = ar.prefer_gapars || br.prefer_gapars || gap_ok;
    ops_.push_back(new_rescale);
    // Redirect the original consumer output to the new rescale output.
    op.outputs[0] = new_rescale.outputs[0];
    tensors_[static_cast<size_t>(op.outputs[0])].abs =
        abs_from_range(tensors_[static_cast<size_t>(op.outputs[0])].range,
                       tensors_[static_cast<size_t>(op.outputs[0])].scale.is_signed);
    tensors_[static_cast<size_t>(new_rescale.outputs[0])].abs =
        abs_from_range(tensors_[static_cast<size_t>(new_rescale.outputs[0])].range,
                       tensors_[static_cast<size_t>(new_rescale.outputs[0])].scale.is_signed);
    // Producer map is now stale; rebuild to keep further hoists consistent.
    producer = rebuild_producer();
  }

  // Hoist identical rescales over Hadamard mul when both inputs have the same rescale.
  for (auto& op : ops_) {
    bool is_mul = (op.kind == OpKind::kHadamard);
    bool is_axpy_like = (op.kind == OpKind::kAxpy);
    if (!is_mul && !is_axpy_like) continue;
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
    bool gap_ok = has_gap(a_prod.outputs[0]) && has_gap(b_prod.outputs[0]);
    bool same_sign = (tensors_[static_cast<size_t>(a_prod.outputs[0])].scale.is_signed ==
                      tensors_[static_cast<size_t>(b_prod.outputs[0])].scale.is_signed);
    if (!same_shift || (!same_sign && !gap_ok)) continue;
    int from_frac = ar.from_frac ? ar.from_frac
                                 : tensors_[static_cast<size_t>(a_prod.inputs[0])].scale.frac_bits;
    int to_frac = ar.to_frac ? ar.to_frac
                             : tensors_[static_cast<size_t>(op.outputs[0])].scale.frac_bits;
    op.inputs[0] = a_prod.inputs[0];
    op.inputs[1] = b_prod.inputs[0];
    tensors_[static_cast<size_t>(op.outputs[0])].scale.frac_bits = from_frac;
    tensors_[static_cast<size_t>(op.outputs[0])].scale.is_signed =
        tensors_[static_cast<size_t>(a_prod.inputs[0])].scale.is_signed ||
        tensors_[static_cast<size_t>(b_prod.inputs[0])].scale.is_signed || gap_ok;
    RangeInterval ra = tensors_[static_cast<size_t>(op.inputs[0])].range;
    RangeInterval rb = tensors_[static_cast<size_t>(op.inputs[1])].range;
    RangeInterval prod = mul_range(ra, rb);
    tensors_[static_cast<size_t>(op.outputs[0])].range = prod;

    OpNode new_rescale;
    new_rescale.kind = OpKind::kRescale;
    new_rescale.inputs = {op.outputs[0]};
    new_rescale.outputs = {add_tensor(tensors_[static_cast<size_t>(op.outputs[0])].scale,
                                      shift_down(prod, from_frac - to_frac))};
    new_rescale.rescale.from_frac = from_frac;
    new_rescale.rescale.to_frac = to_frac;
    new_rescale.rescale.signed_ars = ar.signed_ars;
    new_rescale.rescale.prefer_gapars = ar.prefer_gapars || br.prefer_gapars || gap_ok;
    ops_.push_back(new_rescale);
    op.outputs[0] = new_rescale.outputs[0];
    tensors_[static_cast<size_t>(op.outputs[0])].abs =
        abs_from_range(tensors_[static_cast<size_t>(op.outputs[0])].range,
                       tensors_[static_cast<size_t>(op.outputs[0])].scale.is_signed);
    tensors_[static_cast<size_t>(new_rescale.outputs[0])].abs =
        abs_from_range(tensors_[static_cast<size_t>(new_rescale.outputs[0])].range,
                       tensors_[static_cast<size_t>(new_rescale.outputs[0])].scale.is_signed);
    producer = rebuild_producer();
  }

  // Hoist identical rescales over mul_const/axpy when both inputs/scalars share shifts.
  for (auto& op : ops_) {
    if (op.kind != OpKind::kMulConst && op.kind != OpKind::kAxpy) continue;
    if (op.inputs.empty()) continue;
    int x_tid = op.inputs[0];
    if (x_tid < 0 || static_cast<size_t>(x_tid) >= producer.size()) continue;
    int x_prod_idx = producer[static_cast<size_t>(x_tid)];
    if (x_prod_idx < 0 || static_cast<size_t>(x_prod_idx) >= ops_.size()) continue;
    auto& x_prod = ops_[static_cast<size_t>(x_prod_idx)];
    if (x_prod.kind != OpKind::kRescale) continue;
    bool same_sign = (tensors_[static_cast<size_t>(x_prod.outputs[0])].scale.is_signed ==
                      tensors_[static_cast<size_t>(op.outputs[0])].scale.is_signed);
    bool gap_ok = has_gap(x_prod.outputs[0]);
    if (!same_sign && !gap_ok) continue;
    int from_frac = x_prod.rescale.from_frac
                        ? x_prod.rescale.from_frac
                        : tensors_[static_cast<size_t>(x_prod.inputs[0])].scale.frac_bits;
    int to_frac = x_prod.rescale.to_frac
                      ? x_prod.rescale.to_frac
                      : tensors_[static_cast<size_t>(op.outputs[0])].scale.frac_bits;
    // For axpy, require the y input to share the same rescale shift to preserve alignment.
    int y_prod_idx = -1;
    bool gap_y_ok = false;
    if (op.kind == OpKind::kAxpy && op.inputs.size() >= 2) {
      int y_tid = op.inputs[1];
      if (y_tid < 0 || static_cast<size_t>(y_tid) >= producer.size()) continue;
      y_prod_idx = producer[static_cast<size_t>(y_tid)];
      if (y_prod_idx < 0 || static_cast<size_t>(y_prod_idx) >= ops_.size()) continue;
      auto& y_prod = ops_[static_cast<size_t>(y_prod_idx)];
      if (y_prod.kind != OpKind::kRescale) continue;
      bool same_shift_y = (x_prod.rescale.from_frac == y_prod.rescale.from_frac) &&
                          (x_prod.rescale.to_frac == y_prod.rescale.to_frac);
      if (!same_shift_y) continue;
      gap_y_ok = has_gap(y_prod.outputs[0]);
    }
    // Move rescale after mul_const/axpy when shift matches target scale.
    op.inputs[0] = x_prod.inputs[0];
    tensors_[static_cast<size_t>(op.outputs[0])].scale.frac_bits = from_frac;
    tensors_[static_cast<size_t>(op.outputs[0])].scale.is_signed =
        tensors_[static_cast<size_t>(x_prod.inputs[0])].scale.is_signed || gap_ok;
    RangeInterval xr = tensors_[static_cast<size_t>(op.inputs[0])].range;
    RangeInterval out_range = RangeInterval::whole(true);
    if (op.kind == OpKind::kMulConst) {
      out_range = mul_const_range(xr, op.scalar, op.frac_bits);
    } else {
      if (y_prod_idx < 0) continue;
      RangeInterval yr = tensors_[static_cast<size_t>(op.inputs[1])].range;
      out_range = axpy_range(xr, yr, op.scalar, op.frac_bits);
    }
    tensors_[static_cast<size_t>(op.outputs[0])].range = out_range;
    // Preserve tighter abs bound when moving the rescale.
    if (op.kind == OpKind::kMulConst) {
      tensors_[static_cast<size_t>(op.outputs[0])].abs =
          mul_const_abs(tensors_[static_cast<size_t>(op.inputs[0])].abs, op.scalar, op.frac_bits);
    } else {
      tensors_[static_cast<size_t>(op.outputs[0])].abs =
          axpy_abs(tensors_[static_cast<size_t>(op.inputs[0])].abs,
                   tensors_[static_cast<size_t>(op.inputs[1])].abs,
                   op.scalar,
                   op.frac_bits);
    }
    // Insert trailing rescale.
    OpNode new_rescale;
    new_rescale.kind = OpKind::kRescale;
    new_rescale.inputs = {op.outputs[0]};
      new_rescale.outputs = {add_tensor(tensors_[static_cast<size_t>(op.outputs[0])].scale,
                                        shift_down(out_range, from_frac - to_frac))};
    new_rescale.rescale.from_frac = from_frac;
    new_rescale.rescale.to_frac = to_frac;
    new_rescale.rescale.signed_ars = x_prod.rescale.signed_ars;
    new_rescale.rescale.prefer_gapars =
        x_prod.rescale.prefer_gapars || gap_ok || gap_y_ok;
    ops_.push_back(new_rescale);
    op.outputs[0] = new_rescale.outputs[0];
    tensors_[static_cast<size_t>(op.outputs[0])].abs =
        abs_from_range(tensors_[static_cast<size_t>(op.outputs[0])].range,
                       tensors_[static_cast<size_t>(op.outputs[0])].scale.is_signed);
    tensors_[static_cast<size_t>(new_rescale.outputs[0])].abs =
        abs_from_range(tensors_[static_cast<size_t>(new_rescale.outputs[0])].range,
                       tensors_[static_cast<size_t>(new_rescale.outputs[0])].scale.is_signed);
    producer = rebuild_producer();
  }

  // Hoist a rescale past BiasAdd when the bias can be safely upscaled to the
  // parent precision. This keeps the bias/public weights in the higher-precision
  // domain and lets downstream truncation reuse the shared rescale.
  for (auto& op : ops_) {
    if (op.kind != OpKind::kBiasAdd) continue;
    if (op.inputs.size() != 1) continue;
    int x_tid = op.inputs[0];
    if (x_tid < 0 || static_cast<size_t>(x_tid) >= producer.size()) continue;
    int x_prod_idx = producer[static_cast<size_t>(x_tid)];
    if (x_prod_idx < 0 || static_cast<size_t>(x_prod_idx) >= ops_.size()) continue;
    auto& parent = ops_[static_cast<size_t>(x_prod_idx)];
    if (parent.kind != OpKind::kRescale) continue;
    int from_frac = parent.rescale.from_frac
                        ? parent.rescale.from_frac
                        : tensors_[static_cast<size_t>(parent.inputs[0])].scale.frac_bits;
    int to_frac = parent.rescale.to_frac
                      ? parent.rescale.to_frac
                      : tensors_[static_cast<size_t>(op.outputs[0])].scale.frac_bits;
    if (from_frac <= to_frac) continue;  // only down-scaling rescale can be hoisted
    int shift = from_frac - to_frac;
    std::vector<int64_t> bias_up(op.bias.size(), 0);
    bool overflow = false;
    for (size_t i = 0; i < op.bias.size(); ++i) {
      __int128 v = static_cast<__int128>(op.bias[i]) << shift;
      if (v > static_cast<__int128>(std::numeric_limits<int64_t>::max()) ||
          v < static_cast<__int128>(std::numeric_limits<int64_t>::min())) {
        overflow = true;
        break;
      }
      bias_up[i] = static_cast<int64_t>(v);
    }
    if (overflow) continue;

    op.bias.swap(bias_up);
    op.inputs[0] = parent.inputs[0];
    auto& out_t = tensors_[static_cast<size_t>(op.outputs[0])];
    out_t.scale.frac_bits = from_frac;
    out_t.scale.is_signed = true;
    RangeInterval xr = tensors_[static_cast<size_t>(op.inputs[0])].range;
    RangeInterval br;
    br.is_signed = true;
    if (!op.bias.empty()) {
      br.lo = br.hi = op.bias[0];
      for (size_t i = 1; i < op.bias.size(); ++i) {
        int64_t v = op.bias[i];
        if (v < br.lo) br.lo = v;
        if (v > br.hi) br.hi = v;
      }
    } else {
      br.lo = br.hi = 0;
    }
    RangeInterval out_r = add_range(xr, br);
    out_t.range = out_r;
    uint64_t bias_abs = static_cast<uint64_t>(std::max(std::abs(br.lo), std::abs(br.hi)));
    AbsBound bias_b{true, bias_abs, RangeKind::Proof};
    out_t.abs = add_abs(tensors_[static_cast<size_t>(op.inputs[0])].abs, bias_b);
    out_t.gap = gap_from_abs(out_t.abs, from_frac);

    OpNode new_rescale;
    new_rescale.kind = OpKind::kRescale;
    new_rescale.inputs = {op.outputs[0]};
    Scale final_scale = out_t.scale;
    final_scale.frac_bits = to_frac;
    new_rescale.outputs = {add_tensor(final_scale, shift_down(out_r, shift))};
    new_rescale.rescale.from_frac = from_frac;
    new_rescale.rescale.to_frac = to_frac;
    new_rescale.rescale.signed_ars = parent.rescale.signed_ars;
    bool gap_ok = has_gap(parent.outputs[0]) || has_gap(op.outputs[0]);
    new_rescale.rescale.prefer_gapars = parent.rescale.prefer_gapars || gap_ok;
    ops_.push_back(new_rescale);
    op.outputs[0] = new_rescale.outputs[0];
    tensors_[static_cast<size_t>(op.outputs[0])].abs =
        abs_from_range(tensors_[static_cast<size_t>(op.outputs[0])].range,
                       tensors_[static_cast<size_t>(op.outputs[0])].scale.is_signed);
    tensors_[static_cast<size_t>(new_rescale.outputs[0])].abs =
        abs_from_range(tensors_[static_cast<size_t>(new_rescale.outputs[0])].range,
                       tensors_[static_cast<size_t>(new_rescale.outputs[0])].scale.is_signed);
    producer = rebuild_producer();
  }

  // Hoist rescale feeding LN mean when we have a proof bound and the sum fits
  // in int64_t at the higher precision. This keeps the statistics in the
  // high-precision domain and lets a single trunc handle the downscale.
  for (auto& op : ops_) {
    if (op.kind != OpKind::kMean) continue;
    if (op.inputs.empty()) continue;
    int x_tid = op.inputs[0];
    if (x_tid < 0 || static_cast<size_t>(x_tid) >= producer.size()) continue;
    int prod_idx = producer[static_cast<size_t>(x_tid)];
    if (prod_idx < 0 || static_cast<size_t>(prod_idx) >= ops_.size()) continue;
    auto& parent = ops_[static_cast<size_t>(prod_idx)];
    if (parent.kind != OpKind::kRescale) continue;
    const auto& in_t = tensors_[static_cast<size_t>(parent.inputs[0])];
    if (in_t.abs.kind != RangeKind::Proof) continue;
    int len = std::max(op.length, 1);
    if (!safe_sum_bound(in_t.abs, len)) continue;
    int from_frac = parent.rescale.from_frac
                        ? parent.rescale.from_frac
                        : tensors_[static_cast<size_t>(parent.inputs[0])].scale.frac_bits;
    int to_frac = parent.rescale.to_frac
                      ? parent.rescale.to_frac
                      : tensors_[static_cast<size_t>(op.outputs[0])].scale.frac_bits;
    if (from_frac <= to_frac) continue;  // only downscales are worth hoisting
    int shift = from_frac - to_frac;

    // Rewire mean to consume the pre-rescale tensor.
    op.inputs[0] = parent.inputs[0];
    auto& out_t = tensors_[static_cast<size_t>(op.outputs[0])];
    out_t.scale.frac_bits = from_frac;
    out_t.scale.is_signed = in_t.scale.is_signed;
    RangeInterval r = in_t.range;
    if (len > 1) {
      r.lo /= len;
      r.hi /= len;
    }
    out_t.range = r;
    out_t.abs = in_t.abs;
    out_t.gap = gap_from_abs(out_t.abs, from_frac);

    // Insert trailing rescale to restore the expected frac_bits.
    OpNode new_rescale;
    new_rescale.kind = OpKind::kRescale;
    new_rescale.inputs = {op.outputs[0]};
    Scale final_scale = out_t.scale;
    final_scale.frac_bits = to_frac;
    new_rescale.outputs = {add_tensor(final_scale, shift_down(r, shift))};
    new_rescale.rescale.from_frac = from_frac;
    new_rescale.rescale.to_frac = to_frac;
    new_rescale.rescale.signed_ars = parent.rescale.signed_ars;
    bool gap_ok = has_gap(parent.outputs[0]) || has_gap(op.outputs[0]) || has_gap(new_rescale.outputs[0]);
    new_rescale.rescale.prefer_gapars = parent.rescale.prefer_gapars || gap_ok;
    ops_.push_back(new_rescale);
    op.outputs[0] = new_rescale.outputs[0];
    tensors_[static_cast<size_t>(op.outputs[0])].abs =
        abs_from_range(tensors_[static_cast<size_t>(op.outputs[0])].range,
                       tensors_[static_cast<size_t>(op.outputs[0])].scale.is_signed);
    tensors_[static_cast<size_t>(new_rescale.outputs[0])].abs =
        abs_from_range(tensors_[static_cast<size_t>(new_rescale.outputs[0])].range,
                       tensors_[static_cast<size_t>(new_rescale.outputs[0])].scale.is_signed);
    producer = rebuild_producer();
  }

  // Hoist rescale over MulConst when safe: if input is a rescale with matching sign or gap cert.
  for (auto& op : ops_) {
    if (op.kind != OpKind::kMulConst) continue;
    if (op.inputs.empty()) continue;
    int in_tid = op.inputs[0];
    if (in_tid < 0 || static_cast<size_t>(in_tid) >= producer.size()) continue;
    int prod_idx = producer[static_cast<size_t>(in_tid)];
    if (prod_idx < 0 || static_cast<size_t>(prod_idx) >= ops_.size()) continue;
    auto& parent = ops_[static_cast<size_t>(prod_idx)];
    if (parent.kind != OpKind::kRescale) continue;
    const auto& pr = parent.rescale;
    bool gap_ok = has_gap(parent.outputs[0]);
    bool same_sign = (tensors_[static_cast<size_t>(parent.outputs[0])].scale.is_signed ==
                      tensors_[static_cast<size_t>(parent.inputs[0])].scale.is_signed);
    if (!same_sign && !gap_ok) continue;
    int from_frac = pr.from_frac ? pr.from_frac
                                 : tensors_[static_cast<size_t>(parent.inputs[0])].scale.frac_bits;
    int to_frac = pr.to_frac ? pr.to_frac
                             : tensors_[static_cast<size_t>(op.outputs[0])].scale.frac_bits;
    op.inputs[0] = parent.inputs[0];
    tensors_[static_cast<size_t>(op.outputs[0])].scale.frac_bits = from_frac;
    tensors_[static_cast<size_t>(op.outputs[0])].scale.is_signed =
        tensors_[static_cast<size_t>(parent.inputs[0])].scale.is_signed;
    RangeInterval xr = tensors_[static_cast<size_t>(op.inputs[0])].range;
    RangeInterval mr = mul_const_range(xr, op.scalar, op.frac_bits);
    tensors_[static_cast<size_t>(op.outputs[0])].range = mr;

    OpNode new_rescale;
    new_rescale.kind = OpKind::kRescale;
    new_rescale.inputs = {op.outputs[0]};
    new_rescale.outputs = {add_tensor(tensors_[static_cast<size_t>(op.outputs[0])].scale,
                                      shift_down(mr, from_frac - to_frac))};
    new_rescale.rescale.from_frac = from_frac;
    new_rescale.rescale.to_frac = to_frac;
    new_rescale.rescale.signed_ars = pr.signed_ars;
    new_rescale.rescale.prefer_gapars = pr.prefer_gapars || gap_ok;
    ops_.push_back(new_rescale);
    op.outputs[0] = new_rescale.outputs[0];
    tensors_[static_cast<size_t>(op.outputs[0])].abs =
        abs_from_range(tensors_[static_cast<size_t>(op.outputs[0])].range,
                       tensors_[static_cast<size_t>(op.outputs[0])].scale.is_signed);
    tensors_[static_cast<size_t>(new_rescale.outputs[0])].abs =
        abs_from_range(tensors_[static_cast<size_t>(new_rescale.outputs[0])].range,
                       tensors_[static_cast<size_t>(new_rescale.outputs[0])].scale.is_signed);
    producer = rebuild_producer();
  }

  // Guarded hoist: move rescale across LN variance/norm/affine when proof and
  // mask bounds show no overflow at higher precision.
  auto try_hoist_ln_like = [&](OpKind kind) {
    for (auto& op : ops_) {
      if (op.kind != kind) continue;
      if (op.inputs.empty()) continue;
      int x_tid = op.inputs[0];
      if (x_tid < 0 || static_cast<size_t>(x_tid) >= producer.size()) continue;
      int prod_idx = producer[static_cast<size_t>(x_tid)];
      if (prod_idx < 0 || static_cast<size_t>(prod_idx) >= ops_.size()) continue;
      auto& parent = ops_[static_cast<size_t>(prod_idx)];
      if (parent.kind != OpKind::kRescale) continue;
      const auto& in_t = tensors_[static_cast<size_t>(parent.inputs[0])];
      if (!proof_ok(parent.inputs[0])) continue;
      int from_frac = parent.rescale.from_frac
                          ? parent.rescale.from_frac
                          : in_t.scale.frac_bits;
      int to_frac = parent.rescale.to_frac
                        ? parent.rescale.to_frac
                        : tensors_[static_cast<size_t>(op.outputs[0])].scale.frac_bits;
      if (from_frac <= to_frac) continue;  // only hoist downscales

      // Overflow guards per op kind.
      bool overflow = false;
      RangeInterval out_r = tensors_[static_cast<size_t>(op.outputs[0])].range;
      AbsBound out_ab = tensors_[static_cast<size_t>(op.outputs[0])].abs;
      uint64_t mask_abs = tensors_[static_cast<size_t>(op.outputs[0])].mask_abs;
      if (kind == OpKind::kVar && op.inputs.size() >= 2) {
        // var ≈ E[(x-mean)^2]/2^f: ensure squared sum fits at from_frac.
        const auto& m_t = tensors_[static_cast<size_t>(op.inputs[1])];
        uint64_t diff_abs = sat_add_u64(in_t.abs.max_abs, m_t.abs.max_abs);
        __int128 sq = static_cast<__int128>(diff_abs) * static_cast<__int128>(diff_abs);
        int len = std::max(op.length, 1);
        __int128 scaled = sq >> op.frac_bits;
        if (scaled > std::numeric_limits<int64_t>::max() ||
            scaled < std::numeric_limits<int64_t>::min()) {
          overflow = true;
        } else {
          int64_t div = static_cast<int64_t>(scaled) / len;
          out_r.lo = 0;
          out_r.hi = div;
          out_ab = AbsBound{true, static_cast<uint64_t>(std::abs(div)), RangeKind::Proof};
          mask_abs = std::max(in_t.mask_abs, m_t.mask_abs);
        }
      } else if (kind == OpKind::kRsqrt) {
        // rsqrt output is bounded by 1<<frac_bits.
        out_r.is_signed = true;
        out_r.lo = 0;
        out_r.hi = static_cast<int64_t>(1ll << from_frac);
        out_ab = AbsBound{true, static_cast<uint64_t>(1ull << from_frac), RangeKind::Proof};
        mask_abs = default_mask_bound(from_frac);
      } else if (kind == OpKind::kAffine) {
        const bool has_gamma = (op.inputs.size() > 1 && op.inputs[1] >= 0);
        const bool has_beta = (op.inputs.size() > 2 && op.inputs[2] >= 0);
        if (!has_gamma && !has_beta) overflow = true;
        if (!overflow) {
          RangeInterval g_range = has_gamma ? tensors_[static_cast<size_t>(op.inputs[1])].range
                                            : RangeInterval{true, 1, 1};
          RangeInterval b_range = has_beta ? tensors_[static_cast<size_t>(op.inputs[2])].range
                                            : RangeInterval{true, 0, 0};
          RangeInterval prod = mul_range(in_t.range, g_range);
          prod = shift_down(prod, op.frac_bits);
          out_r = add_range(prod, b_range);
          AbsBound ag = has_gamma ? tensors_[static_cast<size_t>(op.inputs[1])].abs
                                  : AbsBound{true, 1, RangeKind::Proof};
          AbsBound abeta = has_beta ? tensors_[static_cast<size_t>(op.inputs[2])].abs
                                    : AbsBound{true, 0, RangeKind::Proof};
          out_ab = add_abs(hadamard_abs(in_t.abs, ag, op.frac_bits), abeta);
          mask_abs = std::max<uint64_t>(in_t.mask_abs,
                                        std::max(has_gamma ? tensors_[static_cast<size_t>(op.inputs[1])].mask_abs : 0ull,
                                                 has_beta ? tensors_[static_cast<size_t>(op.inputs[2])].mask_abs : 0ull));
        }
      }
      if (overflow) continue;

      op.inputs[0] = parent.inputs[0];
      tensors_[static_cast<size_t>(op.outputs[0])].scale.frac_bits = from_frac;
      tensors_[static_cast<size_t>(op.outputs[0])].scale.is_signed = true;
      tensors_[static_cast<size_t>(op.outputs[0])].range = out_r;
      tensors_[static_cast<size_t>(op.outputs[0])].abs = out_ab;
      tensors_[static_cast<size_t>(op.outputs[0])].mask_abs = mask_abs;
      tensors_[static_cast<size_t>(op.outputs[0])].gap = gap_from_abs(out_ab, from_frac, mask_abs);

      OpNode new_rescale;
      new_rescale.kind = OpKind::kRescale;
      new_rescale.inputs = {op.outputs[0]};
      Scale final_scale = tensors_[static_cast<size_t>(op.outputs[0])].scale;
      final_scale.frac_bits = to_frac;
      int shift = from_frac - to_frac;
      new_rescale.outputs = {add_tensor(final_scale, shift_down(out_r, shift))};
      new_rescale.rescale.from_frac = from_frac;
      new_rescale.rescale.to_frac = to_frac;
      new_rescale.rescale.signed_ars = parent.rescale.signed_ars;
      bool gap_ok = has_gap(parent.outputs[0]) || has_gap(op.outputs[0]) || has_gap(new_rescale.outputs[0]);
      new_rescale.rescale.prefer_gapars = parent.rescale.prefer_gapars || gap_ok;
      ops_.push_back(new_rescale);
      op.outputs[0] = new_rescale.outputs[0];
      tensors_[static_cast<size_t>(op.outputs[0])].abs =
          abs_from_range(tensors_[static_cast<size_t>(op.outputs[0])].range,
                         tensors_[static_cast<size_t>(op.outputs[0])].scale.is_signed);
      tensors_[static_cast<size_t>(new_rescale.outputs[0])].abs =
          abs_from_range(tensors_[static_cast<size_t>(new_rescale.outputs[0])].range,
                         tensors_[static_cast<size_t>(new_rescale.outputs[0])].scale.is_signed);
      producer = rebuild_producer();
    }
  };
  try_hoist_ln_like(OpKind::kVar);
  try_hoist_ln_like(OpKind::kRsqrt);
  try_hoist_ln_like(OpKind::kAffine);
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
    site.accum_range = matmul_accum_range(site.x_range, site.w_range, matmul.K);
    const auto& tin = tensors_[static_cast<size_t>(ops_[op.rescale.matmul_op].inputs[0])];
    AbsBound w_abs = abs_from_range(site.w_range, true);
    site.accum_abs = matmul_accum_abs(tin.abs, w_abs, matmul.K);
    std::optional<GapCert> g = tin.gap;
    if (!g && tin.abs.kind == RangeKind::Proof && tin.abs.is_signed) {
      g = gap_from_abs(tin.abs, src_frac, tin.mask_abs);
    }
    site.prefer_gapars = g && can_gapars(*g);
    site.gap_cert = g;
    cfg.matmuls.push_back(site);
  }
  return run_truncation_pass(cfg, ctx);
}

}  // namespace compiler
