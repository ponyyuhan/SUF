#include "nn/transformer_layer.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <random>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include "compiler/truncation_lowering.hpp"
#include "gates/composite_fss.hpp"
#include "gates/tables/rsqrt_piecewise_affine_init.hpp"
#include "suf/suf_silu_builders.hpp"
#include "runtime/phase_tasks.hpp"

namespace nn {

namespace {

inline int64_t to_signed(uint64_t v) { return static_cast<int64_t>(v); }
inline uint64_t to_ring(int64_t v) { return static_cast<uint64_t>(v); }

struct RowBroadcastTripleMaterial {
  int rows = 0;
  int cols = 0;
  std::vector<uint64_t> A0, A1;
  std::vector<uint64_t> B0, B1;
  std::vector<uint64_t> C0, C1;
};

RowBroadcastTripleMaterial make_row_broadcast_triples(int rows, int cols, std::mt19937_64& rng) {
  RowBroadcastTripleMaterial mat;
  mat.rows = rows;
  mat.cols = cols;
  size_t count = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  mat.A0.resize(count);
  mat.A1.resize(count);
  mat.B0.resize(static_cast<size_t>(rows));
  mat.B1.resize(static_cast<size_t>(rows));
  mat.C0.resize(count);
  mat.C1.resize(count);

  std::vector<uint64_t> B(rows);
  for (int r = 0; r < rows; ++r) {
    uint64_t b = rng();
    uint64_t b0 = rng();
    uint64_t b1 = b - b0;
    B[static_cast<size_t>(r)] = b;
    mat.B0[static_cast<size_t>(r)] = b0;
    mat.B1[static_cast<size_t>(r)] = b1;
  }
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      size_t idx = static_cast<size_t>(r * cols + c);
      uint64_t a = rng();
      uint64_t a0 = rng();
      uint64_t a1 = a - a0;
      uint64_t c_val = proto::mul_mod(a, B[static_cast<size_t>(r)]);
      uint64_t c0 = rng();
      uint64_t c1 = c_val - c0;
      mat.A0[idx] = a0;
      mat.A1[idx] = a1;
      mat.C0[idx] = c0;
      mat.C1[idx] = c1;
    }
  }
  return mat;
}

class CachedRowBroadcastTripleProvider : public runtime::RowBroadcastTripleProvider {
 public:
  explicit CachedRowBroadcastTripleProvider(int party) : party_(party) {}

  runtime::RowBroadcastTriple reserve_mul(int rows, int cols) override {
    const uint64_t key = (static_cast<uint64_t>(rows) << 32) | static_cast<uint32_t>(cols);
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      std::mt19937_64 rng(base_seed_ ^ key);
      auto mat = make_row_broadcast_triples(rows, cols, rng);
      it = cache_.emplace(key, std::move(mat)).first;
    }
    const auto& mat = it->second;
    const auto& A = (party_ == 0) ? mat.A0 : mat.A1;
    const auto& B = (party_ == 0) ? mat.B0 : mat.B1;
    const auto& C = (party_ == 0) ? mat.C0 : mat.C1;
    return {std::span<const uint64_t>(A.data(), A.size()),
            std::span<const uint64_t>(B.data(), B.size()),
            std::span<const uint64_t>(C.data(), C.size())};
  }

 private:
  static constexpr uint64_t base_seed_ = 0x6c6e7275ull;  // "lnru"
  int party_ = 0;
  std::unordered_map<uint64_t, RowBroadcastTripleMaterial> cache_;
};

// Deterministic triple cache (per party) keyed by count.
static std::vector<proto::BeaverTriple64Share>& ensure_cached_triples(
    std::unordered_map<uint64_t, std::vector<proto::BeaverTriple64Share>>& cache,
    uint64_t seed_base,
    size_t count,
    int party) {
  auto it = cache.find(count);
  if (it != cache.end()) return it->second;
  std::mt19937_64 rng(seed_base ^ static_cast<uint64_t>(count));
  std::vector<proto::BeaverTriple64Share> triples(count);
  for (auto& tri : triples) {
    uint64_t a = rng();
    uint64_t b = rng();
    uint64_t c = proto::mul_mod(a, b);
    uint64_t a0 = rng();
    uint64_t b0 = rng();
    uint64_t c0 = rng();
    tri = (party == 0) ? proto::BeaverTriple64Share{a0, b0, c0}
                       : proto::BeaverTriple64Share{a - a0, b - b0, c - c0};
  }
  auto [ins_it, _] = cache.emplace(count, std::move(triples));
  return ins_it->second;
}

inline void ensure_beaver_triples(gates::CompositeKeyPair& kp, size_t need, std::mt19937_64& rng) {
  auto fill = [&](std::vector<proto::BeaverTriple64Share>& dst0,
                  std::vector<proto::BeaverTriple64Share>& dst1) {
    while (dst0.size() < need || dst1.size() < need) {
      uint64_t a = rng(), b = rng(), c = proto::mul_mod(a, b);
      uint64_t a0 = rng();
      uint64_t a1 = a - a0;
      uint64_t b0 = rng();
      uint64_t b1 = b - b0;
      uint64_t c0 = rng();
      uint64_t c1 = c - c0;
      dst0.push_back({a0, b0, c0});
      dst1.push_back({a1, b1, c1});
    }
  };
  fill(kp.k0.triples, kp.k1.triples);
}

// Build a SUF that emits affine-init coefficients adjusted for fixed-point
// evaluation: out0 = a0 - (a1 * offset >> fb), out1 = a1.
inline suf::SUF<uint64_t> build_rsqrt_affine_eval_suf(const gates::PiecewisePolySpec& spec) {
  suf::SUF<uint64_t> F;
  F.n_bits = 64;
  F.r_out = 2;
  F.l_out = 0;
  F.degree = 0;

  std::vector<gates::PiecewiseInterval> intervals = spec.intervals;
  std::sort(intervals.begin(), intervals.end(),
            [](const gates::PiecewiseInterval& a, const gates::PiecewiseInterval& b) {
              return a.start < b.start;
            });
  if (intervals.empty()) return F;
  F.alpha.clear();
  F.alpha.reserve(intervals.size() + 1);
  F.alpha.push_back(intervals.front().start);

  if (F.alpha.front() != 0) {
    suf::SufPiece<uint64_t> zero_piece;
    zero_piece.polys.resize(2);
    zero_piece.polys[0].coeffs = {0};
    zero_piece.polys[1].coeffs = {0};
    F.pieces.push_back(std::move(zero_piece));
    F.alpha.insert(F.alpha.begin(), 0ull);
  }

  auto clamp_to_ring = []( __int128 v) -> uint64_t {
    if (v > static_cast<__int128>(std::numeric_limits<int64_t>::max())) {
      v = static_cast<__int128>(std::numeric_limits<int64_t>::max());
    }
    if (v < static_cast<__int128>(std::numeric_limits<int64_t>::min())) {
      v = static_cast<__int128>(std::numeric_limits<int64_t>::min());
    }
    return static_cast<uint64_t>(static_cast<int64_t>(v));
  };

  for (const auto& iv : intervals) {
    int64_t a0 = (!iv.pack.coeffs.empty()) ? iv.pack.coeffs[0] : 0;
    int64_t a1 = (iv.pack.coeffs.size() > 1) ? iv.pack.coeffs[1] : 0;
    int64_t offset = iv.pack.offset;
    __int128 prod = static_cast<__int128>(a1) * static_cast<__int128>(offset);
    int64_t offset_term = static_cast<int64_t>(prod >> spec.frac_bits_in);
    int64_t adj = a0 - offset_term;

    suf::SufPiece<uint64_t> piece;
    piece.polys.resize(2);
    piece.polys[0].coeffs = {clamp_to_ring(adj)};
    piece.polys[1].coeffs = {clamp_to_ring(a1)};
    F.pieces.push_back(std::move(piece));
    F.alpha.push_back(iv.end);
  }
  return F;
}

struct RsqrtTaskMaterial {
  suf::SUF<uint64_t> suf;
  gates::CompositeKeyPair keys;
  compiler::TruncationLoweringResult trunc_f;
  compiler::TruncationLoweringResult trunc_2f;
  gates::PiecewisePolySpec init_spec;
  int frac_bits = 0;
  int nr_iters = 1;
};

RsqrtTaskMaterial make_rsqrt_material(proto::PfssBackendBatch& backend,
                                      int frac_bits,
                                      int nr_iters,
                                      double eps,
                                      double vmax,
                                      std::mt19937_64& rng,
                                      int rows) {
  auto spec = gates::make_rsqrt_affine_init_spec(frac_bits, eps, vmax);
  auto suf_gate = build_rsqrt_affine_eval_suf(spec);
  std::vector<uint64_t> r_out(static_cast<size_t>(suf_gate.r_out), 0ull);
  auto kp = gates::composite_gen_backend_with_masks(
      suf_gate, backend, rng, rng(), r_out, static_cast<size_t>(rows), compiler::GateKind::Rsqrt);
  kp.k0.compiled.gate_kind = compiler::GateKind::Rsqrt;
  kp.k1.compiled.gate_kind = compiler::GateKind::Rsqrt;

  compiler::GateParams p;
  p.kind = compiler::GateKind::FaithfulARS;
  p.frac_bits = frac_bits;
  auto trunc_f = compiler::lower_truncation_gate(backend, rng, p, static_cast<size_t>(rows));
  p.frac_bits = 2 * frac_bits;
  auto trunc_2f = compiler::lower_truncation_gate(backend, rng, p, static_cast<size_t>(rows));
  std::fill(trunc_f.keys.k0.r_out_share.begin(), trunc_f.keys.k0.r_out_share.end(), 0ull);
  std::fill(trunc_f.keys.k1.r_out_share.begin(), trunc_f.keys.k1.r_out_share.end(), 0ull);
  std::fill(trunc_2f.keys.k0.r_out_share.begin(), trunc_2f.keys.k0.r_out_share.end(), 0ull);
  std::fill(trunc_2f.keys.k1.r_out_share.begin(), trunc_2f.keys.k1.r_out_share.end(), 0ull);

  RsqrtTaskMaterial out;
  out.init_spec = spec;
  out.suf = std::move(suf_gate);
  out.keys = std::move(kp);
  out.trunc_f = std::move(trunc_f);
  out.trunc_2f = std::move(trunc_2f);
  out.frac_bits = frac_bits;
  out.nr_iters = nr_iters;
  return out;
}

runtime::RsqrtTaskBundle make_rsqrt_bundle(RsqrtTaskMaterial& mat) {
  runtime::RsqrtTaskBundle b{};
  b.suf = &mat.suf;
  b.key0 = &mat.keys.k0;
  b.key1 = &mat.keys.k1;
  b.trunc_f = &mat.trunc_f;
  b.trunc_2f = &mat.trunc_2f;
  b.init_spec = &mat.init_spec;
  b.frac_bits = mat.frac_bits;
  b.nr_iters = mat.nr_iters;
  return b;
}

}  // namespace

void transformer_layer_forward(const TransformerConfig& cfg,
                               int party,
                               net::Chan& ch,
                               const TensorView<uint64_t>& X_share,
                               const TensorView<int64_t>& Wqkv_public,
                               const TensorView<int64_t>& Wout_public,
                               const TensorView<int64_t>& W1_public,
                               const TensorView<int64_t>& W2_public,
                               KVCache& cache,
                               TensorView<uint64_t> Y_share,
                               LayerContext* ctx,
                               runtime::PhaseExecutor* pe) {
  runtime::PhaseExecutor local_pe;
  bool created_local_pe = false;
  if (pe == nullptr) {
    pe = &local_pe;
    created_local_pe = true;
  }
  if (!ctx || !ctx->trunc_ctx) {
    throw std::runtime_error("transformer_layer_forward: LayerContext with trunc_ctx required");
  }
  // Ensure any previous async PFSS flush is complete before reusing batches.
  if (ctx->pfss_async_runner) {
    ctx->pfss_async_runner->wait();
    ctx->pfss_async_runner.reset();
  }
  // Enable async PFSS only when a dedicated PFSS channel is present and the caller supplied a long-lived executor.
  bool can_async_pfss = (ctx->pfss_net_chan != nullptr) && !created_local_pe;
  ctx->allow_async_pfss = can_async_pfss ? true : false;
  pe->set_keep_batches(true);
  if (ctx->force_eager_pfss) {
    pe->set_lazy_mode(false);
  } else {
    pe->set_lazy_mode(true);
  }
  runtime::PfssLayerPlanner::Limits layer_lim;
  layer_lim.max_coeff_jobs = 1ull << 15;
  layer_lim.max_trunc_jobs = 1ull << 15;
  layer_lim.max_coeff_hatx_words = 1ull << 21;
  layer_lim.max_trunc_hatx_words = 1ull << 21;
  layer_lim.max_coeff_hatx_bytes = layer_lim.max_coeff_hatx_words * sizeof(uint64_t);
  layer_lim.max_trunc_hatx_bytes = layer_lim.max_trunc_hatx_words * sizeof(uint64_t);
  layer_lim.max_coeff_flushes = 1ull << 9;
  layer_lim.max_trunc_flushes = 1ull << 9;
  runtime::PfssLayerPlanner layer_planner;
  runtime::PfssLayerPlanner* layer_planner_ptr = ctx->pfss_layer_planner;
  if (layer_planner_ptr == nullptr) {
    layer_planner.set_limits(layer_lim);
    layer_planner_ptr = &layer_planner;
  }
  runtime::PfssLayerPlanner* prev_layer_planner = ctx->pfss_layer_planner;
  ctx->pfss_layer_planner = layer_planner_ptr;
  // Share a single PFSS batch for coeff+trunc to maximize fusion.
  if (ctx->pfss_batch == nullptr) ctx->pfss_batch = &pe->pfss_coeff_batch();
  ctx->enable_hoist = true;
  ctx->open_collector = &pe->open_collector();
  proto::PfssBackendBatch& backend = ctx->trunc_backend();
  net::Chan* pfss_nc = (ctx && ctx->pfss_net_chan) ? ctx->pfss_net_chan : &ch;
  // Apply conservative limits to PFSS batches and opens to avoid runaway buffering.
  runtime::PfssSuperBatch::Limits pfss_lim;
  pfss_lim.max_pending_jobs = 1ull << 12;
  pfss_lim.max_pending_hatx_words = 1ull << 20;
  pfss_lim.max_pending_hatx_bytes = pfss_lim.max_pending_hatx_words * sizeof(uint64_t);
  pfss_lim.max_flushes = 1ull << 10;
  if (ctx && ctx->uses_gpu_backend()) {
    pfss_lim.max_pending_jobs = 1ull << 15;
    pfss_lim.max_pending_hatx_words = 1ull << 22;
    pfss_lim.max_pending_hatx_bytes = pfss_lim.max_pending_hatx_words * sizeof(uint64_t);
    pfss_lim.max_pending_device_bytes = pfss_lim.max_pending_hatx_bytes;
  }
  pe->pfss_coeff_batch().set_limits(pfss_lim);
  pe->pfss_trunc_batch().set_limits(pfss_lim);
  runtime::OpenCollector::Limits open_lim;
  open_lim.max_pending_words = 1ull << 22;
  pe->open_collector().set_limits(open_lim);
  runtime::PhaseExecutor::LazyLimits lazy_lim;
  lazy_lim.open_pending_words = open_lim.max_pending_words;
  lazy_lim.coeff_pending_jobs = pfss_lim.max_pending_jobs;
  lazy_lim.trunc_pending_jobs = pfss_lim.max_pending_jobs;
  lazy_lim.hatx_pending_words = pfss_lim.max_pending_hatx_words;
  pe->set_lazy_limits(lazy_lim);
  pe->set_max_flushes(1ull << 11);

  size_t B = X_share.shape[0];
  size_t T = X_share.shape[1];
  size_t D = cfg.attn.D;
  assert(X_share.shape[2] == D);

  // Phase resources shared across tasks.
  runtime::PhaseResources R{};
  runtime::ProtoChanFromNet pch(*pfss_nc);
  R.party = party;
  R.net_chan = &ch;
  R.pfss_backend = &backend;
  R.pfss_chan = &pch;
  R.pfss_coeff = &pe->pfss_coeff_batch();
  R.pfss_trunc = &pe->pfss_coeff_batch();  // share batch for trunc + coeff
  R.opens = &pe->open_collector();
  runtime::PfssPhasePlanner phase_planner;
  if (!(ctx && ctx->force_eager_pfss)) {
    R.pfss_planner = &phase_planner;
  }
  if (layer_planner_ptr) {
    layer_planner_ptr->begin_layer();
  }
  bool restore_disable = false;
  runtime::PfssLayerPlanner::BarrierPolicy attn_barrier{.drain_all = true};
  runtime::PfssLayerPlanner::BarrierPolicy ln_barrier{.drain_all = true};
  if (ctx && !ctx->disable_inner_barriers) {
    restore_disable = true;
    // Enable a finer super-plan: keep batches alive, but insert explicit
    // barriers after LN1/QKV-Softmax/Out and after LN2/MLP instead of only at layer end.
    ctx->disable_inner_barriers = true;
  }
  bool allow_barriers = (!ctx) || (!ctx->disable_inner_barriers) || restore_disable;
  auto record_phase_plan = [&](runtime::PfssPhasePlanner& planner) {
    if (layer_planner_ptr) {
      const auto& st = planner.stats();
      if (st.coeff_jobs == 0 && st.trunc_jobs == 0 && st.coeff_flushes == 0 && st.trunc_flushes == 0) return;
      layer_planner_ptr->record_phase(planner, pe->pfss_coeff_batch(), pe->pfss_trunc_batch());
      pe->pfss_coeff_batch().reset_stats();
      pe->pfss_trunc_batch().reset_stats();
    }
  };
  phase_planner.bind(R.pfss_coeff, R.pfss_trunc);
  R.pfss_planner = &phase_planner;
  auto drain_barrier = [&](const runtime::PfssLayerPlanner::BarrierPolicy& pol) {
    if (layer_planner_ptr) {
      runtime::ProtoChanFromNet pch_bar(*pfss_nc);
      layer_planner_ptr->barrier(
          party, backend, pe->pfss_coeff_batch(), pe->pfss_trunc_batch(), pch_bar, R.opens, R.net_chan, pol);
    }
  };

  int rows = static_cast<int>(B * T);
  int cols = static_cast<int>(D);
  int fb = cfg.frac_bits;
  double eps = 1.0 / 1024.0;
  int rsqrt_iters = 1;

  // Shared trunc bundles for LN.
  std::mt19937_64 rng_trunc(0x6c6e7472ull);
  compiler::GateParams p_faithful;
  p_faithful.kind = compiler::GateKind::FaithfulARS;
  p_faithful.frac_bits = fb;
  p_faithful.per_element_masks = true;
  compiler::GateParams p_gap = p_faithful;
  p_gap.kind = compiler::GateKind::GapARS;

  auto mean_faithful = compiler::lower_truncation_gate(backend, rng_trunc, p_faithful, static_cast<size_t>(rows));
  compiler::GateParams p_var_faithful = p_faithful;
  p_var_faithful.frac_bits = 2 * fb;
  auto var_faithful = compiler::lower_truncation_gate(backend, rng_trunc, p_var_faithful, static_cast<size_t>(rows));
  auto norm_faithful = compiler::lower_truncation_gate(
      backend, rng_trunc, p_faithful, static_cast<size_t>(rows) * static_cast<size_t>(cols));
  auto mean_gap = compiler::lower_truncation_gate(backend, rng_trunc, p_gap, static_cast<size_t>(rows));
  compiler::GateParams p_var_gap = p_gap;
  p_var_gap.frac_bits = 2 * fb;
  auto var_gap = compiler::lower_truncation_gate(backend, rng_trunc, p_var_gap, static_cast<size_t>(rows));
  auto norm_gap = compiler::lower_truncation_gate(
      backend, rng_trunc, p_gap, static_cast<size_t>(rows) * static_cast<size_t>(cols));

  runtime::TruncChoice mean_choice{nullptr, &mean_faithful, fb, true};
  runtime::TruncChoice var_choice{nullptr, &var_faithful, 2 * fb, true};
  runtime::TruncChoice norm_choice{nullptr, &norm_faithful, fb, true};

  uint64_t inv_len_qf =
      static_cast<uint64_t>(std::llround((1.0 / static_cast<double>(cols)) * std::ldexp(1.0, fb)));
  uint64_t eps_qf =
      (party == 0) ? static_cast<uint64_t>(std::llround(eps * std::ldexp(1.0, fb))) : 0ull;

  static CachedRowBroadcastTripleProvider row_triples_p0(0);
  static CachedRowBroadcastTripleProvider row_triples_p1(1);
  auto& row_triples = (party == 0) ? row_triples_p0 : row_triples_p1;

  static std::unordered_map<uint64_t, std::vector<proto::BeaverTriple64Share>> ln_triple_cache_p0;
  static std::unordered_map<uint64_t, std::vector<proto::BeaverTriple64Share>> ln_triple_cache_p1;
  auto& ln_triple_cache = (party == 0) ? ln_triple_cache_p0 : ln_triple_cache_p1;
  size_t ln_elems = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  auto& ln1_triples = ensure_cached_triples(ln_triple_cache, 0x6c6e316eull, ln_elems, party);
  auto& ln2_triples = ensure_cached_triples(ln_triple_cache, 0x6c6e326eull, ln_elems, party);

  std::mt19937_64 rng_rsqrt1(0x6c6e7273ull);
  auto rsqrt_mat1 = make_rsqrt_material(
      backend, fb, rsqrt_iters, eps, /*vmax=*/16.0, rng_rsqrt1, rows);
  ensure_beaver_triples(rsqrt_mat1.keys,
                        static_cast<size_t>(rows) * (3 * static_cast<size_t>(rsqrt_mat1.nr_iters) + 1),
                        rng_rsqrt1);
  rsqrt_mat1.keys.k0.r_in_share_vec.assign(static_cast<size_t>(rows), rsqrt_mat1.keys.k0.r_in_share);
  rsqrt_mat1.keys.k1.r_in_share_vec.assign(static_cast<size_t>(rows), rsqrt_mat1.keys.k1.r_in_share);
  auto rsqrt_bundle1 = make_rsqrt_bundle(rsqrt_mat1);

  std::mt19937_64 rng_rsqrt2(0x6c6e7274ull);
  auto rsqrt_mat2 = make_rsqrt_material(
      backend, fb, rsqrt_iters, eps, /*vmax=*/16.0, rng_rsqrt2, rows);
  ensure_beaver_triples(rsqrt_mat2.keys,
                        static_cast<size_t>(rows) * (3 * static_cast<size_t>(rsqrt_mat2.nr_iters) + 1),
                        rng_rsqrt2);
  rsqrt_mat2.keys.k0.r_in_share_vec.assign(static_cast<size_t>(rows), rsqrt_mat2.keys.k0.r_in_share);
  rsqrt_mat2.keys.k1.r_in_share_vec.assign(static_cast<size_t>(rows), rsqrt_mat2.keys.k1.r_in_share);
  auto rsqrt_bundle2 = make_rsqrt_bundle(rsqrt_mat2);

  auto build_ln_bundle = [&](const runtime::RsqrtTaskBundle& rsqrt_bundle,
                             std::span<const proto::BeaverTriple64Share> mul_triples) {
    runtime::LayerNormTaskBundle b{};
    b.mean_trunc = mean_choice;
    b.var_trunc = var_choice;
    b.norm_trunc = norm_choice;
    b.rsqrt = rsqrt_bundle;
    b.inv_len_qf = inv_len_qf;
    b.eps_qf = eps_qf;
    b.frac_bits = fb;
    b.mul_triples = mul_triples;
    b.row_triples = &row_triples;
    // Range hints: inputs are Qf. Mean stays within input range; variance is non-negative
    // and in Q2f before trunc; norm is roughly unit-scale.
    compiler::RangeInterval in_range = compiler::RangeInterval::whole(true);
    if (ctx) {
      // If the caller recorded a tensor range for the LN input, reuse it.
      in_range = (ctx->graph.tensors().empty())
                     ? compiler::RangeInterval::whole(true)
                     : ctx->graph.tensors().back().range;
    }
    b.mean_range = compiler::shift_down(in_range, fb);  // sum*invL is already Qf before trunc
    b.var_range = compiler::RangeInterval::whole(true);
    b.var_range.lo = 0;
    b.var_range.is_signed = true;
    // Conservatively bound variance by |x|^2 * cols.
    uint64_t abs_lo = (in_range.lo < 0) ? static_cast<uint64_t>(-in_range.lo) : 0ull;
    uint64_t abs_hi = (in_range.hi < 0) ? static_cast<uint64_t>(-in_range.hi) : static_cast<uint64_t>(in_range.hi);
    uint64_t abs_max = std::max(abs_lo, abs_hi);
    __int128 var_bound = static_cast<__int128>(abs_max) * static_cast<__int128>(abs_max);
    var_bound = var_bound >> fb;  // one downscale from square
    var_bound *= static_cast<int64_t>(cols);
    int64_t vb = static_cast<int64_t>(std::min<__int128>(var_bound, std::numeric_limits<int64_t>::max()));
    b.var_range.hi = vb;
    b.norm_range = compiler::RangeInterval::whole(true);
    b.norm_range.lo = -(static_cast<int64_t>(1) << fb);
    b.norm_range.hi = static_cast<int64_t>(1) << fb;
    b.norm_range.is_signed = true;
    return b;
  };

  std::vector<uint64_t> ln1(B * T * D, 0);
  pe->begin_phase(runtime::PhaseExecutor::Phase::kLN1);
  if (layer_planner_ptr) layer_planner_ptr->enter_phase();
  auto ln1_bundle = build_ln_bundle(
      rsqrt_bundle1,
      std::span<const proto::BeaverTriple64Share>(ln1_triples.data(), ln1_triples.size()));
  pe->add_task(std::make_unique<runtime::LayerNormTask>(
      ln1_bundle,
      std::span<const uint64_t>(X_share.data, X_share.numel()),
      std::span<uint64_t>(ln1.data(), ln1.size()),
      rows,
      cols));
  pe->run(R);
  // Barrier after LN1 to keep qkv/softmax PFSS batched with subsequent phases but allow planner limits.
  if (allow_barriers) {
    drain_barrier(ln_barrier);
    record_phase_plan(phase_planner);
  }

  // Record a clamp for LN1 output to tighten ranges for downstream attention.
  if (ctx) {
    compiler::RangeInterval ln_range = clamp_layernorm_range(fb);
    SecretTensor ln1_t =
        make_secret_tensor(ctx, view3(ln1.data(), B, T, D), make_scale(fb, true));
    (void)record_clamp(ctx, ln1_t, ln_range, ln1_t.scale);
  }

  // Attention (already task-based, no local shifts).
  pe->begin_phase(runtime::PhaseExecutor::Phase::kQKV_Score);
  if (layer_planner_ptr) layer_planner_ptr->enter_phase();
  std::vector<uint64_t> attn_out(B * T * D, 0);
  attention_forward(cfg.attn,
                    party,
                    ch,
                    view3(ln1.data(), B, T, D),
                    Wqkv_public,
                    Wout_public,
                    cache,
                    view3(attn_out.data(), B, T, D),
                    ctx,
                    pe);

  // Residual add
  for (size_t i = 0; i < attn_out.size(); ++i) {
    attn_out[i] = proto::add_mod(attn_out[i], X_share.data[i]);
  }
  // Barrier after attention region to drain PFSS/Open if planner is present.
  if (allow_barriers) {
    drain_barrier(attn_barrier);
    record_phase_plan(phase_planner);
  }

  // LayerNorm 2 via task.
  std::vector<uint64_t> ln2(attn_out.size(), 0);
  pe->begin_phase(runtime::PhaseExecutor::Phase::kLN2_MLP);
  if (layer_planner_ptr) layer_planner_ptr->enter_phase();
  auto ln2_bundle = build_ln_bundle(
      rsqrt_bundle2,
      std::span<const proto::BeaverTriple64Share>(ln2_triples.data(), ln2_triples.size()));
  pe->add_task(std::make_unique<runtime::LayerNormTask>(
      ln2_bundle,
      std::span<const uint64_t>(attn_out.data(), attn_out.size()),
      std::span<uint64_t>(ln2.data(), ln2.size()),
      rows,
      cols));
  pe->run(R);
  if (allow_barriers) {
    drain_barrier(ln_barrier);
    record_phase_plan(phase_planner);
  }

  // Clamp LN2 output to feed tighter ranges into the MLP.
  if (ctx) {
    compiler::RangeInterval ln_range = clamp_layernorm_range(fb);
    SecretTensor ln2_t =
        make_secret_tensor(ctx, view3(ln2.data(), B, T, D), make_scale(fb, true));
    (void)record_clamp(ctx, ln2_t, ln_range, ln2_t.scale);
  }

  // MLP (already task-based for trunc/rescale).
  mlp_forward(cfg.mlp,
              view3(ln2.data(), B, T, D),
              W1_public,
              W2_public,
              Y_share,
              party,
              ch,
              ctx,
              pe);
  // Barrier after MLP region before final residual/flush.
  if (allow_barriers) {
    drain_barrier(attn_barrier);
    record_phase_plan(phase_planner);
  }

  // Residual add
  for (size_t i = 0; i < Y_share.numel(); ++i) {
    Y_share.data[i] = proto::add_mod(Y_share.data[i], attn_out[i]);
  }

  // Final safety flush and layer-level accounting for PFSS.
  if (layer_planner_ptr) {
    runtime::ProtoChanFromNet pch_layer(*pfss_nc);
    runtime::PfssAsyncRunner local_async_runner;
    runtime::PfssAsyncRunner* async_runner = &local_async_runner;
    if (ctx->allow_async_pfss) {
      if (!ctx->pfss_async_runner) ctx->pfss_async_runner = std::make_unique<runtime::PfssAsyncRunner>();
      async_runner = ctx->pfss_async_runner.get();
    }
    static std::shared_ptr<std::mutex> pfss_chan_mu = std::make_shared<std::mutex>();
    bool wait = !(ctx && ctx->allow_async_pfss);
    layer_planner_ptr->finalize_layer(
        party,
        backend,
        pe->pfss_coeff_batch(),
        pe->pfss_trunc_batch(),
        pch_layer,
        async_runner,
        wait,
        pfss_chan_mu.get());
    if (party == 0) {
      const auto& totals = layer_planner_ptr->totals();
      std::cerr << "[pfss-layer] phases=" << totals.phases
                << " coeff_jobs=" << totals.coeff_jobs
                << " trunc_jobs=" << totals.trunc_jobs
                << " coeff_flushes=" << totals.coeff_flushes
                << " trunc_flushes=" << totals.trunc_flushes
                << " coeff_hatx=" << totals.coeff_hatx_words
                << " trunc_hatx=" << totals.trunc_hatx_words << "\n";
    }
  } else {
    // No layer planner: flush any pending batches once at layer end.
    runtime::ProtoChanFromNet pch_final(*pfss_nc);
    pe->finalize_pfss_once(party, backend, pch_final);
    if (pe->open_collector().has_pending()) {
      pe->open_collector().flush(party, ch);
    }
  }
  bool flush_pfss = !(ctx && ctx->allow_async_pfss);
  finalize_layer(*ctx, party, ch, backend, flush_pfss);
  if (ctx && restore_disable) ctx->disable_inner_barriers = false;
  ctx->pfss_layer_planner = prev_layer_planner;
}

}  // namespace nn
