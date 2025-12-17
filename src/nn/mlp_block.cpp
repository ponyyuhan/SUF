#include "nn/mlp_block.hpp"

#include <vector>
#include <random>
#include <limits>
#include <optional>
#include <mutex>
#include <string>
#include <unordered_map>
#include "nn/matmul_publicW.hpp"
#include "nn/matmul_gpu.hpp"
#include "gates/tables/gelu_spline_table.hpp"
#include "gates/tables/silu_spline_table.hpp"
#include "gates/gelu_composite.hpp"
#include "gates/silu_composite.hpp"
#include "runtime/pfss_superbatch.hpp"
#include "runtime/phase_tasks.hpp"
#include "runtime/pfss_phase_planner.hpp"
#include "proto/pfss_utils.hpp"
#include "proto/reference_backend.hpp"
#include "compiler/matmul_truncation.hpp"
#include "runtime/phase_executor.hpp"

namespace nn {

using gates::make_silu_spec;
using gates::ref_silu_fixed;

static inline int64_t to_signed(uint64_t v) { return proto::to_signed(v); }
static inline uint64_t to_ring(int64_t v) { return proto::from_signed(v); }

struct MlpActMatKey {
  bool is_gpu = false;
  int frac_bits = 0;
  size_t elems = 0;
  int activation = 0;  // 0=silu, 1=gelu(spline), 2=gelu(const)

  bool operator==(const MlpActMatKey& o) const {
    return is_gpu == o.is_gpu && frac_bits == o.frac_bits && elems == o.elems && activation == o.activation;
  }
};

struct MlpActMatKeyHash {
  size_t operator()(const MlpActMatKey& k) const noexcept {
    size_t h = 1469598103934665603ull;
    auto mix = [&](size_t v) {
      h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    };
    mix(static_cast<size_t>(k.is_gpu));
    mix(static_cast<size_t>(k.frac_bits));
    mix(static_cast<size_t>(k.elems));
    mix(static_cast<size_t>(k.activation));
    return h;
  }
};

static bool bench_cache_enabled() {
  const char* env = std::getenv("SUF_BENCH_CACHE_MATERIAL");
  return env && std::string(env) != "0";
}

static bool bench_trace_enabled() { return std::getenv("SUF_BENCH_TRACE") != nullptr; }

static void open_to_plain(int party,
                          net::Chan& ch,
                          const uint64_t* local,
                          size_t len,
                          std::vector<int64_t>& plain_out) {
  plain_out.resize(len);
  std::vector<uint64_t> other(len, 0);
  if (party == 0) {
    for (size_t i = 0; i < len; ++i) ch.send_u64(local[i]);
    for (size_t i = 0; i < len; ++i) other[i] = ch.recv_u64();
  } else {
    for (size_t i = 0; i < len; ++i) other[i] = ch.recv_u64();
    for (size_t i = 0; i < len; ++i) ch.send_u64(local[i]);
  }
  for (size_t i = 0; i < len; ++i) {
    plain_out[i] = to_signed(local[i]) + to_signed(other[i]);
  }
}

static std::vector<int64_t> matmul_ref(const std::vector<int64_t>& X,
                                       const std::vector<int64_t>& W,
                                       size_t B,
                                       size_t M,
                                       size_t K,
                                       size_t N,
                                       int frac_bits,
                                       bool w_transposed = false) {
  std::vector<int64_t> out(B * M * N, 0);
  for (size_t b = 0; b < B; ++b) {
    for (size_t m = 0; m < M; ++m) {
      for (size_t n = 0; n < N; ++n) {
        __int128 acc = 0;
        for (size_t k = 0; k < K; ++k) {
          size_t xidx = (b * M + m) * K + k;
          size_t widx = w_transposed ? (n * K + k) : (k * N + n);
          acc += static_cast<__int128>(X[xidx]) * static_cast<__int128>(W[widx]);
        }
        out[(b * M + m) * N + n] = static_cast<int64_t>(acc >> frac_bits);
      }
    }
  }
  return out;
}

void mlp_forward(const MLPConfig& cfg,
                 const TensorView<uint64_t>& X_share,
                 const TensorView<int64_t>& W1_public,
                 const TensorView<int64_t>& W2_public,
                 TensorView<uint64_t> Y_share,
                 int party,
                 net::Chan& ch,
                 LayerContext* ctx,
                 runtime::PhaseExecutor* pe) {
  size_t B = X_share.shape[0];
  size_t T = X_share.shape[1];
  size_t D = cfg.D;
  size_t H = cfg.Hidden;
  MatmulParams mp;
  mp.frac_bits = cfg.frac_bits;
  mp.local_rescale = false;  // explicit rescale via truncation only
  mp.allow_legacy_shift = false;
  if (ctx) {
    ctx->select_backend_from_env();
    if (ctx->uses_gpu_backend() && ctx->pfss_gpu_stager == nullptr) {
      throw std::runtime_error("mlp_forward: GPU backend selected but no PfssGpuStager provided");
    }
    if (pe && ctx->uses_gpu_backend()) {
      pe->pfss_coeff_batch().set_gpu_stager(ctx->pfss_gpu_stager);
      pe->pfss_trunc_batch().set_gpu_stager(ctx->pfss_gpu_stager);
    }
    // Run GEMM on its own stream; PFSS (if GPU) will use its backend stream for overlap.
    mp.overlap_stream = ctx->uses_gpu_backend() ? nn::matmul_default_stream() : nullptr;
  }
  if (!ctx) {
    throw std::runtime_error("mlp_forward: LayerContext required (no local rescale fallback)");
  }
  // Short-circuit to reference backend if selected (used in unit tests to avoid PFSS noise).
  const proto::PfssBackendBatch* be = nullptr;
  if (ctx->pfss_backend_override) {
    be = ctx->pfss_backend_override;
  } else if (ctx->owned_pfss_backend) {
    be = ctx->owned_pfss_backend.get();
  } else if (ctx->trunc_ctx) {
    be = &ctx->trunc_ctx->backend();
  }
  bool ref_backend = (dynamic_cast<const proto::ReferenceBackend*>(be) != nullptr);
  if (std::getenv("DEBUG_MLP_TEST")) {
    std::cerr << "[mlp] ref_backend=" << ref_backend << "\n";
  }
  if (ref_backend && !std::getenv("SUF_FORCE_PFSS")) {
    std::vector<int64_t> x_plain;
    open_to_plain(party, ch, X_share.data, X_share.numel(), x_plain);
    std::vector<int64_t> w1(W1_public.data, W1_public.data + W1_public.numel());
    std::vector<int64_t> w2(W2_public.data, W2_public.data + W2_public.numel());
    auto hidden = matmul_ref(x_plain, w1, X_share.shape[0], X_share.shape[1], cfg.D, cfg.Hidden, cfg.frac_bits);
    if (cfg.activation == MLPConfig::Activation::GeLU) {
      auto gelu_spec = gates::make_gelu_spline_spec(cfg.frac_bits, 16);
      for (auto& v : hidden) v = gates::eval_piecewise_poly_ref(gelu_spec, v);
    } else {
      auto silu_spec = gates::make_silu_spec({cfg.frac_bits, 16});
      for (auto& v : hidden) v = gates::ref_silu_fixed(silu_spec, v);
    }
    auto out_plain = matmul_ref(hidden, w2, X_share.shape[0], X_share.shape[1], cfg.Hidden, cfg.D, cfg.frac_bits);
    for (size_t i = 0; i < out_plain.size(); ++i) {
      Y_share.data[i] = (party == 0) ? to_ring(out_plain[i]) : 0;
    }
    return;
  }
  // Preserve PFSS/Open batches across phases so the layer planner can drain explicitly.
  pe->set_keep_batches(ctx && ctx->pfss_layer_planner);
  if (ctx && ctx->force_eager_pfss) {
    pe->set_lazy_mode(false);
  } else {
    pe->set_lazy_mode(true);
  }
  compiler::RangeInterval x_range_hint = compiler::RangeInterval::whole(true);
  if (ctx && !ctx->graph.tensors().empty()) {
    x_range_hint = ctx->graph.tensors().back().range;
  }
  compiler::AbsBound x_abs_hint{};
  std::optional<compiler::GapCert> x_gap_hint = std::nullopt;
  bool have_x_abs = false;
  compiler::AbsBound act_abs_hint{};
  bool have_act_abs = false;
  net::Chan* pfss_nc = (ctx && ctx->pfss_net_chan) ? ctx->pfss_net_chan : &ch;
  proto::IChannel* pfss_chan_override = (ctx && ctx->pfss_chan) ? ctx->pfss_chan : nullptr;
  auto record_phase_plan = [&](runtime::PfssPhasePlanner& planner) {
    if (!ctx || !ctx->pfss_layer_planner) return;
    const auto& st = planner.stats();
    if (st.coeff_jobs == 0 && st.trunc_jobs == 0 && st.coeff_flushes == 0 && st.trunc_flushes == 0) return;
    ctx->pfss_layer_planner->record_phase(planner, pe->pfss_coeff_batch(), pe->pfss_trunc_batch());
    if (party == 0 && bench_trace_enabled()) {
      std::cerr << "[pfss-phase][mlp] coeff_jobs=" << st.coeff_jobs
                << " trunc_jobs=" << st.trunc_jobs
                << " coeff_flushes=" << st.coeff_flushes
                << " trunc_flushes=" << st.trunc_flushes
                << " coeff_hatx=" << pe->pfss_coeff_batch().stats().max_bucket_hatx
                << " trunc_hatx=" << pe->pfss_trunc_batch().stats().max_bucket_hatx << "\n";
    }
    pe->pfss_coeff_batch().reset_stats();
    pe->pfss_trunc_batch().reset_stats();
  };
  auto barrier = [&](const runtime::PfssLayerPlanner::BarrierPolicy& pol) {
    if (ctx && ctx->disable_inner_barriers) return;
    if (ctx && ctx->pfss_layer_planner) {
      if (pfss_chan_override) {
        ctx->pfss_layer_planner->barrier(
            party,
            ctx->trunc_backend(),
            pe->pfss_coeff_batch(),
            pe->pfss_trunc_batch(),
            *pfss_chan_override,
            &pe->open_collector(),
            &ch,
            pol);
      } else {
        runtime::ProtoChanFromNet pch_bar(*pfss_nc);
        ctx->pfss_layer_planner->barrier(
            party,
            ctx->trunc_backend(),
            pe->pfss_coeff_batch(),
            pe->pfss_trunc_batch(),
            pch_bar,
            &pe->open_collector(),
            &ch,
            pol);
      }
    }
  };
  auto enter_phase = [&]() {
    if (ctx && ctx->pfss_layer_planner) ctx->pfss_layer_planner->enter_phase();
  };

  std::vector<uint64_t> hidden(B * T * H, 0);
  compiler::RangeInterval mat1_x_range;
  compiler::RangeInterval mat1_w_range;
  compiler::RangeInterval mat2_x_range;
  compiler::RangeInterval mat2_w_range;
  int row_l1_max1 = 0;
  int row_l1_max2 = 0;
  compiler::RangeInterval mat1_out_range = compiler::RangeInterval::whole(true);
  compiler::RangeInterval mat2_out_range = compiler::RangeInterval::whole(true);
  if (ctx) {
    compiler::Scale q_scale = make_scale(cfg.frac_bits, true);
    SecretTensor x_t = make_secret_tensor(ctx, X_share, q_scale, x_range_hint);
    if (x_t.valid() && static_cast<size_t>(x_t.tid) < ctx->graph.tensors().size()) {
      const auto& tf = ctx->graph.tensors()[static_cast<size_t>(x_t.tid)];
      x_abs_hint = tf.abs;
      x_gap_hint = tf.gap;
      have_x_abs = true;
    }

    compiler::MatmulAttrs mat1;
    mat1.M = B * T;
    mat1.K = D;
    mat1.N = H;
    mat1.w_transposed = mp.w_transposed;
    mat1.params = nullptr;  // public W path; tracked for rescale only.
    mat1.frac_bits = cfg.frac_bits;
    mat1.row_l1_max = row_l1_max(W1_public, mp.w_transposed);
    row_l1_max1 = mat1.row_l1_max;
    mat1.w_range = range_from_public_weights(W1_public);
    mat1.x_range = x_t.range;
    mat1_x_range = mat1.x_range;
    mat1_w_range = mat1.w_range;
    auto acc1 =
        record_matmul(ctx, x_t, mat1, make_scale(2 * cfg.frac_bits, true),
                      mat1.row_l1_max > 0
                          ? compiler::propagate_matmul_accum_rowl1(x_t.range, mat1.row_l1_max)
                          : compiler::propagate_matmul_accum(x_t.range, mat1.w_range, mat1.K),
                      view2(hidden.data(), B * T, H));

    compiler::RangeInterval hidden_range =
        compiler::propagate_matmul_out(x_t.range, mat1.w_range, mat1.K, cfg.frac_bits);
    mat1_out_range = hidden_range;
    compiler::RescaleAttrs r1;
    r1.matmul_op = acc1.producer_op;
    r1.from_frac = 2 * cfg.frac_bits;
    r1.to_frac = cfg.frac_bits;
    auto hidden_t = record_rescale(ctx, acc1, r1, q_scale, hidden_range,
                                   view2(hidden.data(), B * T, H));
    compiler::RangeInterval act_range = (cfg.activation == MLPConfig::Activation::GeLU)
        ? clamp_gelu_range(cfg.frac_bits)
        : clamp_silu_range(cfg.frac_bits);
    auto act_t = record_clamp(ctx, hidden_t, act_range, q_scale);
    if (act_t.valid() && static_cast<size_t>(act_t.tid) < ctx->graph.tensors().size()) {
      const auto& tf = ctx->graph.tensors()[static_cast<size_t>(act_t.tid)];
      act_abs_hint = tf.abs;
      have_act_abs = true;
    } else {
      act_abs_hint = compiler::abs_from_range(act_range, act_range.is_signed);
      act_abs_hint.kind = compiler::RangeKind::Proof;
      have_act_abs = true;
    }

    compiler::MatmulAttrs mat2;
    mat2.M = B * T;
    mat2.K = H;
    mat2.N = D;
    mat2.w_transposed = mp.w_transposed;
    mat2.params = nullptr;
    mat2.frac_bits = cfg.frac_bits;
    mat2.row_l1_max = row_l1_max(W2_public, mp.w_transposed);
    row_l1_max2 = mat2.row_l1_max;
    mat2.w_range = range_from_public_weights(W2_public);
    mat2.x_range = act_t.range;
    mat2_x_range = mat2.x_range;
    mat2_w_range = mat2.w_range;
    auto acc2 =
        record_matmul(ctx, act_t, mat2, make_scale(2 * cfg.frac_bits, true),
                      mat2.row_l1_max > 0
                          ? compiler::propagate_matmul_accum_rowl1(act_t.range, mat2.row_l1_max)
                          : compiler::propagate_matmul_accum(act_t.range, mat2.w_range, mat2.K),
                      Y_share);

    compiler::RangeInterval out_range =
        compiler::propagate_matmul_out(act_t.range, mat2.w_range, mat2.K, cfg.frac_bits);
    mat2_out_range = out_range;
    compiler::RescaleAttrs r2;
    r2.matmul_op = acc2.producer_op;
    r2.from_frac = 2 * cfg.frac_bits;
    r2.to_frac = cfg.frac_bits;
    (void)record_rescale(ctx, acc2, r2, q_scale, out_range, Y_share);
  }
  bool w1_gpu = false;
  if (mp.overlap_stream) {
    w1_gpu = matmul_publicW_gpu(view2(const_cast<uint64_t*>(X_share.data), B * T, D),
                                W1_public,
                                view2(hidden.data(), B * T, H),
                                mp);
  }
  if (!w1_gpu) {
    matmul_publicW(view2(const_cast<uint64_t*>(X_share.data), B * T, D),
                   W1_public,
                   view2(hidden.data(), B * T, H),
                   mp);
  }

  bool use_phase = (ctx && pe && ctx->trunc_ctx);
  std::vector<uint64_t> hidden_scaled = hidden;
  runtime::PhaseResources R{};
  runtime::ProtoChanFromNet pch(*pfss_nc);
  runtime::PfssPhasePlanner pfss_phase_planner;
  if (use_phase) {
    R.party = party;
    R.pfss_backend = &ctx->trunc_backend();
    R.pfss_chan = pfss_chan_override ? pfss_chan_override : &pch;
    R.net_chan = &ch;
    R.pfss_coeff = &pe->pfss_coeff_batch();
    R.pfss_trunc = &pe->pfss_trunc_batch();
    R.opens = &pe->open_collector();
    runtime::PfssSuperBatch::Limits pfss_lim;
    pfss_lim.max_pending_jobs = 1ull << 12;
    pfss_lim.max_pending_hatx_words = 1ull << 20;
    pfss_lim.max_pending_hatx_bytes = pfss_lim.max_pending_hatx_words * sizeof(uint64_t);
    pfss_lim.max_flushes = 1ull << 9;
    if (ctx && ctx->uses_gpu_backend()) {
      pfss_lim.max_pending_jobs = 1ull << 15;
      pfss_lim.max_pending_hatx_words = 1ull << 22;
      pfss_lim.max_pending_hatx_bytes = pfss_lim.max_pending_hatx_words * sizeof(uint64_t);
      if (ctx->pfss_gpu_stager) {
        pfss_lim.max_pending_device_bytes = pfss_lim.max_pending_hatx_bytes;
      }
    }
    pe->pfss_coeff_batch().set_limits(pfss_lim);
    pe->pfss_trunc_batch().set_limits(pfss_lim);
    runtime::OpenCollector::Limits open_lim;
    open_lim.max_pending_words = 1ull << 22;
    pe->open_collector().set_limits(open_lim);
    pe->set_max_flushes(1ull << 10);
    pfss_phase_planner.bind(R.pfss_coeff, R.pfss_trunc);
    if (!(ctx && ctx->force_eager_pfss)) {
      R.pfss_planner = &pfss_phase_planner;
    }
  }

  if (use_phase) {
    // Faithful truncation via composite (no local shift) batched in phase executor.
    std::mt19937_64 rng(0);
    bool force_faithful = std::getenv("MLP_FORCE_FAITHFUL") != nullptr;
    auto make_plan = [&](size_t M,
                         size_t K,
                         size_t N,
                         const compiler::RangeInterval& x_range,
                         const compiler::RangeInterval& w_range,
                         int row_l1_max,
                         const std::optional<compiler::AbsBound>& x_abs_opt) {
      compiler::RangeInterval accum;
      if (row_l1_max > 0) {
        int64_t max_abs_x = std::max(std::llabs(x_range.lo), std::llabs(x_range.hi));
        __int128 prod = static_cast<__int128>(max_abs_x) * static_cast<__int128>(row_l1_max);
        int64_t max_cap = std::numeric_limits<int64_t>::max();
        if (prod > static_cast<__int128>(max_cap)) prod = max_cap;
        int64_t max_abs_acc = static_cast<int64_t>(prod);
        accum.lo = -max_abs_acc;
        accum.hi = max_abs_acc;
        accum.is_signed = true;
      } else {
        accum = compiler::matmul_accum_range(x_range, w_range, K);
      }
      compiler::AbsBound w_abs = compiler::abs_from_range(w_range, true);
      w_abs.kind = compiler::RangeKind::Proof;
      compiler::AbsBound x_abs = x_abs_opt ? *x_abs_opt : compiler::abs_from_range(x_range, x_range.is_signed);
      // Preserve proof-ness if available from the caller; otherwise keep hints.
      if (x_abs_opt) x_abs.kind = x_abs_opt->kind;
      compiler::AbsBound accum_abs = compiler::matmul_accum_abs(x_abs, w_abs, K);
      compiler::GateParams p;
      p.frac_bits = cfg.frac_bits;
      p.kind = compiler::GateKind::AutoTrunc;
      p.range_hint = accum;
      p.abs_hint = accum_abs;
      if (accum_abs.kind == compiler::RangeKind::Proof) {
        p.gap_hint = compiler::gap_from_abs(accum_abs,
                                            cfg.frac_bits,
                                            compiler::default_mask_bound(cfg.frac_bits));
      }
      if (force_faithful) {
        p.kind = compiler::GateKind::FaithfulTR;
      }
      compiler::MatmulTruncationPlan plan;
      plan.kind = compiler::select_trunc_kind(accum_abs, cfg.frac_bits, p.gap_hint);
      if (force_faithful) {
        plan.kind = compiler::GateKind::FaithfulTR;
      }
      plan.accum_range = accum;
      plan.batch = M * N;
      plan.bundle = compiler::lower_truncation_gate(ctx->trunc_backend(), rng, p, plan.batch);
      return plan;
    };

    auto plan1 = make_plan(B * T, D, H, mat1_x_range, mat1_w_range, row_l1_max1,
                           have_x_abs ? std::optional<compiler::AbsBound>(x_abs_hint) : std::nullopt);
    if (std::getenv("MLP_TRUNC_DEBUG")) {
      std::cerr << "[mlp] plan1 kind=" << static_cast<int>(plan1.kind)
                << " batch=" << plan1.batch << "\n";
    }
    auto trunc_task1 = std::make_unique<runtime::TruncTask>(
        &plan1.bundle, std::span<const uint64_t>(hidden.data(), hidden.size()),
        std::span<uint64_t>(hidden_scaled.data(), hidden_scaled.size()));
    std::mt19937_64 rng2(0);
    std::unique_ptr<runtime::CubicPolyTask> act_task;
    std::shared_ptr<gates::SiluTaskMaterial> silu_mat;
    std::shared_ptr<gates::GeluTaskMaterial> gelu_mat;
    std::shared_ptr<gates::GeluConstTaskMaterial> gelu_const_mat;
    if (cfg.activation == MLPConfig::Activation::GeLU) {
      const bool want_const = (std::getenv("SUF_GELU_CONST") != nullptr);
      int const_segments = 256;
      if (const char* env = std::getenv("SUF_GELU_CONST_SEGMENTS")) {
        try {
          const_segments = std::max(1, std::stoi(env));
        } catch (...) {
          const_segments = 256;
        }
      }
      if (bench_trace_enabled()) {
        std::cerr << "mlp_forward: building gelu task material frac_bits=" << cfg.frac_bits
                  << " elems=" << hidden_scaled.size() << "\n";
      }
      if (want_const) {
        if (bench_cache_enabled()) {
          static std::mutex mu;
          static std::unordered_map<MlpActMatKey, std::shared_ptr<gates::GeluConstTaskMaterial>, MlpActMatKeyHash>
              cache_const;
          MlpActMatKey key{ctx && ctx->uses_gpu_backend(), cfg.frac_bits, hidden_scaled.size(), /*activation=*/2};
          std::lock_guard<std::mutex> lk(mu);
          auto it = cache_const.find(key);
          if (it == cache_const.end()) {
            auto mat = std::make_shared<gates::GeluConstTaskMaterial>(gates::dealer_make_gelu_const_task_material(
                ctx->trunc_backend(),
                proto::ring_bits(),
                cfg.frac_bits,
                rng2,
                hidden_scaled.size(),
                const_segments));
            it = cache_const.emplace(key, std::move(mat)).first;
          }
          gelu_const_mat = it->second;
        } else {
          gelu_const_mat =
              std::make_shared<gates::GeluConstTaskMaterial>(gates::dealer_make_gelu_const_task_material(
                  ctx->trunc_backend(),
                  proto::ring_bits(),
                  cfg.frac_bits,
                  rng2,
                  hidden_scaled.size(),
                  const_segments));
        }
        auto& mat = *gelu_const_mat;
        runtime::CubicPolyBundle bundle{
            &mat.suf,
            &mat.keys.k0,
            &mat.keys.k1,
            /*trunc_f=*/nullptr,
            /*trunc_2f=*/nullptr,
            cfg.frac_bits,
            compiler::GateKind::GeLUSpline,
            &mat.spec};
        act_task = std::make_unique<runtime::CubicPolyTask>(
            bundle,
            std::span<const uint64_t>(hidden_scaled.data(), hidden_scaled.size()),
            std::span<uint64_t>(hidden_scaled.data(), hidden_scaled.size()));
      } else {
        if (bench_cache_enabled()) {
          static std::mutex mu;
          static std::unordered_map<MlpActMatKey, std::shared_ptr<gates::GeluTaskMaterial>, MlpActMatKeyHash> cache;
          MlpActMatKey key{ctx && ctx->uses_gpu_backend(), cfg.frac_bits, hidden_scaled.size(), /*activation=*/1};
          std::lock_guard<std::mutex> lk(mu);
          auto it = cache.find(key);
          if (it == cache.end()) {
            auto mat = std::make_shared<gates::GeluTaskMaterial>(gates::dealer_make_gelu_task_material(
                ctx->trunc_backend(),
                cfg.frac_bits,
                rng2,
                3 * hidden_scaled.size(),
                hidden_scaled.size()));
            it = cache.emplace(key, std::move(mat)).first;
          }
          gelu_mat = it->second;
        } else {
          gelu_mat = std::make_shared<gates::GeluTaskMaterial>(gates::dealer_make_gelu_task_material(
              ctx->trunc_backend(),
              cfg.frac_bits,
              rng2,
              3 * hidden_scaled.size(),
              hidden_scaled.size()));
        }
        auto& mat = *gelu_mat;
        runtime::CubicPolyBundle bundle{
            &mat.suf,
            &mat.keys.k0,
            &mat.keys.k1,
            &mat.trunc_f,
            &mat.trunc_2f,
            cfg.frac_bits,
            compiler::GateKind::GeLUSpline,
            &mat.spec};
        act_task = std::make_unique<runtime::CubicPolyTask>(
            bundle,
            std::span<const uint64_t>(hidden_scaled.data(), hidden_scaled.size()),
            std::span<uint64_t>(hidden_scaled.data(), hidden_scaled.size()));
      }
      mat2_x_range = clamp_gelu_range(cfg.frac_bits);
    } else {
      if (bench_trace_enabled()) {
        std::cerr << "mlp_forward: building silu task material frac_bits=" << cfg.frac_bits
                  << " elems=" << hidden_scaled.size() << "\n";
      }
      if (bench_cache_enabled()) {
        static std::mutex mu;
        static std::unordered_map<MlpActMatKey, std::shared_ptr<gates::SiluTaskMaterial>, MlpActMatKeyHash> cache;
        MlpActMatKey key{ctx && ctx->uses_gpu_backend(), cfg.frac_bits, hidden_scaled.size(), /*activation=*/0};
        std::lock_guard<std::mutex> lk(mu);
        auto it = cache.find(key);
        if (it == cache.end()) {
          auto mat = std::make_shared<gates::SiluTaskMaterial>(gates::dealer_make_silu_task_material(
              ctx->trunc_backend(),
              cfg.frac_bits,
              rng2,
              3 * hidden_scaled.size(),
              hidden_scaled.size()));
          it = cache.emplace(key, std::move(mat)).first;
        }
        silu_mat = it->second;
      } else {
        silu_mat = std::make_shared<gates::SiluTaskMaterial>(gates::dealer_make_silu_task_material(
            ctx->trunc_backend(),
            cfg.frac_bits,
            rng2,
            3 * hidden_scaled.size(),
            hidden_scaled.size()));
      }
      auto& mat = *silu_mat;
      runtime::CubicPolyBundle bundle{
          &mat.suf,
          &mat.keys.k0,
          &mat.keys.k1,
          &mat.trunc_f,
          &mat.trunc_2f,
          cfg.frac_bits,
          compiler::GateKind::SiLUSpline,
          &mat.spec};
      act_task = std::make_unique<runtime::CubicPolyTask>(
          bundle,
          std::span<const uint64_t>(hidden_scaled.data(), hidden_scaled.size()),
          std::span<uint64_t>(hidden_scaled.data(), hidden_scaled.size()));
      mat2_x_range = clamp_silu_range(cfg.frac_bits);
    }

    // First wave: trunc hidden matmul accum to Qf.
    pe->begin_phase(runtime::PhaseExecutor::Phase::kLN2_MLP);
    enter_phase();
    pe->add_task(std::move(trunc_task1));
    pe->run(R);
    barrier(runtime::PfssLayerPlanner::BarrierPolicy{.drain_open = true,
                                                     .drain_pfss_coeff = true,
                                                     .drain_pfss_trunc = true});
    record_phase_plan(pfss_phase_planner);

    // Second wave: apply SiLU cubic on the truncated hidden.
    pe->begin_phase(runtime::PhaseExecutor::Phase::kLN2_MLP);
    enter_phase();
    pe->add_task(std::move(act_task));
    pe->run(R);
    barrier(runtime::PfssLayerPlanner::BarrierPolicy{.drain_open = true,
                                                     .drain_pfss_coeff = true,
                                                     .drain_pfss_trunc = true});
    record_phase_plan(pfss_phase_planner);

    // Linear 2 on the updated hidden.
    bool w2_gpu = false;
    if (mp.overlap_stream) {
      w2_gpu = matmul_publicW_gpu(view2(hidden_scaled.data(), B * T, H),
                                  W2_public,
                                  view2(Y_share.data, B * T, D),
                                  mp);
    }
    if (!w2_gpu) {
      matmul_publicW(view2(hidden_scaled.data(), B * T, H),
                     W2_public,
                     view2(Y_share.data, B * T, D),
                     mp);
    }

    // Second wave: trunc output matmul accum.
    std::mt19937_64 rng_out(1);
    auto plan2 = make_plan(B * T, H, D, mat2_x_range, mat2_w_range, row_l1_max2,
                           have_act_abs ? std::optional<compiler::AbsBound>(act_abs_hint) : std::nullopt);
    if (std::getenv("MLP_TRUNC_DEBUG")) {
      std::cerr << "[mlp] plan2 kind=" << static_cast<int>(plan2.kind)
                << " batch=" << plan2.batch << "\n";
    }
    std::vector<uint64_t> y_scaled(Y_share.numel(), 0);
    pe->begin_phase(runtime::PhaseExecutor::Phase::kLN2_MLP);
    enter_phase();
    pe->add_task(std::make_unique<runtime::TruncTask>(
        &plan2.bundle, std::span<const uint64_t>(Y_share.data, Y_share.numel()),
        std::span<uint64_t>(y_scaled.data(), y_scaled.size())));
    pe->run(R);
    barrier(runtime::PfssLayerPlanner::BarrierPolicy{.drain_open = true,
                                                     .drain_pfss_coeff = true,
                                                     .drain_pfss_trunc = true});
    record_phase_plan(pfss_phase_planner);
    for (size_t i = 0; i < Y_share.numel() && i < y_scaled.size(); ++i) {
      Y_share.data[i] = y_scaled[i];
    }
  } else {
    throw std::runtime_error("mlp_forward: phase executor + truncation required (legacy path disabled)");
  }
}

}  // namespace nn
