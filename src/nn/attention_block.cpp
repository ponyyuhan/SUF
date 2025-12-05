#include "nn/attention_block.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <optional>
#include <random>
#include <vector>
#include "gates/nexp_gate.hpp"
#include "gates/reciprocal_gate.hpp"
#include "gates/reciprocal_composite.hpp"
#include "gates/nexp_composite.hpp"
#include "gates/softmax_composite.hpp"
#include "runtime/pfss_superbatch.hpp"
#include "runtime/phase_tasks.hpp"

namespace nn {

namespace {

inline int64_t to_signed(uint64_t v) { return static_cast<int64_t>(v); }
inline uint64_t to_ring(int64_t v) { return static_cast<uint64_t>(v); }

// Lightweight reciprocal task: opens a scalar (or small vector) and produces
// public reciprocal shares (party0 gets value, party1 gets 0). Uses PhaseExecutor
// waves so the open can batch with other tasks.
class ReciprocalTask : public runtime::detail::PhaseTask {
 public:
  ReciprocalTask(std::span<const uint64_t> in_share,
                 std::span<uint64_t> out_share,
                 const gates::PiecewisePolySpec* init_spec,
                 int frac_bits,
                 int nr_iters)
      : in_(in_share), out_(out_share), spec_(init_spec), frac_bits_(frac_bits), iters_(nr_iters) {
    if (in_.size() != out_.size()) throw std::runtime_error("ReciprocalTask: size mismatch");
    if (!spec_) throw std::runtime_error("ReciprocalTask: init spec null");
  }

  bool done() const override { return st_ == St::Done; }

  runtime::detail::Need step(runtime::PhaseResources& R) override {
    switch (st_) {
      case St::Open: {
        if (!R.net_chan) throw std::runtime_error("ReciprocalTask: net channel missing");
        if (R.opens) {
          h_ = R.opens->enqueue(std::vector<uint64_t>(in_.begin(), in_.end()));
          st_ = St::WaitOpen;
          return runtime::detail::Need::Open;
        }
        opened_.assign(in_.begin(), in_.end());
        st_ = St::Compute;
        [[fallthrough]];
      }
      case St::WaitOpen: {
        if (st_ == St::WaitOpen) {
          if (!R.opens) throw std::runtime_error("ReciprocalTask: no OpenCollector");
          if (!R.opens->ready(h_)) return runtime::detail::Need::Open;
          auto v = R.opens->view(h_);
          opened_.assign(v.begin(), v.end());
          st_ = St::Compute;
        }
        [[fallthrough]];
      }
      case St::Compute: {
        for (size_t i = 0; i < opened_.size(); ++i) {
          int64_t x = to_signed(static_cast<uint64_t>(opened_[i]));
          int64_t inv = gates::ref_reciprocal_fixed(*spec_, x, frac_bits_, iters_);
          // public result: party0 takes inv, party1 takes 0.
          out_[i] = (R.party == 0) ? to_ring(inv) : 0ull;
        }
        st_ = St::Done;
        return runtime::detail::Need::None;
      }
      case St::Done:
        return runtime::detail::Need::None;
    }
    return runtime::detail::Need::None;
  }

 private:
  enum class St { Open, WaitOpen, Compute, Done } st_ = St::Open;
  std::span<const uint64_t> in_;
  std::span<uint64_t> out_;
  const gates::PiecewisePolySpec* spec_ = nullptr;
  int frac_bits_ = 0;
  int iters_ = 1;
  std::vector<int64_t> opened_;
  runtime::OpenHandle h_{};
};

static inline void rescale_buffer(std::vector<uint64_t>& buf, int frac_bits) {
  if (frac_bits <= 0) return;
  for (auto& v : buf) {
    int64_t s = to_signed(v);
    v = to_ring(s >> frac_bits);
  }
}

static inline void rescale_view(const TensorView<uint64_t>& t, int frac_bits) {
  if (frac_bits <= 0) return;
  size_t n = t.numel();
  for (size_t i = 0; i < n; ++i) {
    int64_t s = to_signed(t.data[i]);
    t.data[i] = to_ring(s >> frac_bits);
  }
}

void open_to_plain(int party,
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

}  // namespace

void attention_forward(const AttentionConfig& cfg,
                       int party,
                       net::Chan& ch,
                       const TensorView<uint64_t>& X_share,
                       const TensorView<int64_t>& Wqkv_public,
                       const TensorView<int64_t>& Wout_public,
                       KVCache& cache,
                       TensorView<uint64_t> Y_share,
                       LayerContext* ctx,
                       runtime::PhaseExecutor* pe) {
  size_t B = X_share.shape[0];
  size_t T = X_share.shape[1];
  size_t D = cfg.D;
  size_t H = cfg.H;
  size_t Dh = cfg.Dh;
  int fb = cfg.frac_bits;
  runtime::OpenCollector* opens = pe ? &pe->open_collector() : nullptr;
  bool use_legacy_softmax = cfg.legacy_softmax || (ctx == nullptr);
  bool use_phase_softmax = !use_legacy_softmax && pe && ctx && ctx->trunc_ctx;

  std::vector<uint64_t> qkv(B * T * 3 * D, 0);

  if (ctx) {
    compiler::Scale q_scale = make_scale(fb, true);
    compiler::RangeInterval x_range = compiler::RangeInterval::whole(true);
    SecretTensor x_t = make_secret_tensor(ctx, X_share, q_scale, x_range);

    compiler::MatmulAttrs qkv_attrs;
    qkv_attrs.M = B * T;
    qkv_attrs.K = D;
    qkv_attrs.N = 3 * D;
    qkv_attrs.w_transposed = false;
    qkv_attrs.params = nullptr;
    qkv_attrs.frac_bits = fb;
    qkv_attrs.x_range = x_range;
    qkv_attrs.row_l1_max = row_l1_max(Wqkv_public, qkv_attrs.w_transposed);
    qkv_attrs.w_range = range_from_public_weights(Wqkv_public);
    auto qkv_acc = record_matmul(
        ctx, x_t, qkv_attrs, make_scale(2 * fb, true),
        qkv_attrs.row_l1_max > 0
            ? compiler::propagate_matmul_accum_rowl1(x_range, qkv_attrs.row_l1_max)
            : compiler::propagate_matmul_accum(x_range, qkv_attrs.w_range, qkv_attrs.K),
        view2(qkv.data(), B * T, 3 * D));

    compiler::RescaleAttrs qkv_rescale_attrs;
    qkv_rescale_attrs.matmul_op = qkv_acc.producer_op;
    qkv_rescale_attrs.from_frac = 2 * fb;
    qkv_rescale_attrs.to_frac = fb;
    compiler::RangeInterval qkv_range =
        compiler::propagate_matmul_out(x_range, qkv_attrs.w_range, qkv_attrs.K, fb);
    SecretTensor qkv_t =
        record_rescale(ctx, qkv_acc, qkv_rescale_attrs, q_scale, qkv_range,
                       view2(qkv.data(), B * T, 3 * D));

    compiler::MatmulAttrs out_attrs;
    out_attrs.M = B * T;
    out_attrs.K = D;
    out_attrs.N = D;
    out_attrs.w_transposed = false;
    out_attrs.params = nullptr;
    out_attrs.frac_bits = fb;
    out_attrs.x_range = qkv_t.range;  // conservative; attention stack clamps internally.
    out_attrs.row_l1_max = row_l1_max(Wout_public, out_attrs.w_transposed);
    out_attrs.w_range = range_from_public_weights(Wout_public);
    auto out_acc = record_matmul(
        ctx, qkv_t, out_attrs, make_scale(2 * fb, true),
        out_attrs.row_l1_max > 0
            ? compiler::propagate_matmul_accum_rowl1(qkv_t.range, out_attrs.row_l1_max)
            : compiler::propagate_matmul_accum(qkv_t.range, out_attrs.w_range, out_attrs.K),
        Y_share);

    compiler::RescaleAttrs out_rescale_attrs;
    out_rescale_attrs.matmul_op = out_acc.producer_op;
    out_rescale_attrs.from_frac = 2 * fb;
    out_rescale_attrs.to_frac = fb;
    compiler::RangeInterval out_range =
        compiler::propagate_matmul_out(qkv_t.range, out_attrs.w_range, out_attrs.K, fb);
    (void)record_rescale(ctx, out_acc, out_rescale_attrs, q_scale, out_range, Y_share);
  }

  assert(D == H * Dh);
  assert(Wqkv_public.shape[0] == D && Wqkv_public.shape[1] == 3 * D);
  assert(cache.B == B && cache.H == H && cache.Dh == Dh);

  MatmulParams mp;
  mp.frac_bits = fb;
  mp.w_transposed = false;
  mp.local_rescale = (ctx == nullptr);
  mp.allow_legacy_shift = (ctx == nullptr);

  matmul_publicW(view2(const_cast<uint64_t*>(X_share.data), B * T, D),
                 Wqkv_public,
                 view2(qkv.data(), B * T, 3 * D),
                 mp);
  if (!ctx && !mp.local_rescale) {
    rescale_buffer(qkv, fb);
  }

  std::vector<uint64_t> ctx_shares(B * T * H * Dh, 0);
  size_t init_len = cache.cur_len;

  gates::NExpGateParams nexp_params;
  nexp_params.frac_bits = fb;
  nexp_params.segments = 16;
  auto nexp_spec = gates::make_nexp_spec(nexp_params);
  auto recip_spec =
      gates::make_recip_affine_init_spec(fb, static_cast<double>(std::max(cache.S_max, T + init_len)));
  std::optional<gates::NexpTaskMaterial> nexp_mat;
  runtime::CubicPolyBundle nexp_bundle{};
  std::optional<gates::RecipTaskMaterial> recip_mat;
  runtime::RecipTaskBundle recip_bundle{};
  if (use_phase_softmax) {
    std::mt19937_64 rng(123);
    size_t triple_need = 3 * cache.S_max * B * H;
    nexp_mat = gates::dealer_make_nexp_task_material(ctx->trunc_ctx->backend(),
                                                     nexp_params,
                                                     rng,
                                                     triple_need);
    nexp_bundle = gates::make_nexp_cubic_bundle(*nexp_mat, fb);
    recip_mat = gates::dealer_make_recip_task_material(ctx->trunc_ctx->backend(), fb, /*nr_iters=*/1, rng);
    recip_bundle = gates::make_recip_bundle(*recip_mat);
  }

  int64_t inv_sqrt = static_cast<int64_t>(
      std::llround((1.0 / std::sqrt(static_cast<double>(Dh))) * std::ldexp(1.0, fb)));
  if (inv_sqrt == 0) inv_sqrt = 1;

  std::vector<uint64_t> stepK(B * H * Dh, 0), stepV(B * H * Dh, 0);
  for (size_t t = 0; t < T; ++t) {
    // Slice K/V for this token.
    for (size_t b = 0; b < B; ++b) {
      size_t base = (b * T + t) * 3 * D;
      const uint64_t* k_src = qkv.data() + base + D;
      const uint64_t* v_src = qkv.data() + base + 2 * D;
      for (size_t h = 0; h < H; ++h) {
        for (size_t d = 0; d < Dh; ++d) {
          size_t idx = (b * H + h) * Dh + d;
          stepK[idx] = k_src[h * Dh + d];
          stepV[idx] = v_src[h * Dh + d];
        }
      }
    }

    kv_append_token(cache, view3(stepK.data(), B, H, Dh), view3(stepV.data(), B, H, Dh));
    size_t cur_len = cache.cur_len;

    struct HeadState {
      size_t b = 0;
      size_t h = 0;
      size_t cur_len = 0;
      std::vector<int64_t> q_plain;
      std::vector<int64_t> k_plain;
      std::vector<int64_t> v_plain;
      std::vector<int64_t> scores;
      int64_t max_sc = 0;
      std::vector<uint64_t> diff_share;
      std::vector<uint64_t> exp_share;
      std::vector<int64_t> expv;
    };
    std::vector<HeadState> heads;
    if (use_phase_softmax) {
      pe->begin_phase(runtime::PhaseExecutor::Phase::kSoftmax);
    }

    for (size_t b = 0; b < B; ++b) {
      size_t q_base = (b * T + t) * 3 * D;
      const uint64_t* q_ptr = qkv.data() + q_base;
      for (size_t h = 0; h < H; ++h) {
        const uint64_t* k_head = kv_head_ptr(cache, b, h);
        const uint64_t* v_head = kv_head_ptr_v(cache, b, h);

        HeadState st;
        st.b = b;
        st.h = h;
        st.cur_len = cur_len;
        open_to_plain(party, ch, q_ptr + h * Dh, Dh, st.q_plain);
        open_to_plain(party, ch, k_head, cur_len * Dh, st.k_plain);
        open_to_plain(party, ch, v_head, cur_len * Dh, st.v_plain);

        st.scores.resize(cur_len, 0);
        for (size_t s = 0; s < cur_len; ++s) {
          __int128 acc = 0;
          for (size_t d = 0; d < Dh; ++d) {
            acc += static_cast<__int128>(st.q_plain[d]) * static_cast<__int128>(st.k_plain[s * Dh + d]);
          }
          acc >>= fb;
          acc = (acc * static_cast<__int128>(inv_sqrt)) >> fb;
          st.scores[s] = static_cast<int64_t>(acc);
        }
        st.max_sc = st.scores.empty() ? 0 : *std::max_element(st.scores.begin(), st.scores.end());

        st.expv.resize(cur_len, 0);
        if (use_phase_softmax) {
          st.diff_share.resize(cur_len, 0);
          st.exp_share.resize(cur_len, 0);
          for (size_t i = 0; i < cur_len; ++i) {
            int64_t diff = st.max_sc - st.scores[i];
            st.diff_share[i] = (party == 0) ? to_ring(diff) : 0ull;
          }
          pe->add_task(std::make_unique<runtime::CubicPolyTask>(
              nexp_bundle,
              std::span<const uint64_t>(st.diff_share.data(), st.diff_share.size()),
              std::span<uint64_t>(st.exp_share.data(), st.exp_share.size())));
        }
        heads.push_back(std::move(st));
      }
    }

    if (use_phase_softmax && !heads.empty()) {
      runtime::PhaseResources R;
      runtime::ProtoChanFromNet pch(ch);
      R.party = party;
      R.pfss_backend = &ctx->trunc_ctx->backend();
      R.pfss_chan = &pch;
      R.net_chan = &ch;
      R.pfss = &pe->pfss_batch();
      R.opens = &pe->open_collector();
      pe->run(R);
      pe->pfss_batch().clear();
      pe->open_collector().clear();
    }

    // Second wave: sum and reciprocal on shares (scalar) with PFSS/Open batching.
    std::vector<uint64_t> sum_shares;
    std::vector<uint64_t> inv_shares;
    if (use_phase_softmax && !heads.empty()) {
      pe->begin_phase(runtime::PhaseExecutor::Phase::kSoftmax);
      sum_shares.resize(heads.size(), 0);
      inv_shares.resize(heads.size(), 0);
      for (size_t idx = 0; idx < heads.size(); ++idx) {
        uint64_t acc = 0;
        for (auto v : heads[idx].exp_share) acc = proto::add_mod(acc, v);
        sum_shares[idx] = acc;
        pe->add_task(std::make_unique<runtime::RecipTask>(
            recip_bundle,
            std::span<const uint64_t>(&sum_shares[idx], 1),
            std::span<uint64_t>(&inv_shares[idx], 1)));
      }
      runtime::PhaseResources R;
      runtime::ProtoChanFromNet pch(ch);
      R.party = party;
      R.pfss_backend = &ctx->trunc_ctx->backend();
      R.pfss_chan = &pch;
      R.net_chan = &ch;
      R.pfss = &pe->pfss_batch();
      R.opens = &pe->open_collector();
      pe->run(R);
      pe->pfss_batch().clear();
      pe->open_collector().clear();
    }

    for (size_t idx = 0; idx < heads.size(); ++idx) {
      auto& st = heads[idx];
      uint64_t sum_share = 0;
      if (use_phase_softmax) {
        sum_share = sum_shares[idx];
      } else {
        for (size_t s = 0; s < st.cur_len; ++s) {
          int64_t diff = st.max_sc - st.scores[s];
          st.expv[s] = gates::ref_nexp_fixed(nexp_spec, diff);
          sum_share = proto::add_mod(sum_share, to_ring(st.expv[s]));
        }
        if (sum_share == 0) sum_share = 1;
      }

      uint64_t inv_share = 0;
      if (use_phase_softmax) {
        inv_share = inv_shares[idx];
      } else {
        int64_t inv = gates::ref_reciprocal_fixed(recip_spec, to_signed(sum_share), fb, 1);
        inv_share = (party == 0) ? to_ring(inv) : 0ull;
      }

      std::vector<int64_t> prob(st.cur_len, 0);
      if (use_phase_softmax) {
        // secret (exp_share) * public (inv_share) => local mul then faithful trunc.
        std::vector<uint64_t> prod(st.cur_len, 0);
        for (size_t s = 0; s < st.cur_len; ++s) {
          __int128 p = static_cast<__int128>(to_signed(st.exp_share[s])) * static_cast<__int128>(to_signed(inv_share));
          prod[s] = to_ring(p);
        }
        // Truncate Q2f -> Qf via TruncTask (GapARS bundle).
        runtime::PhaseResources R;
        runtime::ProtoChanFromNet pch(ch);
        R.party = party;
        R.pfss_backend = &ctx->trunc_ctx->backend();
        R.pfss_chan = &pch;
        R.net_chan = &ch;
        R.pfss = &pe->pfss_batch();
        R.opens = &pe->open_collector();
        pe->begin_phase(runtime::PhaseExecutor::Phase::kSoftmax);
        pe->add_task(std::make_unique<runtime::TruncTask>(
            recip_bundle.trunc_fb,
            std::span<const uint64_t>(prod.data(), prod.size()),
            std::span<uint64_t>(prod.data(), prod.size())));
        pe->run(R);
        pe->pfss_batch().clear();
        pe->open_collector().clear();
        for (size_t s = 0; s < st.cur_len; ++s) prob[s] = to_signed(prod[s]);
      } else {
        for (size_t s = 0; s < st.cur_len; ++s) {
          __int128 p = static_cast<__int128>(st.expv[s]) * static_cast<__int128>(to_signed(inv_share));
          prob[s] = static_cast<int64_t>(p >> fb);
        }
      }

      for (size_t d = 0; d < Dh; ++d) {
        __int128 acc = 0;
        for (size_t s = 0; s < st.cur_len; ++s) {
          acc += static_cast<__int128>(prob[s]) * static_cast<__int128>(st.v_plain[s * Dh + d]);
        }
        int64_t ctx_val = static_cast<int64_t>(acc >> fb);
        size_t ctx_idx = ((st.b * T + t) * H + st.h) * Dh + d;
        ctx_shares[ctx_idx] = to_ring((party == 0) ? ctx_val : 0);
      }
    }
  }

  std::vector<uint64_t> merged(B * T * D, 0);
  for (size_t b = 0; b < B; ++b) {
    for (size_t t = 0; t < T; ++t) {
      for (size_t h = 0; h < H; ++h) {
        for (size_t d = 0; d < Dh; ++d) {
          size_t dst = (b * T + t) * D + h * Dh + d;
          size_t src = ((b * T + t) * H + h) * Dh + d;
          merged[dst] = ctx_shares[src];
        }
      }
    }
  }

  matmul_publicW(view2(merged.data(), B * T, D),
                 Wout_public,
                 Y_share,
                 mp);
  if (!mp.local_rescale) rescale_view(Y_share, fb);
}

}  // namespace nn
