#include "nn/transformer_layer.hpp"

#include <cassert>
#include <cmath>
#include <vector>
#include "gates/rsqrt_gate.hpp"
#include "gates/tables/rsqrt_piecewise_affine_init.hpp"

namespace nn {

namespace {

inline int64_t to_signed(uint64_t v) { return static_cast<int64_t>(v); }
inline uint64_t to_ring(int64_t v) { return static_cast<uint64_t>(v); }

void open_plain(int party,
                net::Chan& ch,
                const uint64_t* local,
                size_t len,
                std::vector<int64_t>& out) {
  out.resize(len);
  std::vector<uint64_t> other(len, 0);
  if (party == 0) {
    for (size_t i = 0; i < len; ++i) ch.send_u64(local[i]);
    for (size_t i = 0; i < len; ++i) other[i] = ch.recv_u64();
  } else {
    for (size_t i = 0; i < len; ++i) other[i] = ch.recv_u64();
    for (size_t i = 0; i < len; ++i) ch.send_u64(local[i]);
  }
  for (size_t i = 0; i < len; ++i) out[i] = to_signed(local[i]) + to_signed(other[i]);
}

void layernorm_forward(int party,
                       net::Chan& ch,
                       const uint64_t* x_share,
                       size_t D,
                       int frac_bits,
                       const gates::PiecewisePolySpec& rsqrt_spec,
                       int rsqrt_iters,
                       uint64_t* out_share) {
  std::vector<int64_t> x_plain;
  open_plain(party, ch, x_share, D, x_plain);

  int64_t sum = 0;
  for (auto v : x_plain) sum += v;
  int64_t mu = D == 0 ? 0 : (sum / static_cast<int64_t>(D));

  int64_t var_acc = 0;
  for (auto v : x_plain) {
    int64_t d = v - mu;
    __int128 sq = static_cast<__int128>(d) * static_cast<__int128>(d);
    var_acc += static_cast<int64_t>(sq >> frac_bits);
  }
  int64_t var = D == 0 ? 0 : (var_acc / static_cast<int64_t>(D));
  int64_t eps_fixed = static_cast<int64_t>(std::llround((1.0 / 1024.0) * std::ldexp(1.0, frac_bits)));
  int64_t r = gates::ref_rsqrt_fixed(rsqrt_spec, var + eps_fixed, frac_bits, rsqrt_iters);

  for (size_t i = 0; i < D; ++i) {
    int64_t d = x_plain[i] - mu;
    __int128 prod = static_cast<__int128>(d) * static_cast<__int128>(r);
    int64_t y = static_cast<int64_t>(prod >> frac_bits);
    out_share[i] = to_ring((party == 0) ? y : 0);
  }
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
  if (ctx && ctx->pfss_batch == nullptr) {
    ctx->pfss_batch = &pe->pfss_trunc_batch();
  }
  runtime::PhaseExecutor local_pe;
  if (pe == nullptr) pe = &local_pe;
  proto::PfssBackendBatch* backend = (ctx && ctx->trunc_ctx) ? &ctx->trunc_ctx->backend() : nullptr;
  if (ctx) ctx->open_collector = &pe->open_collector();
  runtime::OpenCollector* opens = (pe) ? &pe->open_collector() : nullptr;
  size_t B = X_share.shape[0];
  size_t T = X_share.shape[1];
  size_t D = cfg.attn.D;
  assert(X_share.shape[2] == D);
  assert(W1_public.shape[0] == D && W1_public.shape[1] == cfg.mlp.Hidden);
  assert(W2_public.shape[0] == cfg.mlp.Hidden && W2_public.shape[1] == D);

  gates::RsqrtParams rparams;
  rparams.frac_bits = cfg.frac_bits;
  rparams.eps = 1.0 / 1024.0;
  rparams.vmax = 16.0;
  auto rsqrt_spec = gates::make_rsqrt_affine_init_spec(rparams.frac_bits, rparams.eps, rparams.vmax);
  int rsqrt_iters = 1;

  // LayerNorm 1
  pe->begin_phase(runtime::PhaseExecutor::Phase::kLN1);
  std::vector<uint64_t> ln1(B * T * D, 0);
  for (size_t b = 0; b < B; ++b) {
    for (size_t t = 0; t < T; ++t) {
      const uint64_t* x_ptr = X_share.data + (b * T + t) * D;
      uint64_t* out_ptr = ln1.data() + (b * T + t) * D;
      layernorm_forward(party, ch, x_ptr, D, cfg.frac_bits, rsqrt_spec, rsqrt_iters, out_ptr);
    }
  }
  if (backend) {
    runtime::ProtoChanFromNet pch(ch);
    pe->flush_phase(party, *backend, pch, ch);
  }

  // Attention
  pe->begin_phase(runtime::PhaseExecutor::Phase::kQKV_Score);
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
  if (backend) {
    runtime::ProtoChanFromNet pch(ch);
    pe->flush_phase(party, *backend, pch, ch);
  }

  // Residual add
  for (size_t i = 0; i < attn_out.size(); ++i) {
    attn_out[i] = to_ring(to_signed(attn_out[i]) + to_signed(X_share.data[i]));
  }

  // LayerNorm 2 + MLP
  pe->begin_phase(runtime::PhaseExecutor::Phase::kLN2_MLP);
  std::vector<uint64_t> ln2(attn_out.size(), 0);
  for (size_t b = 0; b < B; ++b) {
    for (size_t t = 0; t < T; ++t) {
      const uint64_t* x_ptr = attn_out.data() + (b * T + t) * D;
      uint64_t* out_ptr = ln2.data() + (b * T + t) * D;
      layernorm_forward(party, ch, x_ptr, D, cfg.frac_bits, rsqrt_spec, rsqrt_iters, out_ptr);
    }
  }

  // MLP
  mlp_forward(cfg.mlp,
              view3(ln2.data(), B, T, D),
              W1_public,
              W2_public,
              Y_share,
              party,
              ch,
              ctx,
              pe);
  if (backend) {
    runtime::ProtoChanFromNet pch(ch);
    pe->flush_phase(party, *backend, pch, ch);
  }

  // Residual add
  for (size_t i = 0; i < Y_share.numel(); ++i) {
    Y_share.data[i] = to_ring(to_signed(Y_share.data[i]) + to_signed(attn_out[i]));
  }

  if (ctx && backend) {
    finalize_layer(*ctx, party, ch, *backend);
  }
}

}  // namespace nn
