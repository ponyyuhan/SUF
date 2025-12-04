#include "nn/attention_block.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>
#include "gates/nexp_gate.hpp"
#include "gates/reciprocal_gate.hpp"

namespace nn {

namespace {

inline int64_t to_signed(uint64_t v) { return static_cast<int64_t>(v); }
inline uint64_t to_ring(int64_t v) { return static_cast<uint64_t>(v); }

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
                       TensorView<uint64_t> Y_share) {
  size_t B = X_share.shape[0];
  size_t T = X_share.shape[1];
  size_t D = cfg.D;
  size_t H = cfg.H;
  size_t Dh = cfg.Dh;
  int fb = cfg.frac_bits;

  assert(D == H * Dh);
  assert(Wqkv_public.shape[0] == D && Wqkv_public.shape[1] == 3 * D);
  assert(cache.B == B && cache.H == H && cache.Dh == Dh);

  MatmulParams mp;
  mp.frac_bits = fb;
  mp.w_transposed = false;

  // [B*T, 3D]
  std::vector<uint64_t> qkv(B * T * 3 * D, 0);
  matmul_publicW(view2(const_cast<uint64_t*>(X_share.data), B * T, D),
                 Wqkv_public,
                 view2(qkv.data(), B * T, 3 * D),
                 mp);

  std::vector<uint64_t> ctx_shares(B * T * H * Dh, 0);
  size_t init_len = cache.cur_len;

  gates::NExpGateParams nexp_params;
  nexp_params.frac_bits = fb;
  nexp_params.segments = 16;
  auto nexp_spec = gates::make_nexp_spec(nexp_params);
  auto recip_spec =
      gates::make_recip_affine_init_spec(fb, static_cast<double>(std::max(cache.S_max, T + init_len)));

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

    for (size_t b = 0; b < B; ++b) {
      size_t q_base = (b * T + t) * 3 * D;
      const uint64_t* q_ptr = qkv.data() + q_base;
      for (size_t h = 0; h < H; ++h) {
        const uint64_t* k_head = kv_head_ptr(cache, b, h);
        const uint64_t* v_head = kv_head_ptr_v(cache, b, h);

        std::vector<int64_t> q_plain, k_plain, v_plain;
        open_to_plain(party, ch, q_ptr + h * Dh, Dh, q_plain);
        open_to_plain(party, ch, k_head, cur_len * Dh, k_plain);
        open_to_plain(party, ch, v_head, cur_len * Dh, v_plain);

        std::vector<int64_t> scores(cur_len, 0);
        for (size_t s = 0; s < cur_len; ++s) {
          __int128 acc = 0;
          for (size_t d = 0; d < Dh; ++d) {
            acc += static_cast<__int128>(q_plain[d]) * static_cast<__int128>(k_plain[s * Dh + d]);
          }
          acc >>= fb;
          acc = (acc * static_cast<__int128>(inv_sqrt)) >> fb;
          scores[s] = static_cast<int64_t>(acc);
        }

        int64_t max_sc = scores.empty() ? 0 : *std::max_element(scores.begin(), scores.end());
        std::vector<int64_t> expv(cur_len, 0);
        int64_t sum = 0;
        for (size_t s = 0; s < cur_len; ++s) {
          int64_t diff = max_sc - scores[s];
          expv[s] = gates::ref_nexp_fixed(nexp_spec, diff);
          sum += expv[s];
        }
        if (sum == 0) sum = 1;
        int64_t inv = gates::ref_reciprocal_fixed(recip_spec, sum, fb, 1);

        std::vector<int64_t> prob(cur_len, 0);
        for (size_t s = 0; s < cur_len; ++s) {
          __int128 p = static_cast<__int128>(expv[s]) * static_cast<__int128>(inv);
          prob[s] = static_cast<int64_t>(p >> fb);
        }

        for (size_t d = 0; d < Dh; ++d) {
          __int128 acc = 0;
          for (size_t s = 0; s < cur_len; ++s) {
            acc += static_cast<__int128>(prob[s]) * static_cast<__int128>(v_plain[s * Dh + d]);
          }
          int64_t ctx_val = static_cast<int64_t>(acc >> fb);
          size_t ctx_idx = ((b * T + t) * H + h) * Dh + d;
          ctx_shares[ctx_idx] = to_ring((party == 0) ? ctx_val : 0);
        }
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
}

}  // namespace nn
