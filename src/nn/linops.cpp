#include "nn/linops.hpp"

#include <algorithm>
#include <cmath>

namespace nn {

static inline int64_t to_signed(uint64_t v) { return static_cast<int64_t>(v); }
static inline uint64_t to_ring(int64_t v) { return static_cast<uint64_t>(v); }
static inline void rescale_view(const TensorView<uint64_t>& t, int frac_bits) {
  if (frac_bits <= 0) return;
  size_t n = t.numel();
  for (size_t i = 0; i < n; ++i) {
    int64_t s = to_signed(t.data[i]);
    t.data[i] = to_ring(s >> frac_bits);
  }
}

void add(const TensorView<uint64_t>& x,
         const TensorView<uint64_t>& y,
         TensorView<uint64_t> out) {
  size_t n = x.numel();
  for (size_t i = 0; i < n; ++i) out.data[i] = to_ring(to_signed(x.data[i]) + to_signed(y.data[i]));
}

void sub(const TensorView<uint64_t>& x,
         const TensorView<uint64_t>& y,
         TensorView<uint64_t> out) {
  size_t n = x.numel();
  for (size_t i = 0; i < n; ++i) out.data[i] = to_ring(to_signed(x.data[i]) - to_signed(y.data[i]));
}

void mul_const(const TensorView<uint64_t>& x,
               int64_t c,
               int frac_bits,
               TensorView<uint64_t> out,
               bool apply_rescale,
               LayerContext* ctx) {
  bool use_ctx = (ctx != nullptr);
  bool do_shift_inline = apply_rescale && !use_ctx;
  size_t n = x.numel();
  for (size_t i = 0; i < n; ++i) {
    __int128 prod = static_cast<__int128>(to_signed(x.data[i])) * static_cast<__int128>(c);
    int64_t res = static_cast<int64_t>(prod);
    if (do_shift_inline && frac_bits > 0) res >>= frac_bits;
    out.data[i] = to_ring(res);
  }
  if (use_ctx && do_shift_inline) rescale_view(out, frac_bits);
}

void axpy(const TensorView<uint64_t>& x,
          const TensorView<uint64_t>& y,
          int64_t a,
          int frac_bits,
          TensorView<uint64_t> out,
          bool apply_rescale,
          LayerContext* ctx) {
  bool use_ctx = (ctx != nullptr);
  bool do_shift_inline = apply_rescale && !use_ctx;
  size_t n = x.numel();
  for (size_t i = 0; i < n; ++i) {
    __int128 prod = static_cast<__int128>(to_signed(y.data[i])) * static_cast<__int128>(a);
    int64_t res = static_cast<int64_t>(prod);
    if (do_shift_inline && frac_bits > 0) res >>= frac_bits;
    res += to_signed(x.data[i]);
    out.data[i] = to_ring(res);
  }
  if (use_ctx && do_shift_inline) rescale_view(out, frac_bits);
}

void hadamard(const LinOpsContext& ctx,
              const TensorView<uint64_t>& x,
              const TensorView<uint64_t>& y,
              TensorView<uint64_t> out,
              const std::vector<mpc::BeaverTripleA<core::Z2n<64>>>& triples,
              size_t triple_offset,
              int frac_bits,
              bool apply_rescale) {
  size_t n = x.numel();
  for (size_t i = 0; i < n; ++i) {
    auto t = triples[triple_offset + i];
    mpc::AddShare<core::Z2n<64>> xs{core::Z2n<64>(x.data[i])};
    mpc::AddShare<core::Z2n<64>> ys{core::Z2n<64>(y.data[i])};
    auto z = mpc::mul_share(ctx.party, *ctx.ch, xs, ys, t);
    int64_t res = static_cast<int64_t>(z.s.v);
    bool do_shift_inline = apply_rescale && (ctx.graph == nullptr);
    if (do_shift_inline && frac_bits > 0) res >>= frac_bits;
    out.data[i] = to_ring(res);
  }
  if (ctx.graph && do_shift_inline) rescale_view(out, frac_bits);
}

void sum_lastdim(const LinOpsContext& ctx,
                 const TensorView<uint64_t>& x,
                 TensorView<uint64_t> out) {
  (void)ctx;
  size_t outer = 1;
  for (size_t i = 0; i + 1 < x.dims; ++i) outer *= x.shape[i];
  size_t inner = x.shape[x.dims - 1];
  for (size_t o = 0; o < outer; ++o) {
    int64_t acc = 0;
    size_t base = o * inner;
    for (size_t j = 0; j < inner; ++j) acc += to_signed(x.data[base + j]);
    out.data[o] = to_ring(acc);
  }
}

void max_lastdim(const LinOpsContext& ctx,
                 const TensorView<uint64_t>& x,
                 TensorView<uint64_t> out) {
  size_t outer = 1;
  for (size_t i = 0; i + 1 < x.dims; ++i) outer *= x.shape[i];
  size_t inner = x.shape[x.dims - 1];
  for (size_t o = 0; o < outer; ++o) {
    std::vector<uint64_t> other(inner, 0);
    if (ctx.party == 0) {
      for (size_t j = 0; j < inner; ++j) ctx.ch->send_u64(x.data[o * inner + j]);
      for (size_t j = 0; j < inner; ++j) other[j] = ctx.ch->recv_u64();
    } else {
      for (size_t j = 0; j < inner; ++j) other[j] = ctx.ch->recv_u64();
      for (size_t j = 0; j < inner; ++j) ctx.ch->send_u64(x.data[o * inner + j]);
    }
    int64_t best = to_signed(x.data[o * inner]) + static_cast<int64_t>(other[0]);
    for (size_t j = 1; j < inner; ++j) {
      int64_t v = to_signed(x.data[o * inner + j]) + static_cast<int64_t>(other[j]);
      if (v > best) best = v;
    }
    int64_t share = (ctx.party == 0) ? best : 0;
    out.data[o] = to_ring(share);
  }
}

}  // namespace nn
