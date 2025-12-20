#include "runtime/cuda_primitives.hpp"
#include <cuda_runtime.h>

// Provided by cuda/pfss_kernels.cu (compiled into cuda_pfss library).
extern "C" __global__ void unpack_eff_bits_kernel(const uint64_t* packed,
                                                  int eff_bits,
                                                  uint64_t* out,
                                                  size_t N);
extern "C" __global__ void pack_eff_bits_kernel(const uint64_t* in,
                                                int eff_bits,
                                                uint64_t* packed,
                                                size_t N);
extern "C" __global__ void pack_eff_bits_wordwise_kernel(const uint64_t* in,
                                                         int eff_bits,
                                                         uint64_t* packed,
                                                         size_t N,
                                                         size_t packed_words);

namespace {

__device__ inline uint64_t add_mod(uint64_t a, uint64_t b) { return a + b; }
__device__ inline uint64_t mul_mod(uint64_t a, uint64_t b) {
  unsigned __int128 p = static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b);
  return static_cast<uint64_t>(p);
}

__global__ void open_add_to_signed_kernel(const uint64_t* __restrict__ local_share,
                                          const uint64_t* __restrict__ remote_share,
                                          uint64_t* __restrict__ out_signed,
                                          size_t n,
                                          int ring_bits,
                                          uint64_t ring_mask) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  uint64_t v = local_share[idx] + remote_share[idx];
  if (ring_bits >= 64) {
    out_signed[idx] = v;
    return;
  }
  v &= ring_mask;
  const uint64_t sign_bit = uint64_t(1) << (ring_bits - 1);
  if (v & sign_bit) v |= ~ring_mask;
  out_signed[idx] = v;
}

__device__ __forceinline__ uint64_t mask_low_bits(int bits) {
  if (bits <= 0) return 0ull;
  if (bits >= 64) return ~uint64_t(0);
  return (uint64_t(1) << bits) - 1;
}

__global__ void unpack_add_to_signed_kernel(const uint64_t* __restrict__ local_share,
                                            const uint64_t* __restrict__ packed_remote,
                                            int eff_bits,
                                            size_t packed_words,
                                            uint64_t* __restrict__ out_signed,
                                            size_t n,
                                            int ring_bits,
                                            uint64_t ring_mask) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  if (eff_bits >= 64) {
    uint64_t v = local_share[idx] + packed_remote[idx];
    if (ring_bits < 64) {
      v &= ring_mask;
      const uint64_t sign_bit = uint64_t(1) << (ring_bits - 1);
      if (v & sign_bit) v |= ~ring_mask;
    }
    out_signed[idx] = v;
    return;
  }
  const size_t bit_off = idx * static_cast<size_t>(eff_bits);
  const size_t word = bit_off >> 6;
  const int bit = static_cast<int>(bit_off & 63ull);
  uint64_t lo = (word < packed_words) ? packed_remote[word] : 0ull;
  uint64_t hi = (word + 1 < packed_words) ? packed_remote[word + 1] : 0ull;
  uint64_t v = (bit == 0) ? lo : ((lo >> bit) | (hi << (64 - bit)));
  v &= mask_low_bits(eff_bits);
  v = local_share[idx] + v;
  if (ring_bits < 64) {
    v &= ring_mask;
    const uint64_t sign_bit = uint64_t(1) << (ring_bits - 1);
    if (v & sign_bit) v |= ~ring_mask;
  }
  out_signed[idx] = v;
}

__global__ void unpack_add_mod_kernel(const uint64_t* __restrict__ local_share,
                                      const uint64_t* __restrict__ packed_remote,
                                      int eff_bits,
                                      size_t packed_words,
                                      uint64_t* __restrict__ out_mod,
                                      size_t n,
                                      int ring_bits,
                                      uint64_t ring_mask) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  uint64_t v = 0;
  if (eff_bits >= 64) {
    v = local_share[idx] + packed_remote[idx];
  } else {
    const size_t bit_off = idx * static_cast<size_t>(eff_bits);
    const size_t word = bit_off >> 6;
    const int bit = static_cast<int>(bit_off & 63ull);
    uint64_t lo = (word < packed_words) ? packed_remote[word] : 0ull;
    uint64_t hi = (word + 1 < packed_words) ? packed_remote[word + 1] : 0ull;
    uint64_t r = (bit == 0) ? lo : ((lo >> bit) | (hi << (64 - bit)));
    r &= mask_low_bits(eff_bits);
    v = local_share[idx] + r;
  }
  if (ring_bits < 64) v &= ring_mask;
  out_mod[idx] = v;
}

__global__ void beaver_mul_kernel(int party,
                                  const uint64_t* __restrict__ x,
                                  const uint64_t* __restrict__ y,
                                  const uint64_t* __restrict__ a,
                                  const uint64_t* __restrict__ b,
                                  const uint64_t* __restrict__ c,
                                  const uint64_t* __restrict__ d_open,
                                  const uint64_t* __restrict__ e_open,
                                  uint64_t* __restrict__ out,
                                  size_t n) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  uint64_t d = d_open[idx];
  uint64_t e = e_open[idx];
  uint64_t z = c[idx];
  z = add_mod(z, mul_mod(d, b[idx]));
  z = add_mod(z, mul_mod(e, a[idx]));
  if (party == 0) {
    z = add_mod(z, mul_mod(d, e));
  }
  out[idx] = z;
}

struct BeaverTripleAos {
  uint64_t a;
  uint64_t b;
  uint64_t c;
};

__global__ void beaver_mul_aos_kernel(int party,
                                      const BeaverTripleAos* __restrict__ triples,
                                      const uint64_t* __restrict__ d_open,
                                      const uint64_t* __restrict__ e_open,
                                      uint64_t* __restrict__ out,
                                      size_t n) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  uint64_t d = d_open[idx];
  uint64_t e = e_open[idx];
  BeaverTripleAos t = triples[idx];
  uint64_t z = t.c;
  z = add_mod(z, mul_mod(d, t.b));
  z = add_mod(z, mul_mod(e, t.a));
  if (party == 0) {
    z = add_mod(z, mul_mod(d, e));
  }
  out[idx] = z;
}

__global__ void trunc_shift_kernel(const uint64_t* __restrict__ in,
                                   uint64_t* __restrict__ out,
                                   int frac_bits,
                                   size_t n) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  uint64_t x = in[idx];
  if (frac_bits <= 0) {
    out[idx] = x;
  } else if (frac_bits >= 64) {
    out[idx] = 0;
  } else {
    out[idx] = x >> frac_bits;
  }
}

__global__ void trunc_postproc_kernel(int party,
                                      int kind_gapars,
                                      int frac_bits,
                                      uint64_t r_hi_share,
                                      uint64_t m_share,
                                      const uint64_t* __restrict__ hatx_public,
                                      const uint64_t* __restrict__ arith,
                                      size_t arith_stride,
                                      int arith_idx,
                                      const uint64_t* __restrict__ bools,
                                      size_t bool_stride,
                                      int carry_idx,
                                      int sign_idx,
                                      int wrap_idx,
                                      uint64_t* __restrict__ out,
                                      size_t n) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  uint64_t base = arith[arith_idx + idx * arith_stride];
  bool have_carry = bools && carry_idx >= 0 && carry_idx < static_cast<int>(bool_stride);
  bool have_sign = bools && sign_idx >= 0 && sign_idx < static_cast<int>(bool_stride);
  bool have_wrap = bools && wrap_idx >= 0 && wrap_idx < static_cast<int>(bool_stride);
  uint64_t carry = have_carry ? bools[carry_idx + idx * bool_stride] : 0ull;
  uint64_t sign = have_sign ? bools[sign_idx + idx * bool_stride] : 0ull;
  uint64_t wrap = have_wrap ? bools[wrap_idx + idx * bool_stride] : 0ull;
  uint64_t y = base;
  if (kind_gapars) {
    const uint64_t bias_in = uint64_t(1) << 62;  // 2^(64-2)
    uint64_t hatx = hatx_public ? hatx_public[idx] : 0ull;
    uint64_t hatx_biased = add_mod(hatx, bias_in);
    uint64_t top = 0ull;
    if (party == 0) {
      if (frac_bits <= 0) {
        top = hatx_biased;
      } else if (frac_bits >= 64) {
        top = 0ull;
      } else {
        top = hatx_biased >> frac_bits;
      }
    }
    y = add_mod(y, top);
    y = add_mod(y, (~r_hi_share) + 1);  // subtract r_hi_share
    y = add_mod(y, (~carry) + 1);       // subtract carry
    if (frac_bits > 0 && frac_bits < 64) {
      uint64_t msb_hatx = (hatx_biased >> 63) & 1ull;
      if (msb_hatx == 0) {
        y = add_mod(y, m_share);
      }
    }
    if (party == 0 && frac_bits >= 0 && frac_bits <= 62) {
      uint64_t bias_out = uint64_t(1) << (62 - frac_bits);
      y = add_mod(y, (~bias_out) + 1);
    }
    out[idx] = y;
    return;
  }
  if (party == 0 && frac_bits > 0 && frac_bits < 64) {
    uint64_t top = hatx_public ? (hatx_public[idx] >> frac_bits) : 0ull;
    y = add_mod(y, top);
  }
  y = add_mod(y, (~r_hi_share) + 1); // subtract r_hi_share
  // Carry is always subtracted (faithful and GapARS share the same postproc).
  y = add_mod(y, (~carry) + 1);      // subtract carry
  if (frac_bits > 0 && frac_bits < 64) {
    uint64_t modulus = (uint64_t(1) << (64 - frac_bits));
    y = add_mod(y, mul_mod(wrap, modulus));
  }
  if (frac_bits > 0 && frac_bits < 64) {
    uint64_t sign_mask = ~uint64_t(0) << (64 - frac_bits);
    uint64_t sign_term = mul_mod(sign, sign_mask);
    y = add_mod(y, sign_term);
  }
  out[idx] = y;
}

__global__ void horner_cubic_kernel(const uint64_t* __restrict__ x,
                                    const uint64_t* __restrict__ c0,
                                    const uint64_t* __restrict__ c1,
                                    const uint64_t* __restrict__ c2,
                                    const uint64_t* __restrict__ c3,
                                    uint64_t* __restrict__ out,
                                    size_t n) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  uint64_t xv = x[idx];
  uint64_t y = c3[idx];
  y = add_mod(mul_mod(y, xv), c2[idx]);
  y = add_mod(mul_mod(y, xv), c1[idx]);
  y = add_mod(mul_mod(y, xv), c0[idx]);
  out[idx] = y;
}

__global__ void row_broadcast_mul_kernel(int party,
                                         const uint64_t* __restrict__ A,
                                         const uint64_t* __restrict__ B_rows,
                                         const uint64_t* __restrict__ C,
                                         const uint64_t* __restrict__ d_open,
                                         const uint64_t* __restrict__ e_open_rows,
                                         int rows,
                                         int cols,
                                         const int* __restrict__ valid_lens,
                                         uint64_t* __restrict__ out,
                                         size_t n) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  if (cols <= 0) return;
  int r = static_cast<int>(idx / static_cast<size_t>(cols));
  int c = static_cast<int>(idx - static_cast<size_t>(r) * static_cast<size_t>(cols));
  if (r < 0 || r >= rows) return;
  if (valid_lens) {
    int L = valid_lens[r];
    if (c >= L) {
      out[idx] = 0;
      return;
    }
  }
  uint64_t d = d_open[idx];
  uint64_t e = e_open_rows[r];
  uint64_t z = C[idx];
  z = add_mod(z, mul_mod(d, B_rows[r]));
  z = add_mod(z, mul_mod(e, A[idx]));
  if (party == 0) {
    z = add_mod(z, mul_mod(d, e));
  }
  out[idx] = z;
}

// Simple row-sum reduction: one block per row. Assumes cols <= 1024 for now.
__global__ void row_sum_kernel(const uint64_t* __restrict__ mat,
                               int rows,
                               int cols,
                               const int* __restrict__ valid_lens,
                               uint64_t* __restrict__ out_rows) {
  int r = blockIdx.x;
  if (r >= rows) return;
  int L = valid_lens ? valid_lens[r] : cols;
  uint64_t acc = 0;
  for (int c = threadIdx.x; c < L; c += blockDim.x) {
    size_t idx = static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c);
    acc = add_mod(acc, mat[idx]);
  }
  // Reduce within block (naive): shared memory reduction.
  __shared__ uint64_t buf[256];
  int tid = threadIdx.x;
  buf[tid] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) buf[tid] = add_mod(buf[tid], buf[tid + stride]);
    __syncthreads();
  }
  if (tid == 0) out_rows[r] = buf[0];
}

// Row-sum reduction for ragged packed rows: one block per row, row data is
// stored in a single packed array with row_offsets prefix sums.
__global__ void row_sum_ragged_kernel(const uint64_t* __restrict__ vals,
                                      const int* __restrict__ row_offsets,
                                      int rows,
                                      uint64_t* __restrict__ out_rows) {
  int r = blockIdx.x;
  if (r >= rows) return;
  int start = row_offsets ? row_offsets[r] : 0;
  int end = row_offsets ? row_offsets[r + 1] : 0;
  if (start < 0) start = 0;
  if (end < start) end = start;
  uint64_t acc = 0;
  for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
    acc = add_mod(acc, vals[static_cast<size_t>(i)]);
  }
  __shared__ uint64_t buf[256];
  int tid = threadIdx.x;
  buf[tid] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) buf[tid] = add_mod(buf[tid], buf[tid + stride]);
    __syncthreads();
  }
  if (tid == 0) out_rows[r] = buf[0];
}

__device__ __forceinline__ uint64_t beaver_scaled_acc(uint64_t acc_mod,
                                                      int has_scale,
                                                      int64_t mul_const,
                                                      int mul_shift) {
  if (!has_scale) return acc_mod;
  if (mul_shift <= 0) {
    // Low 64 bits of signed product (two's complement); matches host cast behavior.
    int64_t a = static_cast<int64_t>(acc_mod);
    uint64_t lo = static_cast<uint64_t>(a) * static_cast<uint64_t>(mul_const);
    return lo;
  }
  if (mul_shift >= 64) {
    int64_t a = static_cast<int64_t>(acc_mod);
    int64_t hi = __mul64hi(a, mul_const);
    // Arithmetic shift of 128 by >=64: low word comes from hi.
    int s = mul_shift - 64;
    uint64_t out_lo = static_cast<uint64_t>(hi >> s);
    return out_lo;
  }
  int64_t a = static_cast<int64_t>(acc_mod);
  uint64_t lo = static_cast<uint64_t>(a) * static_cast<uint64_t>(mul_const);
  int64_t hi = __mul64hi(a, mul_const);
  const int s = mul_shift;
  // Low word of arithmetic-right-shifted 128-bit product.
  uint64_t out_lo = (lo >> s) | (static_cast<uint64_t>(hi) << (64 - s));
  return out_lo;
}

__global__ void beaver_matmul2d_kernel(int party,
                                       const uint64_t* __restrict__ D_open,
                                       const uint64_t* __restrict__ E_open,
                                       const uint64_t* __restrict__ A_tri,
                                       const uint64_t* __restrict__ B_tri,
                                       const uint64_t* __restrict__ C_tri,
                                       int M,
                                       int K,
                                       int N,
                                       int w_transposed,
                                       int has_scale,
                                       int64_t mul_const,
                                       int mul_shift,
                                       uint64_t* __restrict__ out) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) +
                     static_cast<size_t>(threadIdx.x);
  const size_t total = static_cast<size_t>(M) * static_cast<size_t>(N);
  if (idx >= total) return;
  const int m = static_cast<int>(idx / static_cast<size_t>(N));
  const int n = static_cast<int>(idx - static_cast<size_t>(m) * static_cast<size_t>(N));
  uint64_t acc = C_tri[idx];
  const int party0 = (party == 0);
  for (int k = 0; k < K; ++k) {
    const size_t aidx = static_cast<size_t>(m) * static_cast<size_t>(K) + static_cast<size_t>(k);
    const size_t bidx = w_transposed
                            ? (static_cast<size_t>(n) * static_cast<size_t>(K) + static_cast<size_t>(k))
                            : (static_cast<size_t>(k) * static_cast<size_t>(N) + static_cast<size_t>(n));
    const uint64_t d = D_open[aidx];
    const uint64_t e = E_open[bidx];
    acc = add_mod(acc, mul_mod(d, B_tri[bidx]));
    acc = add_mod(acc, mul_mod(A_tri[aidx], e));
    if (party0) {
      acc = add_mod(acc, mul_mod(d, e));
    }
  }
  acc = beaver_scaled_acc(acc, has_scale, mul_const, mul_shift);
  out[idx] = acc;
}

__global__ void beaver_matmul2d_batched_kernel(int party,
                                               const uint64_t* __restrict__ D_open,
                                               const uint64_t* __restrict__ E_open,
                                               const uint64_t* __restrict__ A_tri,
                                               const uint64_t* __restrict__ B_tri,
                                               const uint64_t* __restrict__ C_tri,
                                               int batches,
                                               int M,
                                               int K,
                                               int N,
                                               int w_transposed,
                                               int has_scale,
                                               int64_t mul_const,
                                               int mul_shift,
                                               uint64_t* __restrict__ out) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) +
                     static_cast<size_t>(threadIdx.x);
  const size_t mn = static_cast<size_t>(M) * static_cast<size_t>(N);
  const size_t total = static_cast<size_t>(batches) * mn;
  if (idx >= total) return;
  const int b = static_cast<int>(idx / mn);
  const size_t off_mn = static_cast<size_t>(b) * mn;
  const size_t idx_mn = idx - off_mn;
  const int m = static_cast<int>(idx_mn / static_cast<size_t>(N));
  const int n = static_cast<int>(idx_mn - static_cast<size_t>(m) * static_cast<size_t>(N));

  const size_t mk = static_cast<size_t>(M) * static_cast<size_t>(K);
  const size_t kn = static_cast<size_t>(K) * static_cast<size_t>(N);
  const size_t off_mk = static_cast<size_t>(b) * mk;
  const size_t off_kn = static_cast<size_t>(b) * kn;

  uint64_t acc = C_tri[idx_mn];
  const int party0 = (party == 0);
  for (int k = 0; k < K; ++k) {
    const size_t aidx = static_cast<size_t>(m) * static_cast<size_t>(K) + static_cast<size_t>(k);
    const size_t bidx = w_transposed
                            ? (static_cast<size_t>(n) * static_cast<size_t>(K) + static_cast<size_t>(k))
                            : (static_cast<size_t>(k) * static_cast<size_t>(N) + static_cast<size_t>(n));
    const uint64_t d = D_open[off_mk + aidx];
    const uint64_t e = E_open[off_kn + bidx];
    acc = add_mod(acc, mul_mod(d, B_tri[bidx]));
    acc = add_mod(acc, mul_mod(A_tri[aidx], e));
    if (party0) {
      acc = add_mod(acc, mul_mod(d, e));
    }
  }
  acc = beaver_scaled_acc(acc, has_scale, mul_const, mul_shift);
  out[off_mn + idx_mn] = acc;
}

__global__ void row_mean_kernel(const uint64_t* __restrict__ mat,
                                int rows,
                                int cols,
                                const int* __restrict__ valid_lens,
                                uint64_t* __restrict__ out_rows) {
  int r = blockIdx.x;
  if (r >= rows) return;
  int L = valid_lens ? valid_lens[r] : cols;
  if (L <= 0) {
    if (threadIdx.x == 0) out_rows[r] = 0;
    return;
  }
  uint64_t acc = 0;
  for (int c = threadIdx.x; c < L; c += blockDim.x) {
    size_t idx = static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c);
    acc = add_mod(acc, mat[idx]);
  }
  __shared__ uint64_t buf[256];
  int tid = threadIdx.x;
  buf[tid] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) buf[tid] = add_mod(buf[tid], buf[tid + stride]);
    __syncthreads();
  }
  if (tid == 0) {
    // Divide by L mod 2^64: use multiplicative inverse modulo 2^64 when L odd; fallback to shift if power of two.
    uint64_t sum = buf[0];
    uint64_t mean = 0;
    if ((L & (L - 1)) == 0) {
      int shift = 0;
      while ((1 << shift) != L && shift < 63) ++shift;
      mean = (shift >= 64) ? 0ull : (sum >> shift);
    } else {
      // Approximate inverse using 128-bit division (host would be better; here we keep it simple).
      unsigned __int128 num = static_cast<unsigned __int128>(sum);
      mean = static_cast<uint64_t>(num / static_cast<unsigned>(L));
    }
    out_rows[r] = mean;
  }
}

__global__ void row_variance_kernel(const uint64_t* __restrict__ mat,
                                    const uint64_t* __restrict__ mean,
                                    int rows,
                                    int cols,
                                    const int* __restrict__ valid_lens,
                                    uint64_t* __restrict__ out_rows) {
  int r = blockIdx.x;
  if (r >= rows) return;
  int L = valid_lens ? valid_lens[r] : cols;
  uint64_t mu = mean[r];
  uint64_t acc = 0;
  for (int c = threadIdx.x; c < L; c += blockDim.x) {
    size_t idx = static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c);
    uint64_t x = mat[idx];
    int64_t diff = static_cast<int64_t>(x) - static_cast<int64_t>(mu);
    uint64_t diff_u = static_cast<uint64_t>(diff);
    acc = add_mod(acc, mul_mod(diff_u, diff_u));
  }
  __shared__ uint64_t buf[256];
  int tid = threadIdx.x;
  buf[tid] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) buf[tid] = add_mod(buf[tid], buf[tid + stride]);
    __syncthreads();
  }
  if (tid == 0) {
    if (L <= 0) {
      out_rows[r] = 0;
      return;
    }
    // Return the row sum of squared diffs; caller applies any scaling (e.g., inv_len)
    // using ring multiplication to preserve mod 2^64 semantics.
    out_rows[r] = buf[0];
  }
}

}  // namespace

extern "C" void launch_beaver_mul_kernel(int party,
                                         const uint64_t* d_x,
                                         const uint64_t* d_y,
                                         const uint64_t* d_a,
                                         const uint64_t* d_b,
                                         const uint64_t* d_c,
                                         const uint64_t* d_d_open,
                                         const uint64_t* d_e_open,
                                         uint64_t* d_out,
                                         size_t n,
                                         void* stream) {
#ifndef SUF_HAVE_CUDA
  (void)party; (void)d_x; (void)d_y; (void)d_a; (void)d_b; (void)d_c; (void)d_d_open; (void)d_e_open; (void)d_out; (void)n; (void)stream;
#else
  if (n == 0) return;
  constexpr int kBlock = 256;
  int grid = static_cast<int>((n + kBlock - 1) / kBlock);
  beaver_mul_kernel<<<grid, kBlock, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      party, d_x, d_y, d_a, d_b, d_c, d_d_open, d_e_open, d_out, n);
#endif
}

extern "C" void launch_beaver_mul_aos_kernel(int party,
                                             const void* d_triples,
                                             const uint64_t* d_d_open,
                                             const uint64_t* d_e_open,
                                             uint64_t* d_out,
                                             size_t n,
                                             void* stream) {
#ifndef SUF_HAVE_CUDA
  (void)party; (void)d_triples; (void)d_d_open; (void)d_e_open; (void)d_out; (void)n; (void)stream;
#else
  if (n == 0) return;
  constexpr int kBlock = 256;
  int grid = static_cast<int>((n + kBlock - 1) / kBlock);
  beaver_mul_aos_kernel<<<grid, kBlock, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      party,
      reinterpret_cast<const BeaverTripleAos*>(d_triples),
      d_d_open,
      d_e_open,
      d_out,
      n);
#endif
}

extern "C" void launch_trunc_shift_kernel(const uint64_t* d_in,
                                          uint64_t* d_out,
                                          int frac_bits,
                                          size_t n,
                                          void* stream) {
#ifndef SUF_HAVE_CUDA
  (void)d_in; (void)d_out; (void)frac_bits; (void)n; (void)stream;
#else
  if (n == 0) return;
  constexpr int kBlock = 256;
  int grid = static_cast<int>((n + kBlock - 1) / kBlock);
  trunc_shift_kernel<<<grid, kBlock, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      d_in, d_out, frac_bits, n);
#endif
}

extern "C" void launch_trunc_postproc_kernel(int party,
                                             int kind_gapars,
                                             int frac_bits,
                                             uint64_t r_hi_share,
                                             uint64_t m_share,
                                             const uint64_t* d_hatx_public,
                                             const uint64_t* d_arith,
                                             size_t arith_stride,
                                             int arith_idx,
                                             const uint64_t* d_bools,
                                             size_t bool_stride,
                                             int carry_idx,
                                             int sign_idx,
                                             int wrap_idx,
                                             uint64_t* d_out,
                                             size_t n,
                                             void* stream) {
#ifndef SUF_HAVE_CUDA
  (void)party; (void)kind_gapars; (void)frac_bits; (void)r_hi_share; (void)m_share;
  (void)d_hatx_public; (void)d_arith; (void)arith_stride; (void)arith_idx;
  (void)d_bools; (void)bool_stride; (void)carry_idx; (void)sign_idx; (void)wrap_idx; (void)d_out; (void)n; (void)stream;
#else
  if (n == 0) return;
  constexpr int kBlock = 256;
  int grid = static_cast<int>((n + kBlock - 1) / kBlock);
  trunc_postproc_kernel<<<grid, kBlock, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      party, kind_gapars, frac_bits, r_hi_share, m_share,
      d_hatx_public, d_arith, arith_stride, arith_idx,
      d_bools, bool_stride, carry_idx, sign_idx, wrap_idx,
      d_out, n);
#endif
}

extern "C" void launch_horner_cubic_kernel(const uint64_t* d_x,
                                           const uint64_t* d_c0,
                                           const uint64_t* d_c1,
                                           const uint64_t* d_c2,
                                           const uint64_t* d_c3,
                                           uint64_t* d_out,
                                           size_t n,
                                           void* stream) {
#ifndef SUF_HAVE_CUDA
  (void)d_x; (void)d_c0; (void)d_c1; (void)d_c2; (void)d_c3; (void)d_out; (void)n; (void)stream;
#else
  if (n == 0) return;
  constexpr int kBlock = 256;
  int grid = static_cast<int>((n + kBlock - 1) / kBlock);
  horner_cubic_kernel<<<grid, kBlock, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      d_x, d_c0, d_c1, d_c2, d_c3, d_out, n);
#endif
}

extern "C" void launch_row_broadcast_mul_kernel(int party,
                                                const uint64_t* d_A,
                                                const uint64_t* d_B_rows,
                                                const uint64_t* d_C,
                                                const uint64_t* d_d_open,
                                                const uint64_t* d_e_open_rows,
                                                int rows,
                                                int cols,
                                                const int* d_valid_lens,
                                                uint64_t* d_out,
                                                size_t n,
                                                void* stream) {
#ifndef SUF_HAVE_CUDA
  (void)party; (void)d_A; (void)d_B_rows; (void)d_C;
  (void)d_d_open; (void)d_e_open_rows; (void)rows; (void)cols; (void)d_valid_lens;
  (void)d_out; (void)n; (void)stream;
#else
  if (n == 0) return;
  constexpr int kBlock = 256;
  int grid = static_cast<int>((n + kBlock - 1) / kBlock);
  row_broadcast_mul_kernel<<<grid, kBlock, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      party, d_A, d_B_rows, d_C, d_d_open, d_e_open_rows,
      rows, cols, d_valid_lens, d_out, n);
#endif
}

extern "C" void launch_row_sum_kernel(const uint64_t* d_mat,
                                       int rows,
                                       int cols,
                                       const int* d_valid_lens,
                                       uint64_t* d_out_rows,
                                       void* stream) {
#ifndef SUF_HAVE_CUDA
  (void)d_mat; (void)rows; (void)cols; (void)d_valid_lens; (void)d_out_rows; (void)stream;
#else
  if (rows <= 0 || cols <= 0) return;
  dim3 grid(rows);
  dim3 block(256);
  row_sum_kernel<<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(d_mat, rows, cols, d_valid_lens, d_out_rows);
#endif
}

extern "C" void launch_row_sum_ragged_kernel(const uint64_t* d_vals,
                                              const int* d_row_offsets,
                                              int rows,
                                              uint64_t* d_out_rows,
                                              void* stream) {
#ifndef SUF_HAVE_CUDA
  (void)d_vals; (void)d_row_offsets; (void)rows; (void)d_out_rows; (void)stream;
#else
  if (rows <= 0) return;
  dim3 grid(rows);
  dim3 block(256);
  row_sum_ragged_kernel<<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      d_vals, d_row_offsets, rows, d_out_rows);
#endif
}

extern "C" void launch_beaver_matmul2d_kernel(int party,
                                              const uint64_t* d_D_open,
                                              const uint64_t* d_E_open,
                                              const uint64_t* d_A_tri_share,
                                              const uint64_t* d_B_tri_share,
                                              const uint64_t* d_C_tri_share,
                                              int M,
                                              int K,
                                              int N,
                                              int w_transposed,
                                              int has_scale,
                                              int64_t mul_const,
                                              int mul_shift,
                                              uint64_t* d_out,
                                              void* stream) {
#ifndef SUF_HAVE_CUDA
  (void)party;
  (void)d_D_open;
  (void)d_E_open;
  (void)d_A_tri_share;
  (void)d_B_tri_share;
  (void)d_C_tri_share;
  (void)M;
  (void)K;
  (void)N;
  (void)w_transposed;
  (void)has_scale;
  (void)mul_const;
  (void)mul_shift;
  (void)d_out;
  (void)stream;
#else
  if (M <= 0 || K <= 0 || N <= 0) return;
  const size_t total = static_cast<size_t>(M) * static_cast<size_t>(N);
  constexpr int kBlock = 256;
  int grid = static_cast<int>((total + static_cast<size_t>(kBlock) - 1) / static_cast<size_t>(kBlock));
	  beaver_matmul2d_kernel<<<grid, kBlock, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
	      party,
	      d_D_open,
	      d_E_open,
	      d_A_tri_share,
	      d_B_tri_share,
	      d_C_tri_share,
	      M,
	      K,
	      N,
	      w_transposed,
	      has_scale,
	      mul_const,
	      mul_shift,
	      d_out);
#endif
}

extern "C" void launch_beaver_matmul2d_batched_kernel(int party,
                                                      const uint64_t* d_D_open,
                                                      const uint64_t* d_E_open,
                                                      const uint64_t* d_A_tri_share,
                                                      const uint64_t* d_B_tri_share,
                                                      const uint64_t* d_C_tri_share,
                                                      int batches,
                                                      int M,
                                                      int K,
                                                      int N,
                                                      int w_transposed,
                                                      int has_scale,
                                                      int64_t mul_const,
                                                      int mul_shift,
                                                      uint64_t* d_out,
                                                      void* stream) {
#ifndef SUF_HAVE_CUDA
  (void)party;
  (void)d_D_open;
  (void)d_E_open;
  (void)d_A_tri_share;
  (void)d_B_tri_share;
  (void)d_C_tri_share;
  (void)batches;
  (void)M;
  (void)K;
  (void)N;
  (void)w_transposed;
  (void)has_scale;
  (void)mul_const;
  (void)mul_shift;
  (void)d_out;
  (void)stream;
#else
  if (batches <= 0 || M <= 0 || K <= 0 || N <= 0) return;
  const size_t total = static_cast<size_t>(batches) * static_cast<size_t>(M) * static_cast<size_t>(N);
  constexpr int kBlock = 256;
  int grid = static_cast<int>((total + static_cast<size_t>(kBlock) - 1) / static_cast<size_t>(kBlock));
  beaver_matmul2d_batched_kernel<<<grid, kBlock, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      party,
      d_D_open,
      d_E_open,
      d_A_tri_share,
      d_B_tri_share,
      d_C_tri_share,
      batches,
      M,
      K,
      N,
      w_transposed,
      has_scale,
      mul_const,
      mul_shift,
      d_out);
#endif
}

extern "C" void launch_row_mean_kernel(const uint64_t* d_mat,
                                        int rows,
                                        int cols,
                                        const int* d_valid_lens,
                                        uint64_t* d_out_rows,
                                        void* stream) {
#ifndef SUF_HAVE_CUDA
  (void)d_mat; (void)rows; (void)cols; (void)d_valid_lens; (void)d_out_rows; (void)stream;
#else
  if (rows <= 0 || cols <= 0) return;
  dim3 grid(rows);
  dim3 block(256);
  row_mean_kernel<<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(d_mat, rows, cols, d_valid_lens, d_out_rows);
#endif
}

extern "C" void launch_row_variance_kernel(const uint64_t* d_mat,
                                            const uint64_t* d_mean,
                                            int rows,
                                            int cols,
                                            const int* d_valid_lens,
                                            uint64_t* d_out_rows,
                                            void* stream) {
#ifndef SUF_HAVE_CUDA
  (void)d_mat; (void)d_mean; (void)rows; (void)cols; (void)d_valid_lens; (void)d_out_rows; (void)stream;
#else
  if (rows <= 0 || cols <= 0) return;
  dim3 grid(rows);
  dim3 block(256);
  row_variance_kernel<<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(d_mat, d_mean, rows, cols, d_valid_lens, d_out_rows);
#endif
}

extern "C" void launch_unpack_eff_bits_kernel(const uint64_t* d_packed,
                                              int eff_bits,
                                              uint64_t* d_out,
                                              size_t n,
                                              void* stream) {
#ifndef SUF_HAVE_CUDA
  (void)d_packed; (void)eff_bits; (void)d_out; (void)n; (void)stream;
#else
  if (n == 0) return;
  constexpr int kBlock = 256;
  int grid = static_cast<int>((n + kBlock - 1) / kBlock);
  unpack_eff_bits_kernel<<<grid, kBlock, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      d_packed, eff_bits, d_out, n);
#endif
}

extern "C" void launch_pack_eff_bits_kernel(const uint64_t* d_in,
                                            int eff_bits,
                                            uint64_t* d_packed,
                                            size_t n,
                                            void* stream) {
#ifndef SUF_HAVE_CUDA
  (void)d_in; (void)eff_bits; (void)d_packed; (void)n; (void)stream;
#else
  if (n == 0) return;
  if (eff_bits <= 0 || eff_bits > 64) return;
  unsigned __int128 total_bits = static_cast<unsigned __int128>(n) *
                                 static_cast<unsigned __int128>(eff_bits);
  size_t packed_words = static_cast<size_t>((total_bits + 63) / 64);
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  // Prefer a wordwise pack kernel (no atomics) for typical widths; fall back to
  // atomic packing for tiny widths.
  constexpr int kBlock = 256;
  if (eff_bits >= 17) {
    int grid = static_cast<int>((packed_words + kBlock - 1) / kBlock);
    pack_eff_bits_wordwise_kernel<<<grid, kBlock, 0, s>>>(d_in, eff_bits, d_packed, n, packed_words);
  } else {
    cudaMemsetAsync(d_packed, 0, packed_words * sizeof(uint64_t), s);
    int grid = static_cast<int>((n + kBlock - 1) / kBlock);
    pack_eff_bits_kernel<<<grid, kBlock, 0, s>>>(d_in, eff_bits, d_packed, n);
  }
#endif
}

extern "C" void launch_open_add_to_signed_kernel(const uint64_t* d_local_share,
                                                 const uint64_t* d_remote_share,
                                                 uint64_t* d_out_signed,
                                                 size_t n,
                                                 int ring_bits,
                                                 uint64_t ring_mask,
                                                 void* stream) {
#ifndef SUF_HAVE_CUDA
  (void)d_local_share; (void)d_remote_share; (void)d_out_signed; (void)n;
  (void)ring_bits; (void)ring_mask; (void)stream;
#else
  if (n == 0) return;
  if (ring_bits <= 0) ring_bits = 1;
  if (ring_bits > 64) ring_bits = 64;
  constexpr int kBlock = 256;
  int grid = static_cast<int>((n + kBlock - 1) / kBlock);
  open_add_to_signed_kernel<<<grid, kBlock, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      d_local_share, d_remote_share, d_out_signed, n, ring_bits, ring_mask);
#endif
}

extern "C" void launch_unpack_add_to_signed_kernel(const uint64_t* d_local_share,
                                                   const uint64_t* d_packed_remote,
                                                   int eff_bits,
                                                   uint64_t* d_out_signed,
                                                   size_t n,
                                                   int ring_bits,
                                                   uint64_t ring_mask,
                                                   void* stream) {
#ifndef SUF_HAVE_CUDA
  (void)d_local_share; (void)d_packed_remote; (void)eff_bits; (void)d_out_signed; (void)n;
  (void)ring_bits; (void)ring_mask; (void)stream;
#else
  if (n == 0) return;
  if (eff_bits <= 0) eff_bits = 1;
  if (eff_bits > 64) eff_bits = 64;
  if (ring_bits <= 0) ring_bits = 1;
  if (ring_bits > 64) ring_bits = 64;
  unsigned __int128 total_bits = static_cast<unsigned __int128>(n) *
                                 static_cast<unsigned __int128>(eff_bits);
  size_t packed_words = static_cast<size_t>((total_bits + 63) / 64);
  constexpr int kBlock = 256;
  int grid = static_cast<int>((n + kBlock - 1) / kBlock);
  unpack_add_to_signed_kernel<<<grid, kBlock, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      d_local_share, d_packed_remote, eff_bits, packed_words, d_out_signed, n, ring_bits, ring_mask);
#endif
}

extern "C" void launch_unpack_add_mod_kernel(const uint64_t* d_local_share,
                                             const uint64_t* d_packed_remote,
                                             int eff_bits,
                                             uint64_t* d_out_mod,
                                             size_t n,
                                             int ring_bits,
                                             uint64_t ring_mask,
                                             void* stream) {
#ifndef SUF_HAVE_CUDA
  (void)d_local_share; (void)d_packed_remote; (void)eff_bits; (void)d_out_mod; (void)n;
  (void)ring_bits; (void)ring_mask; (void)stream;
#else
  if (n == 0) return;
  if (eff_bits <= 0) eff_bits = 1;
  if (eff_bits > 64) eff_bits = 64;
  if (ring_bits <= 0) ring_bits = 1;
  if (ring_bits > 64) ring_bits = 64;
  unsigned __int128 total_bits = static_cast<unsigned __int128>(n) *
                                 static_cast<unsigned __int128>(eff_bits);
  size_t packed_words = static_cast<size_t>((total_bits + 63) / 64);
  constexpr int kBlock = 256;
  int grid = static_cast<int>((n + kBlock - 1) / kBlock);
  unpack_add_mod_kernel<<<grid, kBlock, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      d_local_share, d_packed_remote, eff_bits, packed_words, d_out_mod, n, ring_bits, ring_mask);
#endif
}
