#include "runtime/cuda_primitives.hpp"
#include <cuda_runtime.h>

namespace {

__device__ inline uint64_t add_mod(uint64_t a, uint64_t b) { return a + b; }
__device__ inline uint64_t mul_mod(uint64_t a, uint64_t b) {
  unsigned __int128 p = static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b);
  return static_cast<uint64_t>(p);
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
                                      uint64_t r_in,
                                      const uint64_t* __restrict__ hatx_public,
                                      const uint64_t* __restrict__ arith,
                                      size_t arith_stride,
                                      int arith_idx,
                                      const uint64_t* __restrict__ bools,
                                      size_t bool_stride,
                                      int carry_idx,
                                      int sign_idx,
                                      uint64_t* __restrict__ out,
                                      size_t n) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  uint64_t base = arith[arith_idx + idx * arith_stride];
  bool have_carry = bools && carry_idx >= 0 && carry_idx < static_cast<int>(bool_stride);
  bool have_sign = bools && sign_idx >= 0 && sign_idx < static_cast<int>(bool_stride);
  uint64_t carry = have_carry ? bools[carry_idx + idx * bool_stride] : 0ull;
  uint64_t sign = have_sign ? bools[sign_idx + idx * bool_stride] : 0ull;
  uint64_t y = base;
  if (party == 0 && frac_bits > 0 && frac_bits < 64) {
    uint64_t top = hatx_public ? (hatx_public[idx] >> frac_bits) : 0ull;
    y = add_mod(y, top);
  }
  y = add_mod(y, (~r_hi_share) + 1); // subtract r_hi_share
  // Carry is always subtracted (faithful and GapARS share the same postproc).
  y = add_mod(y, (~carry) + 1);      // subtract carry
  if (frac_bits > 0 && frac_bits < 64 && hatx_public && party == 0) {
    uint64_t modulus = (uint64_t(1) << (64 - frac_bits));
    if (hatx_public[idx] < r_in) {
      y = add_mod(y, modulus);
    }
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
                                             uint64_t r_in,
                                             const uint64_t* d_hatx_public,
                                             const uint64_t* d_arith,
                                             size_t arith_stride,
                                             int arith_idx,
                                             const uint64_t* d_bools,
                                             size_t bool_stride,
                                             int carry_idx,
                                             int sign_idx,
                                             uint64_t* d_out,
                                             size_t n,
                                             void* stream) {
#ifndef SUF_HAVE_CUDA
  (void)party; (void)kind_gapars; (void)frac_bits; (void)r_hi_share; (void)r_in;
  (void)d_hatx_public; (void)d_arith; (void)arith_stride; (void)arith_idx;
  (void)d_bools; (void)bool_stride; (void)carry_idx; (void)sign_idx; (void)d_out; (void)n; (void)stream;
#else
  if (n == 0) return;
  constexpr int kBlock = 256;
  int grid = static_cast<int>((n + kBlock - 1) / kBlock);
  trunc_postproc_kernel<<<grid, kBlock, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      party, kind_gapars, frac_bits, r_hi_share, r_in,
      d_hatx_public, d_arith, arith_stride, arith_idx,
      d_bools, bool_stride, carry_idx, sign_idx,
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
