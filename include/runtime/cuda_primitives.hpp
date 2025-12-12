#pragma once

#include <cstddef>
#include <cstdint>

// Lightweight CUDA primitives used by the PFSS GPU path and CUDA-gated tests.
// These wrappers launch simple kernels defined in cuda/cuda_primitives.cu.

#ifdef __cplusplus
extern "C" {
#endif

// Beaver multiplication: z = c + d*b + e*a (+ d*e if party==0), all mod 2^64.
// Inputs/outputs are device pointers. d_open/e_open are the opened masks.
void launch_beaver_mul_kernel(int party,
                              const uint64_t* d_x,
                              const uint64_t* d_y,
                              const uint64_t* d_a,
                              const uint64_t* d_b,
                              const uint64_t* d_c,
                              const uint64_t* d_d_open,
                              const uint64_t* d_e_open,
                              uint64_t* d_out,
                              size_t n,
                              void* stream /* cudaStream_t */);

// Faithful truncation: y = x >> frac_bits (unsigned, wrap-free).
// Inputs/outputs are device pointers. No carry handling; intended for sanity tests.
void launch_trunc_shift_kernel(const uint64_t* d_in,
                               uint64_t* d_out,
                               int frac_bits,
                               size_t n,
                               void* stream /* cudaStream_t */);

// Faithful/GapARS truncation postproc on device. Assumes PFSS arith payload holds base share
// at arith_idx, and bool payload holds carry/sign bits at given indices (if present).
// hatx_public is the opened masked input (needed for top bits and wrap correction).
void launch_trunc_postproc_kernel(int party,
                                  int kind_gapars,       // 0 = faithful, 1 = GapARS
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
                                  void* stream /* cudaStream_t */);

// Horner cubic: y = (((c3*x)+c2)*x + c1)*x + c0 (mod 2^64).
// Inputs: device arrays of length n for x, c0..c3. Output d_out length n.
void launch_horner_cubic_kernel(const uint64_t* d_x,
                                const uint64_t* d_c0,
                                const uint64_t* d_c1,
                                const uint64_t* d_c2,
                                const uint64_t* d_c3,
                                uint64_t* d_out,
                                size_t n,
                                void* stream /* cudaStream_t */);

// Row-broadcast Beaver mul: mat is rows*cols, vec is rows (broadcast per row).
// A,C are Beaver triple shares shaped like mat (rows*cols); B is per-row (rows).
// d_open is opened (mat - A) per element; e_open is opened (vec - B) per row.
// All pointers are device, output is mat-sized. Optional valid_lens can zero
// columns >= valid_lens[row] (ragged).
void launch_row_broadcast_mul_kernel(int party,
                                     const uint64_t* d_A,
                                     const uint64_t* d_B_rows,
                                     const uint64_t* d_C,
                                     const uint64_t* d_d_open,
                                     const uint64_t* d_e_open_rows,
                                     int rows,
                                     int cols,
                                     const int* d_valid_lens,  // optional, can be nullptr
                                     uint64_t* d_out,
                                     size_t n,  // rows*cols
                                     void* stream /* cudaStream_t */);

// Row reduction: sum each row of a dense matrix (rows*cols) into out_rows.
// All pointers are device. Optional valid_lens per row (device) to handle ragged rows.
void launch_row_sum_kernel(const uint64_t* d_mat,
                           int rows,
                           int cols,
                           const int* d_valid_lens,  // optional, can be nullptr for dense
                           uint64_t* d_out_rows,
                           void* stream /* cudaStream_t */);

// Row mean: mean = sum / len (len from valid_lens or cols). Operates mod 2^64.
void launch_row_mean_kernel(const uint64_t* d_mat,
                            int rows,
                            int cols,
                            const int* d_valid_lens,  // optional, can be nullptr
                            uint64_t* d_out_rows,
                            void* stream /* cudaStream_t */);

// Row variance helper: out_rows[r] = sum((x - mean)^2) over the active columns.
// mean provided per row (device). Expects mat in Qf, mean in Qf, outputs sumsq
// in Q2f (mod 2^64). Caller is responsible for any scaling by 1/len.
void launch_row_variance_kernel(const uint64_t* d_mat,
                                const uint64_t* d_mean,
                                int rows,
                                int cols,
                                const int* d_valid_lens,  // optional, can be nullptr
                                uint64_t* d_out_rows,
                                void* stream /* cudaStream_t */);

// Unpack a dense bitstream of fixed-width integers into u64 values on device.
// d_packed holds packed words (little-endian within each word). Output length n.
void launch_unpack_eff_bits_kernel(const uint64_t* d_packed,
                                   int eff_bits,
                                   uint64_t* d_out,
                                   size_t n,
                                   void* stream /* cudaStream_t */);

#ifdef __cplusplus
}  // extern "C"
#endif
