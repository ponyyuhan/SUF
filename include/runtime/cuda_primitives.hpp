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

// Beaver multiplication (AoS triples): z = c + d*b + e*a (+ d*e if party==0), all mod 2^64.
// `d_triples` points to an array of triples with layout {a,b,c} per element.
void launch_beaver_mul_aos_kernel(int party,
                                  const void* d_triples,  // array of {u64 a,b,c}
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

// In-place add a small per-column mask (AoS layout): out[i*out_words + j] += r_out[j].
// Supports out_words up to 8 efficiently; for larger out_words callers should fall back.
void launch_add_rout_aos_kernel(uint64_t* d_out,
                                size_t elems,
                                int out_words,
                                const uint64_t* r_out,   // host pointer to r_out words
                                size_t r_out_words,
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

// Row reduction for ragged packed rows: `d_vals` stores concatenated rows
// back-to-back, and `d_row_offsets` stores prefix sums of lengths with
// `d_row_offsets[r]` as start and `d_row_offsets[r+1]` as end.
// All pointers are device. `d_row_offsets` must have length rows+1.
void launch_row_sum_ragged_kernel(const uint64_t* d_vals,
                                  const int* d_row_offsets,
                                  int rows,
                                  uint64_t* d_out_rows,
                                  void* stream /* cudaStream_t */);

// 2D Beaver matmul (dense): out = C + D*B + A*E (+ D*E if party==0) where
// D = open(A - A_tri) and E = open(B - B_tri). All arithmetic is mod 2^64.
//
// Shapes:
//   A, D, A_tri: [M x K] (row-major)
//   B, E, B_tri: [K x N] (row-major) if w_transposed==false
//                [N x K] (row-major) if w_transposed==true (i.e., B^T stored)
//   C, C_tri, out: [M x N]
//
// D_open and E_open must be provided as uint64 bit-patterns (opened ring elems).
void launch_beaver_matmul2d_kernel(int party,
                                   const uint64_t* d_D_open,        // M*K
                                   const uint64_t* d_E_open,        // K*N or N*K
                                   const uint64_t* d_A_tri_share,   // M*K
                                   const uint64_t* d_B_tri_share,   // K*N or N*K
                                   const uint64_t* d_C_tri_share,   // M*N
                                   int M,
                                   int K,
                                   int N,
                                   int w_transposed,                // 0/1
                                   int has_scale,                   // 0/1
                                   int64_t mul_const,               // only used if has_scale=1
                                   int mul_shift,                   // only used if has_scale=1
                                   uint64_t* d_out,                 // M*N
                                   void* stream /* cudaStream_t */);

// Batched variant where the *same* triple shares (A_tri,B_tri,C_tri) are reused across all
// batch instances. This matches the benchmark's shape-based triple caching behavior.
void launch_beaver_matmul2d_batched_kernel(int party,
                                           const uint64_t* d_D_open,        // batches*M*K
                                           const uint64_t* d_E_open,        // batches*(K*N or N*K)
                                           const uint64_t* d_A_tri_share,   // M*K
                                           const uint64_t* d_B_tri_share,   // K*N or N*K
                                           const uint64_t* d_C_tri_share,   // M*N
                                           int batches,
                                           int M,
                                           int K,
                                           int N,
                                           int w_transposed,                // 0/1
                                           int has_scale,                   // 0/1
                                           int64_t mul_const,
                                           int mul_shift,
                                           uint64_t* d_out,                 // batches*M*N
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

// Pack fixed-width integers into a dense bitstream on device.
// d_in holds n u64 values; d_packed must have ceil(n*eff_bits/64) words.
void launch_pack_eff_bits_kernel(const uint64_t* d_in,
                                 int eff_bits,
                                 uint64_t* d_packed,
                                 size_t n,
                                 void* stream /* cudaStream_t */);

// Combine additive shares on device and return a sign-extended (int64-compatible)
// representation matching proto::to_signed for the current ring bitwidth.
// This is intended to avoid host-side opened-value scatter when device packing is used.
void launch_open_add_to_signed_kernel(const uint64_t* d_local_share,
                                      const uint64_t* d_remote_share,
                                      uint64_t* d_out_signed,  // sign-extended bit-patterns
                                      size_t n,
                                      int ring_bits,
                                      uint64_t ring_mask,
                                      void* stream /* cudaStream_t */);

// Unpack `eff_bits`-packed remote shares, add to local shares, and sign-extend the sum to the
// current ring bitwidth. This fuses unpack + open_add_to_signed into one kernel to reduce
// global memory traffic and kernel launch overhead in the OpenCollector hot path.
void launch_unpack_add_to_signed_kernel(const uint64_t* d_local_share,
                                        const uint64_t* d_packed_remote,
                                        int eff_bits,
                                        uint64_t* d_out_signed,  // sign-extended bit-patterns
                                        size_t n,
                                        int ring_bits,
                                        uint64_t ring_mask,
                                        void* stream /* cudaStream_t */);

// Unpack `eff_bits`-packed remote shares, add to local shares, and mask to the ring bitwidth
// (raw u64 ring elements, no sign-extension). This is the preferred OpenCollector GPU path
// when downstream consumers only need modulo-2^n values.
void launch_unpack_add_mod_kernel(const uint64_t* d_local_share,
                                  const uint64_t* d_packed_remote,
                                  int eff_bits,
                                  uint64_t* d_out_mod,  // masked ring elements
                                  size_t n,
                                  int ring_bits,
                                  uint64_t ring_mask,
                                  void* stream /* cudaStream_t */);

#ifdef __cplusplus
}  // extern "C"
#endif
