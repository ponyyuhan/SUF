#pragma once

#include "proto/common.hpp"
#include "proto/pfss_backend_batch.hpp"
#include "proto/beaver_mul64.hpp"
#include "proto/bit_ring_ops.hpp"
#include <vector>

namespace proto {

struct GeluStepDCFKeysPacked {
  int d = 3;
  size_t N = 0;

  size_t key_bytes_sign = 0;
  const uint8_t* k_hat_lt_r = nullptr;   // [N][key_bytes_sign]
  const uint8_t* k_hat_lt_r2 = nullptr;  // [N][key_bytes_sign]
  const u64* wrap_sign_share = nullptr;    // [N] additive shares of wrap bit

  const u64* r_in_share = nullptr;       // [N]
  const u64* r_out_share = nullptr;      // [N]

  const u64* base_coeff_flat = nullptr;  // [N*(d+1)]

  int num_cuts = 0;
  size_t key_bytes_cut = 0;
  std::vector<const uint8_t*> cut_keys;  // size num_cuts, each [N][key_bytes_cut]
  const u64* delta_flat = nullptr;       // [num_cuts * N * (d+1)]

  const std::vector<BeaverTriple64Share>* triples64 = nullptr;
};

struct GeluBatchIO {
  const std::vector<u64>* hatx;   // [N] public
  std::vector<u64>* haty_share;   // [N] output shares
};

inline void gelu_eval_batch_step_dcf(
    int party,
    const PfssBackendBatch& fss,
    IChannel& ch,
    const GeluStepDCFKeysPacked& K,
    GeluBatchIO io) {
  const size_t N = io.hatx->size();
  const int out_bytes_bit = 8;

  std::vector<u64> x(N);
  for (size_t i = 0; i < N; i++) {
    x[i] = (party == 0) ? sub_mod((*io.hatx)[i], K.r_in_share[i])
                        : sub_mod(0ull, K.r_in_share[i]);
  }

  std::vector<uint8_t> out_a(N * out_bytes_bit), out_b(N * out_bytes_bit);
  fss.eval_dcf_many_u64(64, K.key_bytes_sign, K.k_hat_lt_r, *io.hatx, out_bytes_bit, out_a.data());
  fss.eval_dcf_many_u64(64, K.key_bytes_sign, K.k_hat_lt_r2, *io.hatx, out_bytes_bit, out_b.data());

  std::vector<u64> a(N), b(N);
  for (size_t i = 0; i < N; i++) {
    a[i] = unpack_u64_le(out_a.data() + 8 * i);
    b[i] = unpack_u64_le(out_b.data() + 8 * i);
  }

  BeaverMul64 mul{party, ch, *K.triples64, 0};
  BitRingOps B{party, mul};

  std::vector<u64> na(N), u(N), w(N);
  for (size_t i = 0; i < N; i++) na[i] = sub_mod((party == 0) ? 1ULL : 0ULL, a[i]);
  mul.mul_batch(b, na, u);
  for (size_t i = 0; i < N; i++) {
    u64 wrap_or = sub_mod(add_mod(na[i], b[i]), u[i]); // OR form
    w[i] = B.SEL(K.wrap_sign_share[i], wrap_or, u[i]);
  }

  std::vector<u64> coeff(N * (K.d + 1), 0);
  for (size_t i = 0; i < N; i++) {
    for (int k = 0; k <= K.d; k++) {
      coeff[i * (K.d + 1) + k] = (party == 0) ? K.base_coeff_flat[i * (K.d + 1) + k] : 0ull;
    }
  }

  const u64 TWO63 = (1ull << 63);
  std::vector<u64> hatx_bias(N);
  for (size_t i = 0; i < N; i++) hatx_bias[i] = add_mod((*io.hatx)[i], TWO63);

  for (int j = 0; j < K.num_cuts; j++) {
    std::vector<uint8_t> outv_bytes(N * 8 * (K.d + 1));
    fss.eval_dcf_many_u64(64, K.key_bytes_cut, K.cut_keys[j], hatx_bias,
                          8 * (K.d + 1), outv_bytes.data());
    for (size_t i = 0; i < N; i++) {
      for (int k = 0; k <= K.d; k++) {
        u64 outv = unpack_u64_le(outv_bytes.data() + (i * (K.d + 1) + k) * 8);
        u64 delta = K.delta_flat[((size_t)j * N + i) * (K.d + 1) + k];
        u64 addc = (party == 0) ? delta : 0ull;
        u64 term = sub_mod(addc, outv);
        coeff[i * (K.d + 1) + k] = add_mod(coeff[i * (K.d + 1) + k], term);
      }
    }
  }

  std::vector<u64> acc(N);
  for (size_t i = 0; i < N; i++) acc[i] = coeff[i * (K.d + 1) + K.d];

  for (int k = K.d - 1; k >= 0; k--) {
    std::vector<u64> prod(N);
    mul.mul_batch(acc, x, prod);
    for (size_t i = 0; i < N; i++) {
      acc[i] = add_mod(prod[i], coeff[i * (K.d + 1) + k]);
    }
  }
  std::vector<u64> delta = acc;

  std::vector<u64> xplus(N);
  mul.mul_batch(w, x, xplus);

  io.haty_share->resize(N);
  for (size_t i = 0; i < N; i++) {
    u64 y = add_mod(xplus[i], delta[i]);
    (*io.haty_share)[i] = add_mod(y, K.r_out_share[i]);
  }
}

}  // namespace proto
