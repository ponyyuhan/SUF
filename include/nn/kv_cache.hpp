#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include "nn/tensor_view.hpp"

namespace nn {

struct KVCache {
  size_t B = 0, H = 0, S_max = 0, Dh = 0;
  size_t cur_len = 0;
  std::vector<uint64_t> K_share;
  std::vector<uint64_t> V_share;

  KVCache() = default;
  KVCache(size_t B_, size_t H_, size_t Smax_, size_t Dh_)
      : B(B_), H(H_), S_max(Smax_), Dh(Dh_),
        K_share(B_ * H_ * Smax_ * Dh_, 0),
        V_share(B_ * H_ * Smax_ * Dh_, 0) {}

  TensorView<uint64_t> view_K() { return view4(K_share.data(), B, H, S_max, Dh); }
  TensorView<uint64_t> view_V() { return view4(V_share.data(), B, H, S_max, Dh); }
};

inline uint64_t* kv_head_ptr(KVCache& cache, size_t b, size_t h) {
  return cache.K_share.data() + ((b * cache.H + h) * cache.S_max * cache.Dh);
}

inline const uint64_t* kv_head_ptr(const KVCache& cache, size_t b, size_t h) {
  return cache.K_share.data() + ((b * cache.H + h) * cache.S_max * cache.Dh);
}

inline uint64_t* kv_head_ptr_v(KVCache& cache, size_t b, size_t h) {
  return cache.V_share.data() + ((b * cache.H + h) * cache.S_max * cache.Dh);
}

inline const uint64_t* kv_head_ptr_v(const KVCache& cache, size_t b, size_t h) {
  return cache.V_share.data() + ((b * cache.H + h) * cache.S_max * cache.Dh);
}

inline void kv_append_token(KVCache& cache,
                            const TensorView<uint64_t>& K_t,
                            const TensorView<uint64_t>& V_t) {
  if (cache.cur_len >= cache.S_max) return;
  size_t slot = cache.cur_len;
  for (size_t b = 0; b < cache.B; ++b) {
    for (size_t h = 0; h < cache.H; ++h) {
      for (size_t d = 0; d < cache.Dh; ++d) {
        size_t src = (b * cache.H + h) * cache.Dh + d;
        size_t dst = ((b * cache.H + h) * cache.S_max + slot) * cache.Dh + d;
        cache.K_share[dst] = K_t.data[src];
        cache.V_share[dst] = V_t.data[src];
      }
    }
  }
  cache.cur_len += 1;
}

}  // namespace nn
