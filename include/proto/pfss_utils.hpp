#pragma once

#include "proto/common.hpp"
#include "proto/pfss_backend.hpp"
#include <stdexcept>

namespace proto {

inline u64 eval_u64_share_from_dcf(const PfssBackend& fss, int in_bits,
                                  const FssKey& kb, u64 x) {
  auto out = fss.eval_dcf(in_bits, kb, fss.u64_to_bits_msb(x, in_bits));
  if (out.size() < 8) return 0;
  u64 val = 0;
  std::memcpy(&val, out.data(), std::min<size_t>(8, out.size()));
  return val;
}

inline std::vector<u64> eval_vec_u64_from_dcf(const PfssBackend& fss, int in_bits,
                                              const FssKey& kb, u64 x, int out_words) {
  auto out = fss.eval_dcf(in_bits, kb, fss.u64_to_bits_msb(x, in_bits));
  std::vector<u64> v(static_cast<size_t>(out_words), 0);
  size_t want = static_cast<size_t>(out_words) * 8;
  size_t copy = std::min(want, out.size());
  std::memcpy(v.data(), out.data(), copy);
  return v;
}

}  // namespace proto
