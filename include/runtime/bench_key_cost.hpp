#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "runtime/bench_accounting.hpp"

#include "compiler/truncation_lowering.hpp"
#include "gates/composite_fss.hpp"

namespace runtime::bench {

using OfflineBytesArray = std::array<uint64_t, static_cast<size_t>(OfflineBytesKind::kCount)>;

inline uint64_t composite_party_key_bytes_no_triples(const gates::CompositePartyKey& k) {
  uint64_t bytes = 0;
  bytes += 8;  // r_in_share
  bytes += 4 + static_cast<uint64_t>(k.r_in_share_vec.size()) * 8;
  bytes += 4 + static_cast<uint64_t>(k.r_out_share.size()) * 8;
  bytes += 4 + static_cast<uint64_t>(k.wrap_share.size()) * 8;
  bytes += 8;  // r_hi_share
  bytes += 8;  // wrap_sign_share
  bytes += 4 + static_cast<uint64_t>(k.extra_params.size()) * 8;

  // pred_meta / cut_pred_meta / coeff_meta fixed fields.
  bytes += 14;

  auto add_keys = [&](const std::vector<proto::FssKey>& ks) {
    bytes += 4;
    for (const auto& fk : ks) {
      bytes += 4;
      bytes += static_cast<uint64_t>(fk.bytes.size());
    }
  };
  add_keys(k.pred_keys);
  add_keys(k.cut_pred_keys);
  add_keys(k.coeff_keys);

  bytes += 4 + static_cast<uint64_t>(k.base_coeff_share.size()) * 8;
  bytes += 4 + static_cast<uint64_t>(k.total_delta_share.size()) * 8;
  return bytes;
}

inline uint64_t composite_party_key_bytes_triples(const gates::CompositePartyKey& k) {
  uint64_t bytes = 0;
  bytes += 4 + static_cast<uint64_t>(k.triples.size()) * 24;
  bytes += 4 + static_cast<uint64_t>(k.bit_triples.size()) * 3;
  return bytes;
}

inline OfflineBytesArray composite_keypair_cost(const gates::CompositeKeyPair& kp) {
  OfflineBytesArray out{};
  const uint64_t c0 = composite_party_key_bytes_no_triples(kp.k0);
  const uint64_t c1 = composite_party_key_bytes_no_triples(kp.k1);
  const uint64_t t0 = composite_party_key_bytes_triples(kp.k0);
  const uint64_t t1 = composite_party_key_bytes_triples(kp.k1);
  // Benchmark accounting targets "dealer output size" (bytes across both parties),
  // matching SIGMA's `dealer.txt` key size object.
  out[static_cast<size_t>(OfflineBytesKind::CompositeTape)] = c0 + c1;
  out[static_cast<size_t>(OfflineBytesKind::BeaverTriple)] = t0 + t1;
  return out;
}

inline OfflineBytesArray truncation_lowering_cost(const compiler::TruncationLoweringResult& res) {
  OfflineBytesArray out = composite_keypair_cost(res.keys);
  for (const auto& pe : res.per_elems) {
    OfflineBytesArray c = composite_keypair_cost(pe.keys);
    for (size_t i = 0; i < out.size(); ++i) out[i] += c[i];
  }
  return out;
}

inline void charge_offline_bytes(const OfflineBytesArray& cost) {
  for (size_t i = 0; i < cost.size(); ++i) {
    add_offline_bytes(static_cast<OfflineBytesKind>(i), cost[i]);
  }
}

}  // namespace runtime::bench
