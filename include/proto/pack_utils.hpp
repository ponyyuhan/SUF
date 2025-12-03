#pragma once

#include "proto/pfss_backend.hpp"
#include <vector>
#include <stdexcept>

namespace proto {

// Flatten a list of equal-size keys into a single buffer [N][key_bytes].
inline std::vector<uint8_t> pack_keys_flat(const std::vector<FssKey>& keys) {
  if (keys.empty()) return {};
  size_t kb = keys.front().bytes.size();
  for (const auto& k : keys) {
    if (k.bytes.size() != kb) throw std::runtime_error("pack_keys_flat: key size mismatch");
  }
  std::vector<uint8_t> flat(keys.size() * kb);
  for (size_t i = 0; i < keys.size(); i++) {
    std::memcpy(flat.data() + i * kb, keys[i].bytes.data(), kb);
  }
  return flat;
}

// Pack cut keys by cut index: cut_keys[c][i] -> flat buffer per cut (for GPU coalescing).
inline std::vector<std::vector<uint8_t>> pack_cut_keys_by_cut(
    const std::vector<std::vector<FssKey>>& cut_keys) {
  // cut_keys has shape [C][N]
  std::vector<std::vector<uint8_t>> flats;
  flats.reserve(cut_keys.size());
  for (const auto& v : cut_keys) flats.push_back(pack_keys_flat(v));
  return flats;
}

}  // namespace proto
