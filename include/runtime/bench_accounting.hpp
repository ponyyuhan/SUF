#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace runtime::bench {

enum class OfflineBytesKind : uint8_t {
  CompositeTape = 0,
  MatmulTriple = 1,
  BeaverTriple = 2,
  RowBroadcastTriple = 3,
  Other = 4,
  kCount = 5,
};

void reset_offline_bytes();
void set_offline_counting_enabled(bool enabled);
bool offline_counting_enabled();

void add_offline_bytes(OfflineBytesKind kind, uint64_t bytes);
uint64_t offline_bytes_total();
uint64_t offline_bytes(OfflineBytesKind kind);
std::array<uint64_t, static_cast<size_t>(OfflineBytesKind::kCount)> offline_bytes_snapshot();

}  // namespace runtime::bench
