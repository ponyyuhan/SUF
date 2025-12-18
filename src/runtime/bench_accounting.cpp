#include "runtime/bench_accounting.hpp"

#include <atomic>

namespace runtime::bench {

namespace {

std::atomic<bool> g_enabled{false};
std::array<std::atomic<uint64_t>, static_cast<size_t>(OfflineBytesKind::kCount)> g_bytes;

struct Init {
  Init() {
    for (auto& x : g_bytes) x.store(0, std::memory_order_relaxed);
  }
};

Init g_init;

}  // namespace

void reset_offline_bytes() {
  for (auto& x : g_bytes) x.store(0, std::memory_order_relaxed);
}

void set_offline_counting_enabled(bool enabled) {
  g_enabled.store(enabled, std::memory_order_relaxed);
}

bool offline_counting_enabled() {
  return g_enabled.load(std::memory_order_relaxed);
}

void add_offline_bytes(OfflineBytesKind kind, uint64_t bytes) {
  if (!offline_counting_enabled() || bytes == 0) return;
  const size_t idx = static_cast<size_t>(kind);
  if (idx >= g_bytes.size()) return;
  g_bytes[idx].fetch_add(bytes, std::memory_order_relaxed);
}

uint64_t offline_bytes(OfflineBytesKind kind) {
  const size_t idx = static_cast<size_t>(kind);
  if (idx >= g_bytes.size()) return 0;
  return g_bytes[idx].load(std::memory_order_relaxed);
}

uint64_t offline_bytes_total() {
  uint64_t sum = 0;
  for (size_t i = 0; i < static_cast<size_t>(OfflineBytesKind::kCount); ++i) {
    sum += g_bytes[i].load(std::memory_order_relaxed);
  }
  return sum;
}

std::array<uint64_t, static_cast<size_t>(OfflineBytesKind::kCount)> offline_bytes_snapshot() {
  std::array<uint64_t, static_cast<size_t>(OfflineBytesKind::kCount)> out{};
  for (size_t i = 0; i < out.size(); ++i) {
    out[i] = g_bytes[i].load(std::memory_order_relaxed);
  }
  return out;
}

}  // namespace runtime::bench
