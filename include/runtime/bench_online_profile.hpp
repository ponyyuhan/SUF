#pragma once

#include <array>
#include <atomic>
#include <cstdint>

namespace runtime::bench {

enum class OnlineTimeKind : uint8_t {
  OpenFlushTotal = 0,
  OpenPack = 1,
  OpenComm = 2,
  OpenScatter = 3,
  PfssFlushEvalTotal = 4,
  PfssFlushEvalStageHatx = 5,
  PfssFlushEvalEval = 6,
  PfssFlushEvalStageOut = 7,
  PfssFinalizeTotal = 8,
  PfssMaterializeHost = 9,
  PfssPredEval = 10,
  PfssCoeffEval = 11,
  MatmulTotal = 12,
  MatmulUpload = 13,
  MatmulKernel = 14,
  MatmulDownload = 15,
  kCount = 16,
};

struct OnlineProfileSnapshot {
  std::array<uint64_t, static_cast<size_t>(OnlineTimeKind::kCount)> ns{};
};

inline std::atomic<bool> g_online_profile_enabled{false};
inline std::array<std::atomic<uint64_t>, static_cast<size_t>(OnlineTimeKind::kCount)> g_online_profile_ns{};

inline void reset_online_profile() {
  for (auto& x : g_online_profile_ns) x.store(0, std::memory_order_relaxed);
}

inline void set_online_profiling_enabled(bool enabled) {
  g_online_profile_enabled.store(enabled, std::memory_order_relaxed);
}

inline bool online_profiling_enabled() {
  return g_online_profile_enabled.load(std::memory_order_relaxed);
}

inline void add_online_ns(OnlineTimeKind kind, uint64_t ns) {
  if (!online_profiling_enabled() || ns == 0) return;
  const size_t idx = static_cast<size_t>(kind);
  if (idx >= g_online_profile_ns.size()) return;
  g_online_profile_ns[idx].fetch_add(ns, std::memory_order_relaxed);
}

inline OnlineProfileSnapshot snapshot_online_profile() {
  OnlineProfileSnapshot out{};
  for (size_t i = 0; i < out.ns.size(); ++i) {
    out.ns[i] = g_online_profile_ns[i].load(std::memory_order_relaxed);
  }
  return out;
}

}  // namespace runtime::bench
