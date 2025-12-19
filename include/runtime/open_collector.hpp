#pragma once

#include <atomic>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>
#if __has_include(<span>)
  #include <span>
#else
  #if !defined(SUF_SPAN_FALLBACK_DEFINED)
    #define SUF_SPAN_FALLBACK_DEFINED
    namespace std {
      template<typename T>
      class span {
       public:
        span() : data_(nullptr), size_(0) {}
        span(const T* ptr, std::size_t n) : data_(ptr), size_(n) {}
        template <typename U, typename = std::enable_if_t<std::is_same_v<std::remove_const_t<T>, U>>>
        span(const std::vector<U>& v) : data_(v.data()), size_(v.size()) {}
        std::size_t size() const { return size_; }
        bool empty() const { return size_ == 0; }
        const T* data() const { return data_; }
        T* data() { return const_cast<T*>(data_); }
        const T& operator[](std::size_t i) const { return data_[i]; }
        T& operator[](std::size_t i) { return const_cast<T&>(data_[i]); }
        span subspan(std::size_t off, std::size_t n) const {
          if (off > size_) return span();
          std::size_t len = (off + n > size_) ? (size_ - off) : n;
          return span(data_ + off, len);
        }
        const T* begin() const { return data_; }
        const T* end() const { return data_ + size_; }
       private:
        const T* data_;
        std::size_t size_;
      };
      template <typename T>
      const T* begin(span<T> s) { return s.data(); }
      template <typename T>
      const T* end(span<T> s) { return s.data() + s.size(); }
    }
  #endif
#endif

#include "mpc/net.hpp"

namespace runtime {

// Categorize openings for benchmark statistics.
enum class OpenKind : uint8_t {
  kOther = 0,
  kBeaver = 1,  // Beaver-style opens: (x-a), (y-b), matmul (A-a,B-b), row-broadcast, etc.
  kMask = 2,    // Masked-value opens: x_hat = x + r_in (for PFSS predicate/coeff evaluation).
  kCount = 3,
};

struct OpenSlot {
  std::vector<int64_t> opened;
  size_t n = 0;
  std::atomic<bool> ready{false};
};

struct OpenHandle {
  std::shared_ptr<OpenSlot> slot;
  size_t offset = 0;
  size_t len = 0;
};

// Minimal batched-open collector: gathers multiple additive-share vectors and
// performs one send/recv burst on flush. Optional env-gated eff-bits packing
// can reduce traffic when shares are known to have small magnitude:
// - enable: `SUF_OPEN_PACK_EFFBITS=1`
// - cap bits: `SUF_OPEN_PACK_MAX_BITS` (default 56)
// - auto pack: `SUF_OPEN_PACK_AUTO=1` with `SUF_OPEN_PACK_MIN_SAVINGS_PCT`
// - dynamic bits: `SUF_OPEN_PACK_DYNAMIC=1` (uses per-flush max bitwidth)
class OpenCollector {
 public:
  struct Stats {
    size_t flushes = 0;
    size_t opened_words = 0;
    std::array<size_t, static_cast<size_t>(OpenKind::kCount)> opened_words_by_kind{};
    size_t max_pending_words = 0;
    uint64_t flush_ns = 0;
  };
  struct Limits {
    size_t max_pending_words = 1ull << 24;  // generous guardrail
  };

  ~OpenCollector();

  // Enqueue a buffer of local shares to be opened; returns a handle to view later.
  OpenHandle enqueue(const std::vector<uint64_t>& diff, OpenKind kind = OpenKind::kOther);

  // Flush all enqueued opens over the channel; results become available via view().
  void flush(int party, net::Chan& ch);

  // Pending words currently buffered (before flush).
  size_t pending_words() const { return pending_words_; }
  const Limits& limits() const { return limits_; }

  // View opened values for a handle. Valid until next clear/flush.
  std::span<const int64_t> view(const OpenHandle& h) const;

  bool empty() const { return requests_.empty(); }
  bool has_pending() const { return !requests_.empty(); }
  bool ready(const OpenHandle& h) const;

  void clear();

  void set_limits(const Limits& lim) { limits_ = lim; }

  const Stats& stats() const { return stats_; }
  void reset_stats() { stats_ = Stats{}; }

 private:
  struct Request {
    std::vector<uint64_t> diff;
    std::shared_ptr<OpenSlot> slot;
    size_t offset = 0;
    size_t len = 0;
    OpenKind kind = OpenKind::kOther;
  };
  std::vector<Request> requests_;
  Stats stats_;
  Limits limits_;
  size_t pending_words_ = 0;
  // Scratch buffers to avoid per-flush allocations in hot paths.
  std::vector<uint64_t> send_flat_buf_;
  std::vector<uint64_t> recv_flat_buf_;
  std::vector<uint64_t> other_flat_buf_;
  std::vector<uint64_t> packed_local_buf_;
  std::vector<uint64_t> packed_remote_buf_;
#ifdef SUF_HAVE_CUDA
  struct DevicePackScratch {
    uint64_t* d_in = nullptr;
    uint64_t* d_packed = nullptr;
    uint64_t* d_out = nullptr;
    size_t in_cap = 0;
    size_t packed_cap = 0;
    size_t out_cap = 0;
  };
  DevicePackScratch pack_scratch_;
#endif
};

}  // namespace runtime
