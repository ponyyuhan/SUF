#pragma once

#include <cstddef>
#include <cstdint>
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
        const T* data() const { return data_; }
        const T& operator[](std::size_t i) const { return data_[i]; }
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

struct OpenHandle {
  size_t offset = static_cast<size_t>(-1);
  size_t len = 0;
};

// Minimal batched-open collector: gathers multiple additive-share vectors and
// performs one send/recv burst on flush. Not optimized for packing; intended
// as a structural step toward phase-level open fusion.
class OpenCollector {
 public:
  // Enqueue a buffer of local shares to be opened; returns a handle to view later.
  OpenHandle enqueue(const std::vector<uint64_t>& diff);

  // Flush all enqueued opens over the channel; results become available via view().
  void flush(int party, net::Chan& ch);

  // View opened values for a handle. Valid until next clear/flush.
  std::span<const int64_t> view(const OpenHandle& h) const;

  bool empty() const { return requests_.empty(); }
  bool has_pending() const { return !requests_.empty(); }
  bool ready(const OpenHandle& h) const;

  void clear();

 private:
  struct Request {
    std::vector<uint64_t> diff;
    size_t offset = 0;
    size_t len = 0;
  };
  std::vector<Request> requests_;
  std::vector<int64_t> opened_;
  bool opened_valid_ = false;
};

}  // namespace runtime
