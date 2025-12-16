#pragma once

#include <cstddef>
#include <cstdint>

namespace net {

// Minimal synchronous channel abstraction; demos can provide in-memory pipes.
struct Chan {
  virtual void send_u64(uint64_t) = 0;
  virtual uint64_t recv_u64() = 0;
  // Optional bulk operations (default loops through scalar send/recv).
  virtual void send_u64s(const uint64_t* data, size_t n) {
    for (size_t i = 0; i < n; ++i) send_u64(data[i]);
  }
  virtual void recv_u64s(uint64_t* data, size_t n) {
    for (size_t i = 0; i < n; ++i) data[i] = recv_u64();
  }
  virtual ~Chan() = default;
};

}  // namespace net
