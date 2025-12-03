#pragma once

#include <cstdint>

namespace net {

// Minimal synchronous channel abstraction; demos can provide in-memory pipes.
struct Chan {
  virtual void send_u64(uint64_t) = 0;
  virtual uint64_t recv_u64() = 0;
  virtual ~Chan() = default;
};

}  // namespace net
