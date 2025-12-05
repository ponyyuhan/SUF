#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "mpc/net.hpp"

namespace runtime {

struct OpenHandle {
  size_t offset = static_cast<size_t>(-1);
  size_t len = 0;
};

struct OpenView {
  const int64_t* data = nullptr;
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
  OpenView view(const OpenHandle& h) const;

  bool empty() const { return requests_.empty(); }

  void clear();

 private:
  struct Request {
    std::vector<uint64_t> diff;
    size_t offset = 0;
    size_t len = 0;
  };
  std::vector<Request> requests_;
  std::vector<int64_t> opened_;
};

}  // namespace runtime
