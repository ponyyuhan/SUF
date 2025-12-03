#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace proto {

struct IChannel {
  virtual ~IChannel() = default;
  virtual void send_bytes(const void* data, size_t n) = 0;
  virtual void recv_bytes(void* data, size_t n) = 0;
};

inline void exchange_u64_vec(IChannel& ch,
                             const std::vector<uint64_t>& sendv,
                             std::vector<uint64_t>& recvv) {
  if (recvv.size() != sendv.size()) recvv.resize(sendv.size());
  ch.send_bytes(sendv.data(), sendv.size() * sizeof(uint64_t));
  ch.recv_bytes(recvv.data(), recvv.size() * sizeof(uint64_t));
}

}  // namespace proto
