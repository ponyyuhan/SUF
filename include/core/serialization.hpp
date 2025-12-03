#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace core {

// Minimal helpers for little-endian encoding/decoding to byte vectors.
inline void append_u32(std::vector<uint8_t>& out, uint32_t v) {
  for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xff));
}

inline void append_u64(std::vector<uint8_t>& out, uint64_t v) {
  for (int i = 0; i < 8; ++i) out.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xff));
}

inline uint32_t read_u32(const std::vector<uint8_t>& buf, size_t& offset) {
  if (offset + 4 > buf.size()) throw std::runtime_error("serialization: truncated u32");
  uint32_t v = 0;
  for (int i = 0; i < 4; ++i) v |= static_cast<uint32_t>(buf[offset + i]) << (8 * i);
  offset += 4;
  return v;
}

inline uint64_t read_u64(const std::vector<uint8_t>& buf, size_t& offset) {
  if (offset + 8 > buf.size()) throw std::runtime_error("serialization: truncated u64");
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(buf[offset + i]) << (8 * i);
  offset += 8;
  return v;
}

}  // namespace core
