#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>
#include <cstring>

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

// Fixed-size little-endian packing helpers used across proto/runtime.
inline std::vector<uint8_t> pack_u64_le(uint64_t x) {
  std::vector<uint8_t> out(8);
  std::memcpy(out.data(), &x, 8);
  return out;
}

inline uint64_t unpack_u64_le(const uint8_t* p) {
  uint64_t x;
  std::memcpy(&x, p, 8);
  return x;
}

inline std::vector<uint8_t> pack_u64_vec_le(const std::vector<uint64_t>& ws) {
  std::vector<uint8_t> out(ws.size() * 8);
  for (size_t i = 0; i < ws.size(); i++) {
    std::memcpy(out.data() + 8 * i, &ws[i], 8);
  }
  return out;
}

inline std::vector<uint64_t> unpack_u64_vec_le(const std::vector<uint8_t>& bytes) {
  if (bytes.size() % 8 != 0) throw std::runtime_error("unpack_u64_vec_le: size not multiple of 8");
  size_t n = bytes.size() / 8;
  std::vector<uint64_t> ws(n);
  for (size_t i = 0; i < n; i++) ws[i] = unpack_u64_le(bytes.data() + 8 * i);
  return ws;
}

}  // namespace core
