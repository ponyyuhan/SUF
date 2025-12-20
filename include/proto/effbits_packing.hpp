#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace proto {

namespace detail {

inline uint64_t mask_bits_u64(int bits) {
  if (bits <= 0) return 0ull;
  if (bits >= 64) return ~uint64_t(0);
  return (uint64_t(1) << bits) - 1ull;
}

inline size_t gcd_size_t(size_t a, size_t b) {
  while (b != 0) {
    size_t t = a % b;
    a = b;
    b = t;
  }
  return a;
}

}  // namespace detail

inline size_t packed_words_u64(size_t elems, int eff_bits) {
  if (eff_bits <= 0 || eff_bits > 64) return elems;
  if (eff_bits == 64) return elems;
  unsigned __int128 total_bits = static_cast<unsigned __int128>(elems) *
                                 static_cast<unsigned __int128>(eff_bits);
  total_bits += 63;
  return static_cast<size_t>(total_bits / 64);
}

inline size_t packed_bytes(size_t elems, int eff_bits) {
  if (eff_bits <= 0 || eff_bits > 64) return elems * sizeof(uint64_t);
  if (eff_bits == 64) return elems * sizeof(uint64_t);
  unsigned __int128 total_bits = static_cast<unsigned __int128>(elems) *
                                 static_cast<unsigned __int128>(eff_bits);
  total_bits += 7;
  return static_cast<size_t>(total_bits / 8);
}

// Packs the low `eff_bits` of each input word into a dense u64 buffer.
// The caller may send only the first `packed_bytes(elems, eff_bits)` bytes
// of the resulting buffer; the remaining bytes in the final word are zero.
inline void pack_eff_bits_u64_into(const uint64_t* xs,
                                   size_t elems,
                                   int eff_bits,
                                   std::vector<uint64_t>& out) {
  if (eff_bits <= 0 || eff_bits > 64) {
    throw std::runtime_error("pack_eff_bits_u64_into: eff_bits out of range");
  }
  if (eff_bits == 64) {
    out.assign(xs, xs + elems);
    return;
  }
  const size_t words = packed_words_u64(elems, eff_bits);
  out.resize(words);
  if (elems == 0) return;
  const uint64_t mask = detail::mask_bits_u64(eff_bits);
  const size_t g = detail::gcd_size_t(static_cast<size_t>(eff_bits), 64);
  const size_t block_in = 64 / g;
  const size_t block_out = (block_in * static_cast<size_t>(eff_bits)) / 64;
  const size_t full_blocks = elems / block_in;
  const size_t tail = elems - full_blocks * block_in;

#ifdef _OPENMP
#pragma omp parallel for if (full_blocks >= 8) schedule(static)
#endif
  for (size_t b = 0; b < full_blocks; ++b) {
    const size_t in_off = b * block_in;
    const size_t out_off = b * block_out;
    unsigned __int128 acc = 0;
    int acc_bits = 0;
    size_t out_w = 0;
    for (size_t i = 0; i < block_in; ++i) {
      uint64_t v = xs[in_off + i] & mask;
      acc |= (static_cast<unsigned __int128>(v) << acc_bits);
      acc_bits += eff_bits;
      while (acc_bits >= 64) {
        out[out_off + out_w++] = static_cast<uint64_t>(acc);
        acc >>= 64;
        acc_bits -= 64;
      }
    }
  }

  if (tail) {
    const size_t in_off = full_blocks * block_in;
    const size_t out_off = full_blocks * block_out;
    unsigned __int128 acc = 0;
    int acc_bits = 0;
    size_t out_w = 0;
    for (size_t i = 0; i < tail; ++i) {
      uint64_t v = xs[in_off + i] & mask;
      acc |= (static_cast<unsigned __int128>(v) << acc_bits);
      acc_bits += eff_bits;
      while (acc_bits >= 64) {
        if (out_off + out_w >= out.size()) break;
        out[out_off + out_w++] = static_cast<uint64_t>(acc);
        acc >>= 64;
        acc_bits -= 64;
      }
    }
    if (acc_bits > 0 && out_off + out_w < out.size()) {
      out[out_off + out_w++] = static_cast<uint64_t>(acc);
    }
  }

  // Clear any unused high bits in the last word so callers can safely send only
  // `packed_bytes(elems, eff_bits)` bytes without leaking stale data.
  const unsigned __int128 total_bits =
      static_cast<unsigned __int128>(elems) * static_cast<unsigned __int128>(eff_bits);
  const size_t last_bits = static_cast<size_t>((total_bits == 0) ? 0 : ((total_bits - 1) % 64 + 1));
  if (!out.empty() && last_bits > 0 && last_bits < 64) {
    out.back() &= (uint64_t(1) << last_bits) - 1ull;
  }
}

inline void unpack_eff_bits_u64_into(const uint64_t* packed,
                                     size_t packed_words,
                                     int eff_bits,
                                     size_t elems,
                                     std::vector<uint64_t>& out) {
  if (eff_bits <= 0 || eff_bits > 64) {
    throw std::runtime_error("unpack_eff_bits_u64_into: eff_bits out of range");
  }
  if (eff_bits == 64) {
    out.assign(packed, packed + elems);
    return;
  }
  out.resize(elems);
  if (elems == 0) return;
  const uint64_t mask = detail::mask_bits_u64(eff_bits);
  const size_t g = detail::gcd_size_t(static_cast<size_t>(eff_bits), 64);
  const size_t block_out = 64 / g;
  const size_t block_in = (block_out * static_cast<size_t>(eff_bits)) / 64;
  const size_t full_blocks = elems / block_out;
  const size_t tail = elems - full_blocks * block_out;

#ifdef _OPENMP
#pragma omp parallel for if (full_blocks >= 8) schedule(static)
#endif
  for (size_t b = 0; b < full_blocks; ++b) {
    const size_t out_off = b * block_out;
    const size_t in_off = b * block_in;
    unsigned __int128 acc = 0;
    int acc_bits = 0;
    size_t in_w = 0;
    for (size_t i = 0; i < block_out; ++i) {
      while (acc_bits < eff_bits) {
        uint64_t w = (in_off + in_w < packed_words) ? packed[in_off + in_w] : 0ull;
        ++in_w;
        acc |= (static_cast<unsigned __int128>(w) << acc_bits);
        acc_bits += 64;
      }
      out[out_off + i] = static_cast<uint64_t>(acc) & mask;
      acc >>= eff_bits;
      acc_bits -= eff_bits;
    }
  }

  if (tail) {
    const size_t out_off = full_blocks * block_out;
    const size_t in_off = full_blocks * block_in;
    unsigned __int128 acc = 0;
    int acc_bits = 0;
    size_t in_w = 0;
    for (size_t i = 0; i < tail; ++i) {
      while (acc_bits < eff_bits) {
        uint64_t w = (in_off + in_w < packed_words) ? packed[in_off + in_w] : 0ull;
        ++in_w;
        acc |= (static_cast<unsigned __int128>(w) << acc_bits);
        acc_bits += 64;
      }
      out[out_off + i] = static_cast<uint64_t>(acc) & mask;
      acc >>= eff_bits;
      acc_bits -= eff_bits;
    }
  }
}

}  // namespace proto
