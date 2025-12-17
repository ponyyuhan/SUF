#include "runtime/open_collector.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <stdexcept>

#include "proto/common.hpp"

namespace runtime {

namespace {

int bits_needed_u64(uint64_t v) {
  int bits = 0;
  while (v) {
    v >>= 1;
    bits++;
  }
  return std::max(bits, 1);
}

uint64_t mask_low_bits_u64(int bits) {
  if (bits <= 0) return 0ull;
  if (bits >= 64) return ~uint64_t(0);
  return (uint64_t(1) << bits) - 1;
}

int bits_needed_twos_complement(uint64_t v, int n_bits) {
  if (n_bits <= 0 || n_bits > 64) return 64;
  const uint64_t nmask = mask_low_bits_u64(n_bits);
  v &= nmask;
  if (v == 0) return 1;
  if (n_bits < 64 && v == nmask) return 1;  // -1 in n-bit two's complement
  const bool neg = (n_bits == 64) ? (static_cast<int64_t>(v) < 0)
                                  : ((v & (uint64_t(1) << (n_bits - 1))) != 0);
  if (!neg) {
    return std::min(64, bits_needed_u64(v) + 1);
  }
  uint64_t inv = (~v) & nmask;
  if (inv == 0) return 1;
  return std::min(64, bits_needed_u64(inv) + 1);
}

uint64_t sign_extend_to_nbits(uint64_t v, int from_bits, int n_bits) {
  if (n_bits <= 0 || n_bits > 64) return v;
  if (from_bits <= 0) return 0ull;
  if (from_bits > 64) from_bits = 64;
  const uint64_t from_mask = mask_low_bits_u64(from_bits);
  v &= from_mask;
  if (from_bits < 64) {
    const uint64_t sign_bit = uint64_t(1) << (from_bits - 1);
    if (v & sign_bit) v |= ~from_mask;
  }
  if (n_bits < 64) v &= mask_low_bits_u64(n_bits);
  return v;
}

size_t packed_words_host(size_t elems, int eff_bits) {
  if (eff_bits <= 0 || eff_bits > 64) return elems;
  if (eff_bits == 64) return elems;
  unsigned __int128 total_bits = static_cast<unsigned __int128>(elems) *
                                 static_cast<unsigned __int128>(eff_bits);
  total_bits += 63;
  return static_cast<size_t>(total_bits / 64);
}

uint64_t mask_bits_host(int eff_bits) {
  if (eff_bits <= 0) return 0ull;
  if (eff_bits >= 64) return ~uint64_t(0);
  return (uint64_t(1) << eff_bits) - 1;
}

std::vector<uint64_t> pack_eff_bits_host(const std::vector<uint64_t>& xs, int eff_bits) {
  if (eff_bits <= 0 || eff_bits > 64) {
    throw std::runtime_error("OpenCollector: eff_bits out of range for pack");
  }
  if (eff_bits == 64) return xs;
  size_t words = packed_words_host(xs.size(), eff_bits);
  std::vector<uint64_t> packed(words, 0);
  uint64_t mask = mask_bits_host(eff_bits);
  for (size_t i = 0; i < xs.size(); ++i) {
    uint64_t v = xs[i] & mask;
    size_t bit_idx = i * static_cast<size_t>(eff_bits);
    size_t w = bit_idx >> 6;
    int off = static_cast<int>(bit_idx & 63);
    packed[w] |= (v << off);
    int spill = off + eff_bits - 64;
    if (spill > 0 && w + 1 < packed.size()) {
      packed[w + 1] |= (v >> (eff_bits - spill));
    }
  }
  return packed;
}

std::vector<uint64_t> unpack_eff_bits_host(const std::vector<uint64_t>& packed,
                                           int eff_bits,
                                           size_t elems) {
  if (eff_bits <= 0 || eff_bits > 64) {
    throw std::runtime_error("OpenCollector: eff_bits out of range for unpack");
  }
  if (eff_bits == 64) return packed;
  std::vector<uint64_t> out(elems, 0);
  uint64_t mask = mask_bits_host(eff_bits);
  for (size_t i = 0; i < elems; ++i) {
    size_t bit_idx = i * static_cast<size_t>(eff_bits);
    size_t w = bit_idx >> 6;
    int off = static_cast<int>(bit_idx & 63);
    uint64_t v = (w < packed.size()) ? (packed[w] >> off) : 0ull;
    int spill = off + eff_bits - 64;
    if (spill > 0 && w + 1 < packed.size()) {
      v |= (packed[w + 1] << (eff_bits - spill));
    }
    out[i] = v & mask;
  }
  return out;
}

}  // namespace

OpenHandle OpenCollector::enqueue(const std::vector<uint64_t>& diff) {
  size_t new_pending = pending_words_ + diff.size();
  if (limits_.max_pending_words > 0 && new_pending > limits_.max_pending_words) {
    throw std::runtime_error("OpenCollector: pending open budget exceeded");
  }
  auto slot = std::make_shared<OpenSlot>();
  slot->n = diff.size();
  slot->opened.resize(diff.size(), 0);
  OpenHandle h;
  h.slot = slot;
  h.offset = 0;
  h.len = diff.size();
  Request r;
  r.diff = diff;
  r.slot = slot;
  r.offset = h.offset;
  r.len = h.len;
  requests_.push_back(std::move(r));
  pending_words_ = new_pending;
  if (pending_words_ > stats_.max_pending_words) {
    stats_.max_pending_words = pending_words_;
  }
  return h;
}

void OpenCollector::flush(int party, net::Chan& ch) {
  if (requests_.empty()) return;
  auto t0 = std::chrono::steady_clock::now();
  const bool want_pack = (std::getenv("SUF_OPEN_PACK_EFFBITS") != nullptr);
  const bool signed_pack = (std::getenv("SUF_OPEN_PACK_SIGNED") != nullptr);
  const int max_pack_bits = []() {
    const char* env = std::getenv("SUF_OPEN_PACK_MAX_BITS");
    if (!env) return 48;
    int v = std::atoi(env);
    if (v <= 0 || v > 64) return 48;
    return v;
  }();
  const bool pack = [&]() {
    uint64_t remote = 0;
    const uint64_t local = want_pack ? 1ull : 0ull;
    if (party == 0) {
      ch.send_u64(local);
      remote = ch.recv_u64();
    } else {
      remote = ch.recv_u64();
      ch.send_u64(local);
    }
    return want_pack && (remote != 0);
  }();

  size_t total_words = 0;
  if (!pack) {
    // Fast-path: one bulk exchange per flush, then scatter results.
    for (const auto& req : requests_) {
      if (!req.slot) {
        throw std::runtime_error("OpenCollector: missing result slot");
      }
      if (req.offset + req.len > req.slot->opened.size()) {
        throw std::runtime_error("OpenCollector: request out of range");
      }
      total_words += req.len;
    }
    std::vector<uint64_t> send_flat;
    send_flat.reserve(total_words);
    for (const auto& req : requests_) {
      send_flat.insert(send_flat.end(), req.diff.begin(), req.diff.end());
    }
    std::vector<uint64_t> recv_flat(total_words, 0);
    if (party == 0) {
      if (!send_flat.empty()) ch.send_u64s(send_flat.data(), send_flat.size());
      if (!recv_flat.empty()) ch.recv_u64s(recv_flat.data(), recv_flat.size());
    } else {
      if (!recv_flat.empty()) ch.recv_u64s(recv_flat.data(), recv_flat.size());
      if (!send_flat.empty()) ch.send_u64s(send_flat.data(), send_flat.size());
    }
    size_t off = 0;
    for (const auto& req : requests_) {
      for (size_t i = 0; i < req.len; ++i) {
        uint64_t opened = proto::add_mod(req.diff[i], recv_flat[off + i]);
        req.slot->opened[req.offset + i] = proto::to_signed(opened);
      }
      req.slot->ready.store(true);
      off += req.len;
    }
    requests_.clear();
    stats_.flushes += 1;
    stats_.opened_words += total_words;
    pending_words_ = 0;
    auto t1 = std::chrono::steady_clock::now();
    stats_.flush_ns += static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    return;
  }

  for (const auto& req : requests_) {
    if (!req.slot) {
      throw std::runtime_error("OpenCollector: missing result slot");
    }
    if (req.offset + req.len > req.slot->opened.size()) {
      throw std::runtime_error("OpenCollector: request out of range");
    }
    if (pack) {
      int eff_need = 64;
      if (!signed_pack) {
        uint64_t or_acc = 0;
        for (auto v : req.diff) or_acc |= v;
        eff_need = bits_needed_u64(or_acc);
      } else {
        const int n_bits = proto::ring_bits();
        int need = 1;
        for (auto v : req.diff) {
          need = std::max(need, bits_needed_twos_complement(v, n_bits));
          if (need >= 64) break;
        }
        eff_need = need;
      }
      int eff_local =
          (eff_need > 0 && eff_need < 64 && eff_need <= max_pack_bits) ? eff_need : 64;
      int eff_remote = 0;
      if (party == 0) {
        ch.send_u64(static_cast<uint64_t>(eff_local));
        eff_remote = static_cast<int>(ch.recv_u64());
      } else {
        eff_remote = static_cast<int>(ch.recv_u64());
        ch.send_u64(static_cast<uint64_t>(eff_local));
      }
      int eff = std::max(eff_local, eff_remote);
      if (eff > 0 && eff < 64) {
        std::vector<uint64_t> packed_local = pack_eff_bits_host(req.diff, eff);
        std::vector<uint64_t> packed_remote(packed_local.size(), 0);
        if (party == 0) {
          if (!packed_local.empty()) ch.send_u64s(packed_local.data(), packed_local.size());
          if (!packed_remote.empty()) ch.recv_u64s(packed_remote.data(), packed_remote.size());
        } else {
          if (!packed_remote.empty()) ch.recv_u64s(packed_remote.data(), packed_remote.size());
          if (!packed_local.empty()) ch.send_u64s(packed_local.data(), packed_local.size());
        }
        auto other = unpack_eff_bits_host(packed_remote, eff, req.len);
        if (signed_pack) {
          const int n_bits = proto::ring_bits();
          for (auto& w : other) w = sign_extend_to_nbits(w, eff, n_bits);
        }
        for (size_t i = 0; i < req.len; ++i) {
          uint64_t opened = proto::add_mod(req.diff[i], other[i]);
          req.slot->opened[req.offset + i] = proto::to_signed(opened);
        }
      } else {
        std::vector<uint64_t> other(req.len, 0);
        if (party == 0) {
          if (!req.diff.empty()) ch.send_u64s(req.diff.data(), req.diff.size());
          if (!other.empty()) ch.recv_u64s(other.data(), other.size());
        } else {
          if (!other.empty()) ch.recv_u64s(other.data(), other.size());
          if (!req.diff.empty()) ch.send_u64s(req.diff.data(), req.diff.size());
        }
        for (size_t i = 0; i < req.len; ++i) {
          uint64_t opened = proto::add_mod(req.diff[i], other[i]);
          req.slot->opened[req.offset + i] = proto::to_signed(opened);
        }
      }
    } else {
      std::vector<uint64_t> other(req.len, 0);
      if (party == 0) {
        if (!req.diff.empty()) ch.send_u64s(req.diff.data(), req.diff.size());
        if (!other.empty()) ch.recv_u64s(other.data(), other.size());
      } else {
        if (!other.empty()) ch.recv_u64s(other.data(), other.size());
        if (!req.diff.empty()) ch.send_u64s(req.diff.data(), req.diff.size());
      }
      for (size_t i = 0; i < req.len; ++i) {
        uint64_t opened = proto::add_mod(req.diff[i], other[i]);
        req.slot->opened[req.offset + i] = proto::to_signed(opened);
      }
    }
    req.slot->ready.store(true);
    total_words += req.len;
  }
  requests_.clear();
  stats_.flushes += 1;
  stats_.opened_words += total_words;
  pending_words_ = 0;
  auto t1 = std::chrono::steady_clock::now();
  stats_.flush_ns += static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
}

std::span<const int64_t> OpenCollector::view(const OpenHandle& h) const {
  if (!h.slot || !h.slot->ready.load()) {
    throw std::runtime_error("OpenCollector: view out of range");
  }
  if (h.offset + h.len > h.slot->opened.size()) {
    throw std::runtime_error("OpenCollector: view out of range");
  }
  return std::span<const int64_t>(h.slot->opened.data() + h.offset, h.len);
}

bool OpenCollector::ready(const OpenHandle& h) const {
  return h.slot && h.slot->ready.load() && h.offset + h.len <= h.slot->opened.size();
}

void OpenCollector::clear() {
  requests_.clear();
  pending_words_ = 0;
}

}  // namespace runtime
