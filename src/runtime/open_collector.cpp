#include "runtime/open_collector.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "proto/common.hpp"
#include "runtime/bench_online_profile.hpp"

namespace runtime {

namespace {

bool env_flag_enabled(const char* name) {
  const char* env = std::getenv(name);
  if (!env) return false;
  std::string v(env);
  std::transform(v.begin(), v.end(), v.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return !(v == "0" || v == "false" || v == "off" || v == "no");
}

bool env_flag_enabled_default(const char* name, bool defv) {
  const char* env = std::getenv(name);
  if (!env) return defv;
  std::string v(env);
  std::transform(v.begin(), v.end(), v.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return !(v == "0" || v == "false" || v == "off" || v == "no");
}

int bits_needed_u64_fast(uint64_t v) {
  if (v == 0) return 1;
  return 64 - __builtin_clzll(v);
}

uint64_t mask_low_bits_u64(int bits) {
  if (bits <= 0) return 0ull;
  if (bits >= 64) return ~uint64_t(0);
  return (uint64_t(1) << bits) - 1;
}

int bits_needed_twos_complement_fast(uint64_t v, int n_bits) {
  if (n_bits <= 0 || n_bits > 64) return 64;
  const uint64_t nmask = mask_low_bits_u64(n_bits);
  v &= nmask;
  if (v == 0) return 1;
  if (n_bits < 64 && v == nmask) return 1;
  const bool neg = (n_bits == 64) ? (static_cast<int64_t>(v) < 0)
                                  : ((v & (uint64_t(1) << (n_bits - 1))) != 0);
  if (!neg) {
    return std::min(64, bits_needed_u64_fast(v) + 1);
  }
  uint64_t inv = (~v) & nmask;
  if (inv == 0) return 1;
  return std::min(64, bits_needed_u64_fast(inv) + 1);
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

void pack_eff_bits_host_into(const std::vector<uint64_t>& xs,
                             int eff_bits,
                             std::vector<uint64_t>& out) {
  if (eff_bits <= 0 || eff_bits > 64) {
    throw std::runtime_error("OpenCollector: eff_bits out of range for pack");
  }
  if (eff_bits == 64) {
    out.assign(xs.begin(), xs.end());
    return;
  }
  // Streaming bit pack using a 128-bit accumulator; avoids per-element div/mod.
  const size_t words = packed_words_host(xs.size(), eff_bits);
  out.assign(words, 0);
  const uint64_t mask = mask_bits_host(eff_bits);
  unsigned __int128 acc = 0;
  int acc_bits = 0;
  size_t out_w = 0;
  for (size_t i = 0; i < xs.size(); ++i) {
    uint64_t v = xs[i] & mask;
    acc |= (static_cast<unsigned __int128>(v) << acc_bits);
    acc_bits += eff_bits;
    while (acc_bits >= 64) {
      if (out_w >= out.size()) break;
      out[out_w++] = static_cast<uint64_t>(acc);
      acc >>= 64;
      acc_bits -= 64;
    }
  }
  if (acc_bits > 0 && out_w < out.size()) {
    out[out_w++] = static_cast<uint64_t>(acc);
  }
}

void unpack_eff_bits_host_into(const std::vector<uint64_t>& packed,
                               int eff_bits,
                               size_t elems,
                               std::vector<uint64_t>& out) {
  if (eff_bits <= 0 || eff_bits > 64) {
    throw std::runtime_error("OpenCollector: eff_bits out of range for unpack");
  }
  if (eff_bits == 64) {
    out.assign(packed.begin(), packed.end());
    return;
  }
  out.assign(elems, 0);
  const uint64_t mask = mask_bits_host(eff_bits);
  unsigned __int128 acc = 0;
  int acc_bits = 0;
  size_t in_w = 0;
  for (size_t i = 0; i < elems; ++i) {
    while (acc_bits < eff_bits) {
      uint64_t w = (in_w < packed.size()) ? packed[in_w] : 0ull;
      ++in_w;
      acc |= (static_cast<unsigned __int128>(w) << acc_bits);
      acc_bits += 64;
    }
    out[i] = static_cast<uint64_t>(acc) & mask;
    acc >>= eff_bits;
    acc_bits -= eff_bits;
  }
}

}  // namespace

OpenHandle OpenCollector::enqueue(const std::vector<uint64_t>& diff, OpenKind kind) {
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
  r.kind = (static_cast<size_t>(kind) < static_cast<size_t>(OpenKind::kCount)) ? kind
                                                                               : OpenKind::kOther;
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
  auto add_ns = [](runtime::bench::OnlineTimeKind k,
                   const std::chrono::steady_clock::time_point& a,
                   const std::chrono::steady_clock::time_point& b) {
    if (b <= a) return;
    runtime::bench::add_online_ns(
        k, static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count()));
  };
  const bool want_pack = env_flag_enabled("SUF_OPEN_PACK_EFFBITS");
  const bool signed_pack = env_flag_enabled("SUF_OPEN_PACK_SIGNED");
  const bool auto_pack = env_flag_enabled("SUF_OPEN_PACK_AUTO");
  const double min_savings = []() -> double {
    const char* env = std::getenv("SUF_OPEN_PACK_MIN_SAVINGS_PCT");
    if (!env) return 0.25;
    double v = std::atof(env) / 100.0;
    if (v < 0.0) v = 0.0;
    if (v > 1.0) v = 1.0;
    return v;
  }();
  const int max_pack_bits = []() {
    const char* env = std::getenv("SUF_OPEN_PACK_MAX_BITS");
    // Default to 56 so 50–51 bit rings (SIGMA-style fixed-point) are packable.
    if (!env) return 56;
    int v = std::atoi(env);
    if (v <= 0 || v > 64) return 56;
    return v;
  }();
  size_t total_words = 0;
  std::vector<size_t> req_offsets;
  req_offsets.reserve(requests_.size());
  std::array<size_t, static_cast<size_t>(OpenKind::kCount)> words_by_kind{};
  for (const auto& req : requests_) {
    if (!req.slot) {
      throw std::runtime_error("OpenCollector: missing result slot");
    }
    if (req.offset + req.len > req.slot->opened.size()) {
      throw std::runtime_error("OpenCollector: request out of range");
    }
    req_offsets.push_back(total_words);
    total_words += req.len;
    const size_t k = static_cast<size_t>(req.kind);
    if (k < words_by_kind.size()) words_by_kind[k] += req.len;
  }
  const bool dynamic_pack = env_flag_enabled_default("SUF_OPEN_PACK_DYNAMIC", false);
  const int ring_bits = proto::ring_bits();
  const int n_bits = ring_bits;
  const bool calc_max_bits = dynamic_pack && want_pack && ring_bits > 0 && ring_bits <= 64;
  int max_bits = 1;
  // Pack input (and optionally compute effective bits) once per flush.
  const auto t_pack0 = std::chrono::steady_clock::now();
  auto& send_flat = send_flat_buf_;
  send_flat.resize(total_words);
  const size_t nreq = requests_.size();
  if (calc_max_bits) {
    int max_bits_local = 1;
    if (signed_pack) {
#ifdef _OPENMP
#pragma omp parallel for if (nreq >= 8) reduction(max : max_bits_local)
#endif
      for (size_t idx = 0; idx < nreq; ++idx) {
        const auto& req = requests_[idx];
        const size_t len = req.len;
        const size_t off = req_offsets[idx];
        if (len) {
          std::memcpy(send_flat.data() + off, req.diff.data(), len * sizeof(uint64_t));
        }
        for (size_t i = 0; i < len; ++i) {
          uint64_t v = req.diff[i];
          int bits = bits_needed_twos_complement_fast(v, ring_bits);
          if (bits > max_bits_local) max_bits_local = bits;
        }
      }
    } else {
#ifdef _OPENMP
#pragma omp parallel for if (nreq >= 8) reduction(max : max_bits_local)
#endif
      for (size_t idx = 0; idx < nreq; ++idx) {
        const auto& req = requests_[idx];
        const size_t len = req.len;
        const size_t off = req_offsets[idx];
        if (len) {
          std::memcpy(send_flat.data() + off, req.diff.data(), len * sizeof(uint64_t));
        }
        for (size_t i = 0; i < len; ++i) {
          uint64_t v = req.diff[i];
          int bits = bits_needed_u64_fast(v);
          if (bits > max_bits_local) max_bits_local = bits;
        }
      }
    }
    max_bits = max_bits_local;
  } else {
#ifdef _OPENMP
#pragma omp parallel for if (nreq >= 8)
#endif
    for (size_t idx = 0; idx < nreq; ++idx) {
      const auto& req = requests_[idx];
      const size_t len = req.len;
      const size_t off = req_offsets[idx];
      if (len) {
        std::memcpy(send_flat.data() + off, req.diff.data(), len * sizeof(uint64_t));
      }
    }
  }
  const auto t_pack1 = std::chrono::steady_clock::now();

  const auto t_comm0 = std::chrono::steady_clock::now();
  const bool pack = [&]() {
    uint64_t remote = 0;
    uint64_t local = want_pack ? 1ull : 0ull;
    if (local && auto_pack) {
      int eff = 64;
      if (ring_bits > 0 && ring_bits < 64 && ring_bits <= max_pack_bits) eff = ring_bits;
      if (dynamic_pack && max_bits > 0 && max_bits < eff) eff = max_bits;
      const double total = static_cast<double>(total_words);
      if (total > 0.0) {
        const double packed = static_cast<double>(packed_words_host(total_words, eff));
        const double savings = (total - packed) / total;
        if (savings < min_savings) local = 0ull;
      }
    }
    if (party == 0) {
      ch.send_u64(local);
      remote = ch.recv_u64();
    } else {
      remote = ch.recv_u64();
      ch.send_u64(local);
    }
    return (local != 0) && (remote != 0);
  }();
  const auto t_comm1 = std::chrono::steady_clock::now();
  if (!pack) {
    // Fast-path: one bulk exchange per flush, then scatter results.
    auto& recv_flat = recv_flat_buf_;
    recv_flat.resize(total_words);
    const auto t_comm2 = std::chrono::steady_clock::now();
    if (party == 0) {
      if (!send_flat.empty()) ch.send_u64s(send_flat.data(), send_flat.size());
      if (!recv_flat.empty()) ch.recv_u64s(recv_flat.data(), recv_flat.size());
    } else {
      if (!recv_flat.empty()) ch.recv_u64s(recv_flat.data(), recv_flat.size());
      if (!send_flat.empty()) ch.send_u64s(send_flat.data(), send_flat.size());
    }
    const auto t_comm3 = std::chrono::steady_clock::now();
    const auto t_scatter0 = std::chrono::steady_clock::now();
    const uint64_t* recv_ptr = recv_flat.data();
#ifdef _OPENMP
#pragma omp parallel for if (nreq >= 8)
#endif
    for (size_t idx = 0; idx < nreq; ++idx) {
      const auto& req = requests_[idx];
      size_t off = req_offsets[idx];
      for (size_t i = 0; i < req.len; ++i) {
        uint64_t opened = proto::add_mod(req.diff[i], recv_ptr[off + i]);
        req.slot->opened[req.offset + i] = proto::to_signed(opened);
      }
      req.slot->ready.store(true);
    }
    const auto t_scatter1 = std::chrono::steady_clock::now();
    requests_.clear();
    stats_.flushes += 1;
    stats_.opened_words += total_words;
    for (size_t k = 0; k < words_by_kind.size(); ++k) {
      stats_.opened_words_by_kind[k] += words_by_kind[k];
    }
    pending_words_ = 0;
    auto t1 = std::chrono::steady_clock::now();
    stats_.flush_ns += static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    add_ns(runtime::bench::OnlineTimeKind::OpenFlushTotal, t0, t1);
    add_ns(runtime::bench::OnlineTimeKind::OpenComm, t_comm0, t_comm1);
    add_ns(runtime::bench::OnlineTimeKind::OpenPack, t_pack0, t_pack1);
    add_ns(runtime::bench::OnlineTimeKind::OpenComm, t_comm2, t_comm3);
    add_ns(runtime::bench::OnlineTimeKind::OpenScatter, t_scatter0, t_scatter1);
    return;
  }

  // Packed path: pack the entire flush with a single bitwidth decision.
  // This mirrors Sigma’s “communication packing”: values live in Z_{2^n}, so we can safely pack to `n` bits.

  int eff_local = 64;
  if (ring_bits > 0 && ring_bits < 64 && ring_bits <= max_pack_bits) {
    eff_local = ring_bits;
  }
  if (dynamic_pack && max_bits > 0 && max_bits < eff_local) eff_local = max_bits;
  int eff_remote = 0;
  const auto t_comm2 = std::chrono::steady_clock::now();
  if (party == 0) {
    ch.send_u64(static_cast<uint64_t>(eff_local));
    eff_remote = static_cast<int>(ch.recv_u64());
  } else {
    eff_remote = static_cast<int>(ch.recv_u64());
    ch.send_u64(static_cast<uint64_t>(eff_local));
  }
  const auto t_comm3 = std::chrono::steady_clock::now();
  int eff = std::max(eff_local, eff_remote);
  if (eff <= 0 || eff > 64) eff = 64;

  auto& other_flat = other_flat_buf_;
  other_flat.resize(total_words);
  uint64_t pack2_ns = 0;
  if (eff > 0 && eff < 64) {
    const auto t_pack2_0 = std::chrono::steady_clock::now();
    auto& packed_local = packed_local_buf_;
    auto& packed_remote = packed_remote_buf_;
    pack_eff_bits_host_into(send_flat, eff, packed_local);
    packed_remote.resize(packed_local.size());
    const auto t_pack2_1 = std::chrono::steady_clock::now();
    const auto t_comm4 = std::chrono::steady_clock::now();
    if (party == 0) {
      if (!packed_local.empty()) ch.send_u64s(packed_local.data(), packed_local.size());
      if (!packed_remote.empty()) ch.recv_u64s(packed_remote.data(), packed_remote.size());
    } else {
      if (!packed_remote.empty()) ch.recv_u64s(packed_remote.data(), packed_remote.size());
      if (!packed_local.empty()) ch.send_u64s(packed_local.data(), packed_local.size());
    }
    const auto t_comm5 = std::chrono::steady_clock::now();
    const auto t_pack2_2 = std::chrono::steady_clock::now();
    unpack_eff_bits_host_into(packed_remote, eff, total_words, other_flat);
    if (signed_pack && n_bits > 0 && n_bits <= 64) {
      for (auto& w : other_flat) w = sign_extend_to_nbits(w, eff, n_bits);
    }
    const auto t_pack2_3 = std::chrono::steady_clock::now();
    pack2_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         (t_pack2_1 - t_pack2_0) + (t_pack2_3 - t_pack2_2))
                                         .count());
    add_ns(runtime::bench::OnlineTimeKind::OpenComm, t_comm4, t_comm5);
  } else {
    // Fallback: no packing.
    const auto t_comm4 = std::chrono::steady_clock::now();
    if (party == 0) {
      if (!send_flat.empty()) ch.send_u64s(send_flat.data(), send_flat.size());
      if (!other_flat.empty()) ch.recv_u64s(other_flat.data(), other_flat.size());
    } else {
      if (!other_flat.empty()) ch.recv_u64s(other_flat.data(), other_flat.size());
      if (!send_flat.empty()) ch.send_u64s(send_flat.data(), send_flat.size());
    }
    const auto t_comm5 = std::chrono::steady_clock::now();
    add_ns(runtime::bench::OnlineTimeKind::OpenComm, t_comm4, t_comm5);
  }

  const auto t_scatter0 = std::chrono::steady_clock::now();
  const uint64_t* other_ptr = other_flat.data();
#ifdef _OPENMP
#pragma omp parallel for if (nreq >= 8)
#endif
  for (size_t idx = 0; idx < nreq; ++idx) {
    const auto& req = requests_[idx];
    size_t off = req_offsets[idx];
    for (size_t i = 0; i < req.len; ++i) {
      uint64_t opened = proto::add_mod(req.diff[i], other_ptr[off + i]);
      req.slot->opened[req.offset + i] = proto::to_signed(opened);
    }
    req.slot->ready.store(true);
  }
  const auto t_scatter1 = std::chrono::steady_clock::now();
  requests_.clear();
  stats_.flushes += 1;
  stats_.opened_words += total_words;
  for (size_t k = 0; k < words_by_kind.size(); ++k) {
    stats_.opened_words_by_kind[k] += words_by_kind[k];
  }
  pending_words_ = 0;
  auto t1 = std::chrono::steady_clock::now();
  stats_.flush_ns += static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
  add_ns(runtime::bench::OnlineTimeKind::OpenFlushTotal, t0, t1);
  add_ns(runtime::bench::OnlineTimeKind::OpenComm, t_comm0, t_comm1);
  add_ns(runtime::bench::OnlineTimeKind::OpenPack, t_pack0, t_pack1);
  add_ns(runtime::bench::OnlineTimeKind::OpenComm, t_comm2, t_comm3);
  if (pack2_ns) runtime::bench::add_online_ns(runtime::bench::OnlineTimeKind::OpenPack, pack2_ns);
  add_ns(runtime::bench::OnlineTimeKind::OpenScatter, t_scatter0, t_scatter1);
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
