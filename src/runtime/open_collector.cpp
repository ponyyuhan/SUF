#include "runtime/open_collector.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "proto/common.hpp"
#include "runtime/bench_online_profile.hpp"
#ifdef SUF_HAVE_CUDA
#include <cuda_runtime.h>
#include "runtime/cuda_primitives.hpp"
#endif

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

size_t gcd_size_t(size_t a, size_t b) {
  while (b != 0) {
    size_t t = a % b;
    a = b;
    b = t;
  }
  return a;
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
  const size_t n = xs.size();
  if (n == 0) return;
  const size_t g = gcd_size_t(static_cast<size_t>(eff_bits), 64);
  const size_t block_in = 64 / g;
  const size_t block_out = (block_in * static_cast<size_t>(eff_bits)) / 64;
  const size_t full_blocks = n / block_in;
  const size_t tail = n - full_blocks * block_in;

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
  if (elems == 0) return;
  const size_t g = gcd_size_t(static_cast<size_t>(eff_bits), 64);
  const size_t block_in = 64 / g;
  const size_t block_out = (block_in * static_cast<size_t>(eff_bits)) / 64;
  const size_t full_blocks = elems / block_in;
  const size_t tail = elems - full_blocks * block_in;

#ifdef _OPENMP
#pragma omp parallel for if (full_blocks >= 8) schedule(static)
#endif
  for (size_t b = 0; b < full_blocks; ++b) {
    const size_t out_off = b * block_in;
    const size_t in_off = b * block_out;
    unsigned __int128 acc = 0;
    int acc_bits = 0;
    size_t in_w = 0;
    for (size_t i = 0; i < block_in; ++i) {
      while (acc_bits < eff_bits) {
        uint64_t w = packed[in_off + in_w++];
        acc |= (static_cast<unsigned __int128>(w) << acc_bits);
        acc_bits += 64;
      }
      out[out_off + i] = static_cast<uint64_t>(acc) & mask;
      acc >>= eff_bits;
      acc_bits -= eff_bits;
    }
  }
  if (tail) {
    const size_t out_off = full_blocks * block_in;
    const size_t in_off = full_blocks * block_out;
    unsigned __int128 acc = 0;
    int acc_bits = 0;
    size_t in_w = 0;
    for (size_t i = 0; i < tail; ++i) {
      while (acc_bits < eff_bits) {
        uint64_t w = (in_off + in_w < packed.size()) ? packed[in_off + in_w] : 0ull;
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

}  // namespace

OpenCollector::~OpenCollector() {
#ifdef SUF_HAVE_CUDA
  if (pack_scratch_.d_in) cudaFree(pack_scratch_.d_in);
  if (pack_scratch_.d_packed) cudaFree(pack_scratch_.d_packed);
  if (pack_scratch_.d_out) cudaFree(pack_scratch_.d_out);
  pack_scratch_ = DevicePackScratch{};
#endif
}

OpenHandle OpenCollector::enqueue(const std::vector<uint64_t>& diff, OpenKind kind) {
  auto res = reserve(diff.size(), kind);
  if (!diff.empty()) {
    std::memcpy(res.diff.data(), diff.data(), diff.size() * sizeof(uint64_t));
  }
  return res.handle;
}

OpenCollector::Reserve OpenCollector::reserve(size_t len, OpenKind kind) {
  const size_t new_pending = pending_words_ + len;
  if (limits_.max_pending_words > 0 && new_pending > limits_.max_pending_words) {
    throw std::runtime_error("OpenCollector: pending open budget exceeded");
  }
  OpenHandle h;
  h.gen = generation_;
  h.offset = pending_words_;
  h.len = len;
  Request r;
  r.offset = h.offset;
  r.len = h.len;
  r.kind = (static_cast<size_t>(kind) < static_cast<size_t>(OpenKind::kCount)) ? kind
                                                                               : OpenKind::kOther;
  requests_.push_back(r);
  pending_words_ = new_pending;
  if (pending_words_ > stats_.max_pending_words) {
    stats_.max_pending_words = pending_words_;
  }
  pending_flat_buf_.resize(pending_words_);
  std::span<uint64_t> view(pending_flat_buf_.data() + h.offset, h.len);
  return Reserve{h, view};
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
  const bool device_pack = env_flag_enabled("SUF_OPEN_PACK_DEVICE");
  const bool device_scatter = env_flag_enabled_default("SUF_OPEN_PACK_DEVICE_SCATTER", false);
  const size_t device_min_words = []() -> size_t {
    const char* env = std::getenv("SUF_OPEN_PACK_DEVICE_MIN_WORDS");
    if (!env) return 1ull << 18;
    long long v = std::atoll(env);
    if (v <= 0) return 1ull << 18;
    return static_cast<size_t>(v);
  }();
  size_t total_words = pending_words_;
  std::array<size_t, static_cast<size_t>(OpenKind::kCount)> words_by_kind{};
  for (const auto& req : requests_) {
    if (req.offset + req.len > total_words) {
      throw std::runtime_error("OpenCollector: request out of range");
    }
    const size_t k = static_cast<size_t>(req.kind);
    if (k < words_by_kind.size()) words_by_kind[k] += req.len;
  }
  if (pending_flat_buf_.size() != total_words) {
    throw std::runtime_error("OpenCollector: pending buffer size mismatch");
  }
  const bool dynamic_pack = env_flag_enabled_default("SUF_OPEN_PACK_DYNAMIC", false);
  const int ring_bits = proto::ring_bits();
  const int n_bits = ring_bits;
  const bool calc_max_bits = dynamic_pack && want_pack && ring_bits > 0 && ring_bits <= 64;
  int max_bits = 1;
  // Materialize input (and optionally compute effective bits) once per flush.
  const auto t_pack0 = std::chrono::steady_clock::now();
  auto& send_flat = send_flat_buf_;
  // Swap so we can use the contiguous pending buffer as the send buffer without copying.
  send_flat.swap(pending_flat_buf_);
  pending_flat_buf_.clear();
  if (calc_max_bits) {
    int max_bits_local = 1;
    if (signed_pack) {
#ifdef _OPENMP
#pragma omp parallel for if (total_words >= (1ull << 16)) reduction(max : max_bits_local) schedule(static)
#endif
      for (size_t i = 0; i < total_words; ++i) {
        uint64_t v = send_flat[i];
        int bits = bits_needed_twos_complement_fast(v, ring_bits);
        if (bits > max_bits_local) max_bits_local = bits;
      }
    } else {
#ifdef _OPENMP
#pragma omp parallel for if (total_words >= (1ull << 16)) reduction(max : max_bits_local) schedule(static)
#endif
      for (size_t i = 0; i < total_words; ++i) {
        uint64_t v = send_flat[i];
        int bits = bits_needed_u64_fast(v);
        if (bits > max_bits_local) max_bits_local = bits;
      }
    }
    max_bits = max_bits_local;
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
    // Fast-path: one bulk exchange per flush (no packing).
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
    opened_flat_buf_.resize(total_words);
#ifdef _OPENMP
#pragma omp parallel for if (total_words >= (1ull << 16)) schedule(static)
#endif
    for (size_t i = 0; i < total_words; ++i) {
      uint64_t opened = proto::add_mod(send_flat[i], recv_ptr[i]);
      opened_flat_buf_[i] = proto::to_signed(opened);
    }
    const auto t_scatter1 = std::chrono::steady_clock::now();
    requests_.clear();
    stats_.flushes += 1;
    stats_.opened_words += total_words;
    for (size_t k = 0; k < words_by_kind.size(); ++k) {
      stats_.opened_words_by_kind[k] += words_by_kind[k];
    }
    pending_words_ = 0;
    // Keep send buffer capacity for next flush but reset it so it doesn't retain old values.
    send_flat.clear();
    opened_ready_ = true;
    opened_generation_ = generation_;
    generation_ += 1;
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
  bool opened_from_device = false;
  if (eff > 0 && eff < 64) {
    auto& packed_local = packed_local_buf_;
    auto& packed_remote = packed_remote_buf_;
    const size_t packed_words = packed_words_host(total_words, eff);
    packed_local.resize(packed_words);
    packed_remote.resize(packed_words);
    bool used_device_pack = false;
#ifdef SUF_HAVE_CUDA
    const bool want_device_pack = device_pack && total_words >= device_min_words;
    if (want_device_pack) {
      auto check_cuda = [](cudaError_t st, const char* what) {
        if (st != cudaSuccess) {
          throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(st));
        }
      };
      auto ensure_buffers = [&](size_t words, size_t packed_words_local) {
        const size_t in_bytes = words * sizeof(uint64_t);
        const size_t packed_bytes = packed_words_local * sizeof(uint64_t);
        if (pack_scratch_.in_cap < in_bytes) {
          if (pack_scratch_.d_in) cudaFree(pack_scratch_.d_in);
          check_cuda(cudaMalloc(&pack_scratch_.d_in, in_bytes), "cudaMalloc open_pack d_in");
          pack_scratch_.in_cap = in_bytes;
        }
        if (pack_scratch_.packed_cap < packed_bytes) {
          if (pack_scratch_.d_packed) cudaFree(pack_scratch_.d_packed);
          check_cuda(cudaMalloc(&pack_scratch_.d_packed, packed_bytes), "cudaMalloc open_pack d_packed");
          pack_scratch_.packed_cap = packed_bytes;
        }
        if (pack_scratch_.out_cap < in_bytes) {
          if (pack_scratch_.d_out) cudaFree(pack_scratch_.d_out);
          check_cuda(cudaMalloc(&pack_scratch_.d_out, in_bytes), "cudaMalloc open_pack d_out");
          pack_scratch_.out_cap = in_bytes;
        } else if (!pack_scratch_.d_out) {
          check_cuda(cudaMalloc(&pack_scratch_.d_out, in_bytes), "cudaMalloc open_pack d_out");
          pack_scratch_.out_cap = in_bytes;
        }
      };
      try {
        const auto t_pack2_0 = std::chrono::steady_clock::now();
        ensure_buffers(total_words, packed_words);
        check_cuda(cudaMemcpy(pack_scratch_.d_in, send_flat.data(),
                              total_words * sizeof(uint64_t),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy H2D open_pack");
        launch_pack_eff_bits_kernel(pack_scratch_.d_in, eff, pack_scratch_.d_packed,
                                    total_words, nullptr);
        check_cuda(cudaMemcpy(packed_local.data(), pack_scratch_.d_packed,
                              packed_words * sizeof(uint64_t),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H open_pack");
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
        check_cuda(cudaMemcpy(pack_scratch_.d_packed, packed_remote.data(),
                              packed_words * sizeof(uint64_t),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy H2D open_unpack");
        launch_unpack_eff_bits_kernel(pack_scratch_.d_packed, eff, pack_scratch_.d_out,
                                      total_words, nullptr);
        if (device_scatter) {
          // Optional: compute opened values on device (including sign-extension to
          // the current ring) to avoid the host-side scatter loop. This is usually
          // only a win when each party has its own GPU; on single-GPU benchmarks,
          // it can increase contention and inflate comm time.
          launch_open_add_to_signed_kernel(pack_scratch_.d_in,
                                           pack_scratch_.d_out,
                                           pack_scratch_.d_out,
                                           total_words,
                                           n_bits,
                                           proto::ring_mask(),
                                           nullptr);
          opened_flat_buf_.resize(total_words);
          check_cuda(cudaMemcpy(opened_flat_buf_.data(), pack_scratch_.d_out,
                                total_words * sizeof(uint64_t),
                                cudaMemcpyDeviceToHost),
                     "cudaMemcpy D2H open_opened");
          opened_from_device = true;
        } else {
          check_cuda(cudaMemcpy(other_flat.data(), pack_scratch_.d_out,
                                total_words * sizeof(uint64_t),
                                cudaMemcpyDeviceToHost),
                     "cudaMemcpy D2H open_unpack");
          if (signed_pack && n_bits > 0 && n_bits <= 64) {
            for (auto& w : other_flat) w = sign_extend_to_nbits(w, eff, n_bits);
          }
        }
        const auto t_pack2_3 = std::chrono::steady_clock::now();
        pack2_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                             (t_pack2_1 - t_pack2_0) + (t_pack2_3 - t_pack2_2))
                                             .count());
        add_ns(runtime::bench::OnlineTimeKind::OpenComm, t_comm4, t_comm5);
        used_device_pack = true;
      } catch (const std::exception&) {
        used_device_pack = false;
      }
    }
#endif
    if (!used_device_pack) {
      const auto t_pack2_0 = std::chrono::steady_clock::now();
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
    }
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
  if (!opened_from_device) {
    const uint64_t* other_ptr = other_flat.data();
    opened_flat_buf_.resize(total_words);
#ifdef _OPENMP
#pragma omp parallel for if (total_words >= (1ull << 16)) schedule(static)
#endif
    for (size_t i = 0; i < total_words; ++i) {
      uint64_t opened = proto::add_mod(send_flat[i], other_ptr[i]);
      opened_flat_buf_[i] = proto::to_signed(opened);
    }
  }
  const auto t_scatter1 = std::chrono::steady_clock::now();
  requests_.clear();
  stats_.flushes += 1;
  stats_.opened_words += total_words;
  for (size_t k = 0; k < words_by_kind.size(); ++k) {
    stats_.opened_words_by_kind[k] += words_by_kind[k];
  }
  pending_words_ = 0;
  send_flat.clear();
  opened_ready_ = true;
  opened_generation_ = generation_;
  generation_ += 1;
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
  if (!opened_ready_ || h.gen != opened_generation_) {
    throw std::runtime_error("OpenCollector: view out of range");
  }
  if (h.offset + h.len > opened_flat_buf_.size()) {
    throw std::runtime_error("OpenCollector: view out of range");
  }
  return std::span<const int64_t>(opened_flat_buf_.data() + h.offset, h.len);
}

bool OpenCollector::ready(const OpenHandle& h) const {
  return opened_ready_ &&
         h.gen == opened_generation_ &&
         h.offset + h.len <= opened_flat_buf_.size();
}

void OpenCollector::clear() {
  requests_.clear();
  pending_words_ = 0;
  pending_flat_buf_.clear();
  send_flat_buf_.clear();
  recv_flat_buf_.clear();
  other_flat_buf_.clear();
  packed_local_buf_.clear();
  packed_remote_buf_.clear();
  opened_flat_buf_.clear();
  opened_ready_ = false;
  opened_generation_ = static_cast<uint64_t>(-1);
  generation_ += 1;
}

}  // namespace runtime
