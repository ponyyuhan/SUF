// CUDA PFSS backend: host semantics identical to ClearBackend but with
// self-contained key blobs and simple GPU kernels for batched pred/coeff eval.
// This is a correctness-first implementation; kernels are straightforward
// compare/LUT walkers so we can exercise GPU plumbing while keeping the key
// format device-decodable. Replace the kernels with AES-CTR/DPF traversal for
// performance later.

#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <cstdint>
#include <string>
#include <array>
#include <random>
#include <mutex>
#include <openssl/aes.h>
#include "proto/pfss_backend.hpp"
#include "proto/pfss_backend_batch.hpp"
#include "proto/pfss_interval_lut_ext.hpp"
#include "gates/composite_fss.hpp"
#include "proto/packed_backend.hpp"

extern "C" __global__ void packed_cmp_kernel_keyed(const uint8_t* keys_flat,
                                                   size_t key_bytes,
                                                   const uint64_t* xs,
                                                   uint64_t* out_masks,
                                                   size_t N);
extern "C" __global__ void vector_lut_kernel_keyed(const uint8_t* keys_flat,
                                                   size_t key_bytes,
                                                   const uint64_t* xs,
                                                   uint64_t* out,
                                                   size_t N);

namespace proto {

namespace {

struct __attribute__((packed)) DcfKeyHeader {
  uint16_t in_bits;
  uint16_t payload_len;
};

struct __attribute__((packed)) IntervalKeyHeader {
  uint16_t in_bits;
  uint16_t out_words;
  uint32_t intervals;  // number of payload rows
  uint8_t party;
  uint8_t reserved[3];
  uint64_t nonce;
  uint8_t seed[16];
  uint8_t round_keys[176];
};

struct __attribute__((packed)) PackedCmpKeyHeader {
  uint16_t in_bits;
  uint16_t num_thr;
  uint8_t party;
  uint8_t reserved[3];
  uint64_t nonce;
  uint8_t seed[16];
  uint8_t round_keys[176];
};

inline void check_cuda(cudaError_t st, const char* what) {
  if (st != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(st));
  }
}

inline uint64_t mask_bits(int bits) {
  if (bits <= 0) return 0;
  if (bits >= 64) return ~uint64_t(0);
  return (uint64_t(1) << bits) - 1ull;
}

inline std::array<uint8_t,176> expand_aes_round_keys(const std::array<uint8_t,16>& seed) {
  AES_KEY key;
  AES_set_encrypt_key(seed.data(), 128, &key);
  std::array<uint8_t,176> rk{};
  std::memcpy(rk.data(), key.rd_key, 176);
  return rk;
}

inline std::array<uint8_t,16> seed_from_id(uint64_t id, uint64_t tweak) {
  std::array<uint8_t,16> seed{};
  std::mt19937_64 rng(id ^ (tweak * 0x9e3779b97f4a7c15ull));
  for (auto& b : seed) b = static_cast<uint8_t>(rng() & 0xFFu);
  return seed;
}

// Helpers to build and parse compact key blobs.
std::vector<uint8_t> build_dcf_key(int in_bits,
                                   const std::vector<u8>& alpha_bits,
                                   const std::vector<u8>& payload) {
  if (in_bits <= 0 || in_bits > 64) throw std::runtime_error("GpuPfssBackend: in_bits out of range");
  if (static_cast<int>(alpha_bits.size()) != in_bits) {
    throw std::runtime_error("GpuPfssBackend: alpha_bits size mismatch");
  }
  DcfKeyHeader hdr;
  hdr.in_bits = static_cast<uint16_t>(in_bits);
  hdr.payload_len = static_cast<uint16_t>(payload.size());
  std::vector<uint8_t> bytes(sizeof(DcfKeyHeader) + alpha_bits.size() + payload.size());
  std::memcpy(bytes.data(), &hdr, sizeof(DcfKeyHeader));
  std::memcpy(bytes.data() + sizeof(DcfKeyHeader), alpha_bits.data(), alpha_bits.size());
  if (!payload.empty()) {
    std::memcpy(bytes.data() + sizeof(DcfKeyHeader) + alpha_bits.size(), payload.data(), payload.size());
  }
  return bytes;
}

std::vector<uint8_t> build_interval_key(int party,
                                        uint64_t nonce,
                                        const IntervalLutDesc& desc,
                                        const std::vector<uint64_t>& payload_plain,
                                        const std::array<uint8_t,16>& seed) {
  if (desc.out_words <= 0) throw std::runtime_error("GpuPfssBackend: out_words must be >0");
  if (desc.cutpoints.size() < 2) throw std::runtime_error("GpuPfssBackend: need >=2 cutpoints");
  size_t intervals = desc.cutpoints.size() - 1;
  if (payload_plain.size() != intervals * static_cast<size_t>(desc.out_words)) {
    throw std::runtime_error("GpuPfssBackend: payload size mismatch");
  }
  IntervalKeyHeader hdr;
  hdr.in_bits = static_cast<uint16_t>(desc.in_bits);
  hdr.out_words = static_cast<uint16_t>(desc.out_words);
  hdr.intervals = static_cast<uint32_t>(intervals);
  hdr.party = static_cast<uint8_t>(party);
  hdr.nonce = nonce;
  std::memcpy(hdr.seed, seed.data(), seed.size());
  auto rk = expand_aes_round_keys(seed);
  std::memcpy(hdr.round_keys, rk.data(), rk.size());
  size_t bytes = sizeof(IntervalKeyHeader) + sizeof(uint64_t) * (intervals + 1) +
                 sizeof(uint64_t) * payload_plain.size();
  std::vector<uint8_t> blob(bytes);
  uint8_t* p = blob.data();
  std::memcpy(p, &hdr, sizeof(hdr));
  p += sizeof(hdr);
  std::memcpy(p, desc.cutpoints.data(), sizeof(uint64_t) * (intervals + 1));
  p += sizeof(uint64_t) * (intervals + 1);
  if (!payload_plain.empty()) {
    std::memcpy(p, payload_plain.data(), sizeof(uint64_t) * payload_plain.size());
  }
  return blob;
}

std::vector<uint8_t> build_packed_cmp_key(int party,
                                          uint64_t nonce,
                                          int in_bits,
                                          const std::vector<uint64_t>& thresholds,
                                          const std::array<uint8_t,16>& seed) {
  if (in_bits <= 0 || in_bits > 64) throw std::runtime_error("GpuPfssBackend: packed in_bits out of range");
  if (thresholds.empty()) throw std::runtime_error("GpuPfssBackend: packed thresholds empty");
  PackedCmpKeyHeader hdr;
  hdr.in_bits = static_cast<uint16_t>(in_bits);
  hdr.num_thr = static_cast<uint16_t>(thresholds.size());
  hdr.party = static_cast<uint8_t>(party);
  hdr.nonce = nonce;
  std::memcpy(hdr.seed, seed.data(), seed.size());
  auto rk = expand_aes_round_keys(seed);
  std::memcpy(hdr.round_keys, rk.data(), rk.size());
  size_t bytes = sizeof(PackedCmpKeyHeader) + sizeof(uint64_t) * thresholds.size();
  std::vector<uint8_t> blob(bytes);
  std::memcpy(blob.data(), &hdr, sizeof(hdr));
  uint64_t* thr_out = reinterpret_cast<uint64_t*>(blob.data() + sizeof(hdr));
  uint64_t mask = mask_bits(in_bits);
  for (size_t i = 0; i < thresholds.size(); i++) {
    thr_out[i] = thresholds[i] & mask;
  }
  return blob;
}

inline bool eval_dcf_host(const FssKey& key, int in_bits, const std::vector<u8>& x_bits,
                          std::vector<uint8_t>& out_bytes) {
  if (static_cast<int>(x_bits.size()) != in_bits) throw std::runtime_error("eval_dcf_host: x_bits mismatch");
  if (key.bytes.size() < sizeof(DcfKeyHeader)) throw std::runtime_error("eval_dcf_host: key too short");
  auto* hdr = reinterpret_cast<const DcfKeyHeader*>(key.bytes.data());
  if (hdr->in_bits != static_cast<uint16_t>(in_bits)) throw std::runtime_error("eval_dcf_host: in_bits mismatch");
  const uint8_t* alpha = key.bytes.data() + sizeof(DcfKeyHeader);
  const uint8_t* payload = alpha + in_bits;
  bool lt = false;
  for (int i = 0; i < in_bits; i++) {
    uint8_t xb = static_cast<uint8_t>(x_bits[static_cast<size_t>(i)] ? 1 : 0);
    uint8_t ab = alpha[static_cast<size_t>(i)] & 1u;
    if (xb < ab) { lt = true; break; }
    if (xb > ab) { lt = false; break; }
  }
  out_bytes.assign(hdr->payload_len, 0u);
  if (lt && hdr->payload_len > 0) {
    std::memcpy(out_bytes.data(), payload, hdr->payload_len);
  }
  return lt;
}

inline void eval_interval_host(const IntervalLutDesc& desc,
                               const FssKey& key,
                               const std::vector<uint64_t>& xs,
                               std::vector<uint8_t>& out_bytes) {
  if (key.bytes.size() < sizeof(IntervalKeyHeader)) throw std::runtime_error("eval_interval_host: key too short");
  const auto* hdr = reinterpret_cast<const IntervalKeyHeader*>(key.bytes.data());
  (void)desc;
  size_t intervals = hdr->intervals;
  if (intervals == 0) throw std::runtime_error("eval_interval_host: no intervals");
  const uint64_t* cuts = reinterpret_cast<const uint64_t*>(key.bytes.data() + sizeof(IntervalKeyHeader));
  const uint64_t* payload = cuts + (intervals + 1);
  out_bytes.resize(xs.size() * static_cast<size_t>(hdr->out_words) * sizeof(uint64_t));
  auto* out64 = reinterpret_cast<uint64_t*>(out_bytes.data());
  AES_KEY aes;
  AES_set_encrypt_key(hdr->seed, 128, &aes);
  uint64_t mask = mask_bits(hdr->in_bits);
  for (size_t i = 0; i < xs.size(); i++) {
    uint64_t x = xs[i] & mask;
    size_t idx = intervals - 1;
    for (size_t j = 0; j < intervals; j++) {
      if (x >= cuts[j] && x < cuts[j + 1]) { idx = j; break; }
    }
    const uint64_t* row = payload + idx * static_cast<size_t>(hdr->out_words);
    for (int w = 0; w < hdr->out_words; w++) {
      uint64_t ctr = hdr->nonce ^ (static_cast<uint64_t>(i) << 32) ^ (static_cast<uint64_t>(idx) * hdr->out_words + static_cast<uint64_t>(w));
      std::array<uint8_t,16> block{};
      std::memcpy(block.data(), &ctr, sizeof(uint64_t));
      AES_encrypt(block.data(), block.data(), &aes);
      uint64_t ks = 0;
      std::memcpy(&ks, block.data(), sizeof(uint64_t));
      uint64_t share = (hdr->party == 0) ? ks : (row[w] - ks);
      out64[i * static_cast<size_t>(hdr->out_words) + static_cast<size_t>(w)] = share;
    }
  }
}

inline void eval_packed_cmp_host(const FssKey& key,
                                 const std::vector<uint64_t>& xs,
                                 int out_words,
                                 uint64_t* outs_flat) {
  if (key.bytes.size() < sizeof(PackedCmpKeyHeader)) throw std::runtime_error("eval_packed_cmp_host: key too short");
  const auto* hdr = reinterpret_cast<const PackedCmpKeyHeader*>(key.bytes.data());
  const uint64_t* thresholds = reinterpret_cast<const uint64_t*>(key.bytes.data() + sizeof(PackedCmpKeyHeader));
  AES_KEY aes;
  AES_set_encrypt_key(hdr->seed, 128, &aes);
  uint64_t mask = mask_bits(hdr->in_bits);
  for (size_t i = 0; i < xs.size(); i++) {
    uint64_t x = xs[i] & mask;
    std::vector<uint64_t> words(static_cast<size_t>(out_words), 0);
    for (int w = 0; w < out_words; w++) {
      int base = w * 64;
      int limit = std::min(base + 64, static_cast<int>(hdr->num_thr));
      uint64_t word = 0;
      for (int b = base; b < limit; b++) {
        uint64_t thr = thresholds[b] & mask;
        if (x < thr) word |= (1ull << (b - base));
      }
      uint64_t ctr = hdr->nonce ^ (static_cast<uint64_t>(i) << 32) ^ static_cast<uint64_t>(w);
      std::array<uint8_t,16> block{};
      std::memcpy(block.data(), &ctr, sizeof(uint64_t));
      AES_encrypt(block.data(), block.data(), &aes);
      uint64_t ks = 0;
      std::memcpy(&ks, block.data(), sizeof(uint64_t));
      words[static_cast<size_t>(w)] = (hdr->party == 0) ? ks : (word ^ ks);
    }
    for (int w = 0; w < out_words; w++) {
      outs_flat[i * static_cast<size_t>(out_words) + static_cast<size_t>(w)] = words[static_cast<size_t>(w)];
    }
  }
}

__global__ void eval_dcf_many_kernel(const uint8_t* keys_flat,
                                     size_t key_bytes,
                                     int in_bits,
                                     int out_bytes,
                                     const uint64_t* xs,
                                     uint8_t* outs,
                                     size_t N) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  const uint8_t* kp = keys_flat + idx * key_bytes;
  auto* hdr = reinterpret_cast<const DcfKeyHeader*>(kp);
  const uint8_t* alpha = kp + sizeof(DcfKeyHeader);
  const uint8_t* payload = alpha + in_bits;
  uint64_t x = xs[idx];
  bool lt = false;
  for (int i = 0; i < in_bits; i++) {
    uint8_t xb = static_cast<uint8_t>((x >> (in_bits - 1 - i)) & 1ull);
    uint8_t ab = alpha[static_cast<size_t>(i)] & 1u;
    if (xb < ab) { lt = true; break; }
    if (xb > ab) { lt = false; break; }
  }
  uint8_t* out = outs + idx * static_cast<size_t>(out_bytes);
  if (lt) {
    for (int j = 0; j < out_bytes; j++) out[j] = payload[j];
  } else {
    for (int j = 0; j < out_bytes; j++) out[j] = 0u;
  }
}

__global__ void eval_interval_many_kernel(const uint8_t* keys_flat,
                                          size_t key_bytes,
                                          int out_words,
                                          const uint64_t* xs,
                                          uint64_t* outs,
                                          size_t N) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  const uint8_t* kp = keys_flat + idx * key_bytes;
  auto* hdr = reinterpret_cast<const IntervalKeyHeader*>(kp);
  const uint64_t* cuts = reinterpret_cast<const uint64_t*>(kp + sizeof(IntervalKeyHeader));
  const uint64_t* payload = cuts + (static_cast<size_t>(hdr->intervals) + 1);
  uint64_t x = xs[idx];
  size_t iv = static_cast<size_t>(hdr->intervals) - 1;
  for (size_t j = 0; j < hdr->intervals; j++) {
    if (x >= cuts[j] && x < cuts[j + 1]) { iv = j; break; }
  }
  const uint64_t* row = payload + iv * static_cast<size_t>(out_words);
  uint64_t* out = outs + idx * static_cast<size_t>(out_words);
  for (int w = 0; w < out_words; w++) out[w] = row[w];
}

}  // namespace

class GpuPfssBackend final : public PfssIntervalLutExt, public PackedLtBackend {
 public:
  BitOrder bit_order() const override { return BitOrder::MSB_FIRST; }

  std::vector<uint8_t> u64_to_bits_msb(u64 x, int in_bits) const override {
    std::vector<uint8_t> bits(in_bits);
    for (int i = 0; i < in_bits; i++) bits[i] = static_cast<uint8_t>((x >> (in_bits - 1 - i)) & 1u);
    return bits;
  }

  DcfKeyPair gen_dcf(int in_bits,
                     const std::vector<u8>& alpha_bits,
                     const std::vector<u8>& payload_bytes) override {
    DcfKeyPair kp;
    kp.k0.bytes = build_dcf_key(in_bits, alpha_bits, payload_bytes);
    kp.k1.bytes = build_dcf_key(in_bits, alpha_bits, std::vector<u8>(payload_bytes.size(), 0u));
    return kp;
  }

  PackedLtKeyPair gen_packed_lt(int in_bits, const std::vector<u64>& thresholds) override {
    uint64_t nonce = next_id_++;
    auto seed = seed_from_id(nonce, 0);
    PackedLtKeyPair kp;
    kp.k0.bytes = build_packed_cmp_key(0, nonce, in_bits, thresholds, seed);
    kp.k1.bytes = build_packed_cmp_key(1, nonce, in_bits, thresholds, seed);
    return kp;
  }

  IntervalLutKeyPair gen_interval_lut(const IntervalLutDesc& desc) override {
    IntervalLutKeyPair kp;
    uint64_t nonce = next_id_++;
    auto seed = seed_from_id(nonce, 7);
    kp.k0.bytes = build_interval_key(0, nonce, desc, desc.payload_flat, seed);
    kp.k1.bytes = build_interval_key(1, nonce, desc, desc.payload_flat, seed);
    return kp;
  }

  std::vector<uint8_t> eval_dcf(int in_bits, const FssKey& key, const std::vector<u8>& x) const override {
    std::vector<uint8_t> out;
    eval_dcf_host(key, in_bits, x, out);
    return out;
  }

  void eval_dcf_many_u64(int in_bits,
                         size_t key_bytes,
                         const uint8_t* keys_flat,
                         const std::vector<u64>& xs_u64,
                         int out_bytes,
                         uint8_t* outs_flat) const override {
    std::lock_guard<std::mutex> lg(mu_);
    if (xs_u64.empty()) return;
    if (key_bytes < sizeof(DcfKeyHeader)) throw std::runtime_error("GpuPfssBackend: key_bytes too small");
    const auto* hdr0 = reinterpret_cast<const DcfKeyHeader*>(keys_flat);
    if (hdr0->payload_len != static_cast<uint16_t>(out_bytes)) {
      throw std::runtime_error("GpuPfssBackend: out_bytes mismatch payload_len");
    }
    try {
      size_t keys_size = key_bytes * xs_u64.size();
      keys_buf_.ensure(keys_size);
      xs_buf_.ensure(xs_u64.size() * sizeof(uint64_t));
      out_buf_.ensure(xs_u64.size() * static_cast<size_t>(out_bytes));
      check_cuda(cudaMemcpyAsync(keys_buf_.ptr, keys_flat, keys_size, cudaMemcpyHostToDevice, stream_), "cudaMemcpy keys");
      check_cuda(cudaMemcpyAsync(xs_buf_.ptr, xs_u64.data(), xs_u64.size() * sizeof(uint64_t),
                                 cudaMemcpyHostToDevice, stream_), "cudaMemcpy xs");
      constexpr int kBlock = 256;
      int grid = static_cast<int>((xs_u64.size() + kBlock - 1) / kBlock);
      eval_dcf_many_kernel<<<grid, kBlock, 0, stream_>>>(keys_buf_.ptr, key_bytes, in_bits, out_bytes,
                                                         reinterpret_cast<const uint64_t*>(xs_buf_.ptr),
                                                         reinterpret_cast<uint8_t*>(out_buf_.ptr),
                                                         xs_u64.size());
      check_cuda(cudaGetLastError(), "eval_dcf_many_kernel");
      check_cuda(cudaMemcpyAsync(outs_flat, out_buf_.ptr,
                                 xs_u64.size() * static_cast<size_t>(out_bytes),
                                 cudaMemcpyDeviceToHost, stream_),
                 "cudaMemcpy outs");
      check_cuda(cudaStreamSynchronize(stream_), "stream sync");
      return;
    } catch (...) {
      // Fallback to host if CUDA path fails.
    }
    for (size_t i = 0; i < xs_u64.size(); i++) {
      FssKey kb;
      kb.bytes.assign(keys_flat + i * key_bytes, keys_flat + (i + 1) * key_bytes);
      auto bits = u64_to_bits_msb(xs_u64[i], in_bits);
      auto out = eval_dcf(in_bits, kb, bits);
      if (out.size() != static_cast<size_t>(out_bytes)) out.resize(static_cast<size_t>(out_bytes), 0);
      std::memcpy(outs_flat + i * static_cast<size_t>(out_bytes), out.data(), static_cast<size_t>(out_bytes));
    }
  }

  void eval_packed_lt_many(size_t key_bytes,
                           const uint8_t* keys_flat,
                           const std::vector<u64>& xs_u64,
                           int in_bits,
                           int out_words,
                           u64* outs_bitmask) const override {
    std::lock_guard<std::mutex> lg(mu_);
    if (xs_u64.empty()) return;
    if (key_bytes < sizeof(PackedCmpKeyHeader)) throw std::runtime_error("GpuPfssBackend: packed key too small");
    const auto* hdr0 = reinterpret_cast<const PackedCmpKeyHeader*>(keys_flat);
    if (hdr0->in_bits != static_cast<uint16_t>(in_bits)) throw std::runtime_error("GpuPfssBackend: packed in_bits mismatch");
    int expected_words = (static_cast<int>(hdr0->num_thr) + 63) / 64;
    if (expected_words != out_words) throw std::runtime_error("GpuPfssBackend: packed out_words mismatch");
    try {
      size_t keys_size = key_bytes * xs_u64.size();
      keys_buf_.ensure(keys_size);
      xs_buf_.ensure(xs_u64.size() * sizeof(uint64_t));
      out_buf_.ensure(xs_u64.size() * static_cast<size_t>(out_words) * sizeof(uint64_t));
      check_cuda(cudaMemcpyAsync(keys_buf_.ptr, keys_flat, keys_size, cudaMemcpyHostToDevice, stream_), "cudaMemcpy packed keys");
      check_cuda(cudaMemcpyAsync(xs_buf_.ptr, xs_u64.data(), xs_u64.size() * sizeof(uint64_t),
                                 cudaMemcpyHostToDevice, stream_), "cudaMemcpy packed xs");
      constexpr int kBlock = 256;
      int grid = static_cast<int>((xs_u64.size() + kBlock - 1) / kBlock);
      packed_cmp_kernel_keyed<<<grid, kBlock, 0, stream_>>>(keys_buf_.ptr, key_bytes,
                                                            reinterpret_cast<const uint64_t*>(xs_buf_.ptr),
                                                            reinterpret_cast<uint64_t*>(out_buf_.ptr),
                                                            xs_u64.size());
      check_cuda(cudaGetLastError(), "packed_cmp_kernel_keyed");
      check_cuda(cudaMemcpyAsync(outs_bitmask, out_buf_.ptr,
                                 xs_u64.size() * static_cast<size_t>(out_words) * sizeof(uint64_t),
                                 cudaMemcpyDeviceToHost, stream_),
                 "cudaMemcpy packed outs");
      check_cuda(cudaStreamSynchronize(stream_), "stream sync packed");
      return;
    } catch (...) {
      // fallback
    }
    for (size_t i = 0; i < xs_u64.size(); i++) {
      FssKey kb;
      kb.bytes.assign(keys_flat + i * key_bytes, keys_flat + (i + 1) * key_bytes);
      eval_packed_cmp_host(kb, std::vector<uint64_t>{xs_u64[i]}, out_words,
                           outs_bitmask + i * static_cast<size_t>(out_words));
    }
  }

  void eval_interval_lut_many_u64(size_t key_bytes,
                                  const uint8_t* keys_flat,
                                  const std::vector<u64>& xs_u64,
                                  int out_words,
                                  u64* outs_flat) const override {
    std::lock_guard<std::mutex> lg(mu_);
    if (xs_u64.empty()) return;
    if (key_bytes < sizeof(IntervalKeyHeader)) throw std::runtime_error("GpuPfssBackend: interval key too small");
    const auto* hdr0 = reinterpret_cast<const IntervalKeyHeader*>(keys_flat);
    if (hdr0->out_words != static_cast<uint16_t>(out_words)) {
      throw std::runtime_error("GpuPfssBackend: out_words mismatch payload");
    }
    try {
      size_t keys_size = key_bytes * xs_u64.size();
      keys_buf_.ensure(keys_size);
      xs_buf_.ensure(xs_u64.size() * sizeof(uint64_t));
      out_buf_.ensure(xs_u64.size() * static_cast<size_t>(out_words) * sizeof(uint64_t));
      check_cuda(cudaMemcpyAsync(keys_buf_.ptr, keys_flat, keys_size, cudaMemcpyHostToDevice, stream_),
                 "cudaMemcpy interval keys");
      check_cuda(cudaMemcpyAsync(xs_buf_.ptr, xs_u64.data(), xs_u64.size() * sizeof(uint64_t),
                                 cudaMemcpyHostToDevice, stream_),
                 "cudaMemcpy interval xs");
      constexpr int kBlock = 256;
      int grid = static_cast<int>((xs_u64.size() + kBlock - 1) / kBlock);
      vector_lut_kernel_keyed<<<grid, kBlock, 0, stream_>>>(keys_buf_.ptr, key_bytes,
                                                            reinterpret_cast<const uint64_t*>(xs_buf_.ptr),
                                                            reinterpret_cast<uint64_t*>(out_buf_.ptr),
                                                            xs_u64.size());
      check_cuda(cudaGetLastError(), "vector_lut_kernel_keyed");
      check_cuda(cudaMemcpyAsync(outs_flat,
                                 out_buf_.ptr,
                                 xs_u64.size() * static_cast<size_t>(out_words) * sizeof(uint64_t),
                                 cudaMemcpyDeviceToHost, stream_),
                 "cudaMemcpy interval outs");
      check_cuda(cudaStreamSynchronize(stream_), "stream sync");
      return;
    } catch (...) {
      // Fallback to host.
    }
    for (size_t i = 0; i < xs_u64.size(); i++) {
      FssKey kb;
      kb.bytes.assign(keys_flat + i * key_bytes, keys_flat + (i + 1) * key_bytes);
      std::vector<uint64_t> xs_single{xs_u64[i]};
      std::vector<uint8_t> tmp;
      eval_interval_host(IntervalLutDesc{}, kb, xs_single, tmp);
      const uint64_t* tmp64 = reinterpret_cast<const uint64_t*>(tmp.data());
      for (int w = 0; w < out_words; w++) {
        outs_flat[i * static_cast<size_t>(out_words) + static_cast<size_t>(w)] = tmp64[w];
      }
    }
  }

 private:
  struct DeviceBuffer {
    uint8_t* ptr = nullptr;
    size_t cap = 0;
    void ensure(size_t bytes) {
      if (bytes <= cap) return;
      release();
      check_cuda(cudaMalloc(&ptr, bytes), "cudaMalloc buffer");
      cap = bytes;
    }
    void release() {
      if (ptr) cudaFree(ptr);
      ptr = nullptr;
      cap = 0;
    }
  };

  cudaStream_t stream_{nullptr};
  mutable uint64_t next_id_ = 1;
  mutable DeviceBuffer keys_buf_;
  mutable DeviceBuffer xs_buf_;
  mutable DeviceBuffer out_buf_;
  mutable std::mutex mu_;
};

std::unique_ptr<PfssBackendBatch> make_real_gpu_backend() { return std::make_unique<GpuPfssBackend>(); }

}  // namespace proto
