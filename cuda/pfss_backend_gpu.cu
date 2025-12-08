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
#include "proto/pfss_backend.hpp"
#include "proto/pfss_backend_batch.hpp"
#include "proto/pfss_interval_lut_ext.hpp"
#include "gates/composite_fss.hpp"

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
};

inline void check_cuda(cudaError_t st, const char* what) {
  if (st != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(st));
  }
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

std::vector<uint8_t> build_interval_key(const IntervalLutDesc& desc,
                                        const std::vector<uint64_t>& payload_share) {
  if (desc.out_words <= 0) throw std::runtime_error("GpuPfssBackend: out_words must be >0");
  if (desc.cutpoints.size() < 2) throw std::runtime_error("GpuPfssBackend: need >=2 cutpoints");
  size_t intervals = desc.cutpoints.size() - 1;
  if (payload_share.size() != intervals * static_cast<size_t>(desc.out_words)) {
    throw std::runtime_error("GpuPfssBackend: payload size mismatch");
  }
  IntervalKeyHeader hdr;
  hdr.in_bits = static_cast<uint16_t>(desc.in_bits);
  hdr.out_words = static_cast<uint16_t>(desc.out_words);
  hdr.intervals = static_cast<uint32_t>(intervals);
  size_t bytes = sizeof(IntervalKeyHeader) + sizeof(uint64_t) * (intervals + 1) +
                 sizeof(uint64_t) * payload_share.size();
  std::vector<uint8_t> blob(bytes);
  uint8_t* p = blob.data();
  std::memcpy(p, &hdr, sizeof(hdr));
  p += sizeof(hdr);
  std::memcpy(p, desc.cutpoints.data(), sizeof(uint64_t) * (intervals + 1));
  p += sizeof(uint64_t) * (intervals + 1);
  if (!payload_share.empty()) {
    std::memcpy(p, payload_share.data(), sizeof(uint64_t) * payload_share.size());
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
  if (hdr->in_bits != static_cast<uint16_t>(desc.in_bits)) throw std::runtime_error("eval_interval_host: in_bits mismatch");
  if (hdr->out_words != static_cast<uint16_t>(desc.out_words)) throw std::runtime_error("eval_interval_host: out_words mismatch");
  size_t intervals = hdr->intervals;
  if (intervals == 0) throw std::runtime_error("eval_interval_host: no intervals");
  const uint64_t* cuts = reinterpret_cast<const uint64_t*>(key.bytes.data() + sizeof(IntervalKeyHeader));
  const uint64_t* payload = cuts + (intervals + 1);
  out_bytes.resize(xs.size() * static_cast<size_t>(hdr->out_words) * sizeof(uint64_t));
  auto* out64 = reinterpret_cast<uint64_t*>(out_bytes.data());
  for (size_t i = 0; i < xs.size(); i++) {
    uint64_t x = xs[i];
    size_t idx = intervals - 1;
    for (size_t j = 0; j < intervals; j++) {
      if (x >= cuts[j] && x < cuts[j + 1]) { idx = j; break; }
    }
    const uint64_t* row = payload + idx * static_cast<size_t>(hdr->out_words);
    for (int w = 0; w < hdr->out_words; w++) {
      out64[i * static_cast<size_t>(hdr->out_words) + static_cast<size_t>(w)] = row[w];
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

class GpuPfssBackend final : public PfssIntervalLutExt {
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

  IntervalLutKeyPair gen_interval_lut(const IntervalLutDesc& desc) override {
    IntervalLutKeyPair kp;
    kp.k0.bytes = build_interval_key(desc, desc.payload_flat);
    kp.k1.bytes = build_interval_key(desc, std::vector<uint64_t>(desc.payload_flat.size(), 0ull));
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
    if (xs_u64.empty()) return;
    if (key_bytes < sizeof(DcfKeyHeader)) throw std::runtime_error("GpuPfssBackend: key_bytes too small");
    const auto* hdr0 = reinterpret_cast<const DcfKeyHeader*>(keys_flat);
    if (hdr0->payload_len != static_cast<uint16_t>(out_bytes)) {
      throw std::runtime_error("GpuPfssBackend: out_bytes mismatch payload_len");
    }
    try {
      uint8_t* d_keys = nullptr;
      uint64_t* d_xs = nullptr;
      uint8_t* d_out = nullptr;
      size_t keys_size = key_bytes * xs_u64.size();
      check_cuda(cudaMalloc(&d_keys, keys_size), "cudaMalloc keys");
      check_cuda(cudaMalloc(&d_xs, xs_u64.size() * sizeof(uint64_t)), "cudaMalloc xs");
      check_cuda(cudaMalloc(&d_out, xs_u64.size() * static_cast<size_t>(out_bytes)), "cudaMalloc outs");
      check_cuda(cudaMemcpy(d_keys, keys_flat, keys_size, cudaMemcpyHostToDevice), "cudaMemcpy keys");
      check_cuda(cudaMemcpy(d_xs, xs_u64.data(), xs_u64.size() * sizeof(uint64_t), cudaMemcpyHostToDevice), "cudaMemcpy xs");
      constexpr int kBlock = 256;
      int grid = static_cast<int>((xs_u64.size() + kBlock - 1) / kBlock);
      eval_dcf_many_kernel<<<grid, kBlock>>>(d_keys, key_bytes, in_bits, out_bytes, d_xs, d_out, xs_u64.size());
      check_cuda(cudaGetLastError(), "eval_dcf_many_kernel");
      check_cuda(cudaMemcpy(outs_flat, d_out, xs_u64.size() * static_cast<size_t>(out_bytes), cudaMemcpyDeviceToHost),
                 "cudaMemcpy outs");
      cudaFree(d_keys);
      cudaFree(d_xs);
      cudaFree(d_out);
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

  void eval_interval_lut_many_u64(size_t key_bytes,
                                  const uint8_t* keys_flat,
                                  const std::vector<u64>& xs_u64,
                                  int out_words,
                                  u64* outs_flat) const override {
    if (xs_u64.empty()) return;
    if (key_bytes < sizeof(IntervalKeyHeader)) throw std::runtime_error("GpuPfssBackend: interval key too small");
    const auto* hdr0 = reinterpret_cast<const IntervalKeyHeader*>(keys_flat);
    if (hdr0->out_words != static_cast<uint16_t>(out_words)) {
      throw std::runtime_error("GpuPfssBackend: out_words mismatch payload");
    }
    try {
      uint8_t* d_keys = nullptr;
      uint64_t* d_xs = nullptr;
      uint64_t* d_out = nullptr;
      size_t keys_size = key_bytes * xs_u64.size();
      check_cuda(cudaMalloc(&d_keys, keys_size), "cudaMalloc interval keys");
      check_cuda(cudaMalloc(&d_xs, xs_u64.size() * sizeof(uint64_t)), "cudaMalloc interval xs");
      check_cuda(cudaMalloc(&d_out, xs_u64.size() * static_cast<size_t>(out_words) * sizeof(uint64_t)),
                 "cudaMalloc interval outs");
      check_cuda(cudaMemcpy(d_keys, keys_flat, keys_size, cudaMemcpyHostToDevice), "cudaMemcpy interval keys");
      check_cuda(cudaMemcpy(d_xs, xs_u64.data(), xs_u64.size() * sizeof(uint64_t), cudaMemcpyHostToDevice),
                 "cudaMemcpy interval xs");
      constexpr int kBlock = 256;
      int grid = static_cast<int>((xs_u64.size() + kBlock - 1) / kBlock);
      eval_interval_many_kernel<<<grid, kBlock>>>(d_keys, key_bytes, out_words, d_xs, d_out, xs_u64.size());
      check_cuda(cudaGetLastError(), "eval_interval_many_kernel");
      check_cuda(cudaMemcpy(outs_flat,
                            d_out,
                            xs_u64.size() * static_cast<size_t>(out_words) * sizeof(uint64_t),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy interval outs");
      cudaFree(d_keys);
      cudaFree(d_xs);
      cudaFree(d_out);
      return;
    } catch (...) {
      // Fallback to host.
    }
    for (size_t i = 0; i < xs_u64.size(); i++) {
      FssKey kb;
      kb.bytes.assign(keys_flat + i * key_bytes, keys_flat + (i + 1) * key_bytes);
      std::vector<uint64_t> xs_single{xs_u64[i]};
      std::vector<uint8_t> tmp;
      eval_interval_host(IntervalLutDesc{hdr0->in_bits, hdr0->out_words, {}, {}}, kb, xs_single, tmp);
      const uint64_t* tmp64 = reinterpret_cast<const uint64_t*>(tmp.data());
      for (int w = 0; w < out_words; w++) {
        outs_flat[i * static_cast<size_t>(out_words) + static_cast<size_t>(w)] = tmp64[w];
      }
    }
  }

};

std::unique_ptr<PfssBackendBatch> make_real_gpu_backend() { return std::make_unique<GpuPfssBackend>(); }

}  // namespace proto
