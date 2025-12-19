#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "proto/pfss_interval_lut_ext.hpp"
#include "proto/packed_backend.hpp"
#include "proto/sigma_fast_backend_ext.hpp"

#ifdef SUF_HAVE_LIBDPF
#include "dpf/dpf_key.hpp"
#include "grotto/prefix_parity.hpp"
#endif

namespace proto {

#ifdef SUF_HAVE_LIBDPF

// Grotto-backed PFSS adapter: uses libdpf for comparison DCFs, while forwarding
// packed comparisons + interval LUTs to SigmaFast for high-throughput paths.
class GrottoBackend final : public PfssIntervalLutExt, public PackedLtBackend {
 public:
  struct Params {
    size_t block_elems = 128;  // eval block size for prefix parities (compile-time cap)
  };

  GrottoBackend() = default;
  explicit GrottoBackend(Params p) : params_(p) {}

  BitOrder bit_order() const override { return BitOrder::MSB_FIRST; }

  std::vector<u8> u64_to_bits_msb(u64 x, int in_bits) const override {
    std::vector<u8> bits(in_bits);
    for (int i = 0; i < in_bits; i++) {
      int shift = (in_bits - 1 - i);
      bits[static_cast<size_t>(i)] = static_cast<u8>((x >> shift) & 1u);
    }
    return bits;
  }

  DcfKeyPair gen_dcf(int in_bits,
                     const std::vector<u8>& alpha_bits,
                     const std::vector<u8>& payload_bytes) override {
    if (in_bits <= 0 || in_bits > 64) throw std::runtime_error("GrottoBackend: in_bits out of range");
    if (static_cast<int>(alpha_bits.size()) != in_bits) {
      throw std::runtime_error("GrottoBackend: alpha_bits size mismatch");
    }
    uint64_t alpha = bits_to_u64_msb(alpha_bits);
    alpha &= mask_bits(in_bits);
    auto kp = dpf::make_dpf<dpf::prg::aes128>(alpha);
    std::vector<u8> payload0 = payload_bytes;
    std::vector<u8> payload1(payload_bytes.size(), 0u);
    uint64_t id = next_id_++;
    dcf_.emplace(id, DcfEntry(in_bits, std::move(payload0), std::move(payload1),
                              std::move(kp.first), std::move(kp.second)));
    DcfKeyPair out;
    out.k0.bytes = pack_u64_le(id << 1);
    out.k1.bytes = pack_u64_le((id << 1) | 1ull);
    return out;
  }

  std::vector<u8> eval_dcf(int in_bits,
                           const FssKey& kb,
                           const std::vector<u8>& x_bits) const override {
    if (static_cast<int>(x_bits.size()) != in_bits) {
      throw std::runtime_error("GrottoBackend: eval_dcf x_bits size mismatch");
    }
    auto [id, party] = decode_key(kb);
    auto it = dcf_.find(id);
    if (it == dcf_.end()) throw std::runtime_error("GrottoBackend: unknown key id");
    const auto& entry = it->second;
    if (entry.in_bits != in_bits) throw std::runtime_error("GrottoBackend: in_bits mismatch");
    uint64_t x = bits_to_u64_msb(x_bits) & mask_bits(in_bits);
    uint8_t bit_share = eval_lt_share(entry, party, x);
    const auto& payload = (party == 0) ? entry.payload0 : entry.payload1;
    std::vector<u8> out(payload.size(), 0u);
    if (bit_share & 1u) {
      std::memcpy(out.data(), payload.data(), payload.size());
    }
    return out;
  }

  void eval_dcf_many_u64(int in_bits,
                         size_t key_bytes,
                         const uint8_t* keys_flat,
                         const std::vector<u64>& xs_u64,
                         int out_bytes,
                         uint8_t* outs_flat) const override {
    if (xs_u64.empty()) return;
    if (!keys_flat || !outs_flat) throw std::runtime_error("GrottoBackend: null buffers");
    if (key_bytes < 8) throw std::runtime_error("GrottoBackend: key_bytes too small");
    if (in_bits <= 0 || in_bits > 64) throw std::runtime_error("GrottoBackend: in_bits out of range");

    auto decode_bytes = [&](const uint8_t* ptr) {
      u64 kid = unpack_u64_le(ptr);
      return std::make_pair(kid >> 1, static_cast<int>(kid & 1ull));
    };
    const auto [id0, party0] = decode_bytes(keys_flat);
    bool broadcast = true;
    for (size_t i = 1; i < xs_u64.size(); ++i) {
      const auto [id_i, party_i] = decode_bytes(keys_flat + i * key_bytes);
      if (id_i != id0 || party_i != party0) {
        broadcast = false;
        break;
      }
    }
    if (!broadcast) {
      for (size_t i = 0; i < xs_u64.size(); ++i) {
        FssKey kb;
        kb.bytes.assign(keys_flat + i * key_bytes, keys_flat + (i + 1) * key_bytes);
        auto xb = u64_to_bits_msb(xs_u64[i], in_bits);
        auto out = eval_dcf(in_bits, kb, xb);
        if (out.size() != static_cast<size_t>(out_bytes)) {
          throw std::runtime_error("GrottoBackend: output size mismatch");
        }
        std::memcpy(outs_flat + i * static_cast<size_t>(out_bytes), out.data(), out.size());
      }
      return;
    }

    auto it = dcf_.find(id0);
    if (it == dcf_.end()) throw std::runtime_error("GrottoBackend: unknown key id");
    const auto& entry = it->second;
    if (entry.in_bits != in_bits) throw std::runtime_error("GrottoBackend: in_bits mismatch");
    const auto& payload = (party0 == 0) ? entry.payload0 : entry.payload1;
    if (payload.size() != static_cast<size_t>(out_bytes)) {
      throw std::runtime_error("GrottoBackend: output size mismatch");
    }

    std::vector<uint8_t> bit_shares;
    eval_lt_shares(entry, party0, xs_u64, bit_shares);

    for (size_t i = 0; i < xs_u64.size(); ++i) {
      uint8_t* out = outs_flat + i * static_cast<size_t>(out_bytes);
      if (bit_shares[i] & 1u) {
        std::memcpy(out, payload.data(), payload.size());
      } else {
        std::memset(out, 0, payload.size());
      }
    }
  }

  PackedLtKeyPair gen_packed_lt(int in_bits, const std::vector<u64>& thresholds) override {
    return sigma_.gen_packed_lt(in_bits, thresholds);
  }

  void eval_packed_lt_many(size_t key_bytes,
                           const uint8_t* keys_flat,
                           const std::vector<u64>& xs_u64,
                           int in_bits,
                           int out_words,
                           u64* outs_bitmask) const override {
    sigma_.eval_packed_lt_many(key_bytes, keys_flat, xs_u64, in_bits, out_words, outs_bitmask);
  }

  IntervalLutKeyPair gen_interval_lut(const IntervalLutDesc& desc) override {
    return sigma_.gen_interval_lut(desc);
  }

  void eval_interval_lut_many_u64(size_t key_bytes,
                                  const uint8_t* keys_flat,
                                  const std::vector<u64>& xs_u64,
                                  int out_words,
                                  u64* outs_flat) const override {
    sigma_.eval_interval_lut_many_u64(key_bytes, keys_flat, xs_u64, out_words, outs_flat);
  }

 private:
  using DpfKey = dpf::dpf_key<dpf::prg::aes128, dpf::prg::aes128, uint64_t, dpf::bit>;

  struct DcfEntry {
    int in_bits = 0;
    std::vector<u8> payload0;
    std::vector<u8> payload1;
    DpfKey key0;
    DpfKey key1;

    DcfEntry(int bits,
             std::vector<u8> p0,
             std::vector<u8> p1,
             DpfKey&& k0,
             DpfKey&& k1)
        : in_bits(bits), payload0(std::move(p0)), payload1(std::move(p1)),
          key0(std::move(k0)), key1(std::move(k1)) {}
  };

  static uint64_t mask_bits(int bits) {
    if (bits <= 0) return 0ull;
    if (bits >= 64) return ~uint64_t(0);
    return (uint64_t(1) << bits) - 1ull;
  }

  static uint64_t bits_to_u64_msb(const std::vector<u8>& bits) {
    uint64_t x = 0;
    for (size_t i = 0; i < bits.size(); ++i) {
      x = (x << 1) | (static_cast<uint64_t>(bits[i]) & 1ull);
    }
    return x;
  }

  static std::pair<u64, int> decode_key(const FssKey& kb) {
    if (kb.bytes.size() < 8) throw std::runtime_error("GrottoBackend: key truncated");
    u64 kid = unpack_u64_le(kb.bytes.data());
    u64 id = kid >> 1;
    int party = static_cast<int>(kid & 1ull);
    return {id, party};
  }

  static uint8_t prefix_parity_single(const DpfKey& key, uint64_t x) {
    std::array<uint64_t, 1> endpoints{x};
    auto res = grotto::prefix_parities(key, endpoints);
    const auto& parities = std::get<0>(res);
    return parities[0] ? 1u : 0u;
  }

  uint8_t eval_lt_share(const DcfEntry& entry, int party, uint64_t x) const {
    const DpfKey& key = (party == 0) ? entry.key0 : entry.key1;
    uint8_t ge_share = prefix_parity_single(key, x);
    // Convert GE share -> LT share by flipping party 0.
    if (party == 0) ge_share ^= 1u;
    return ge_share;
  }

  template <size_t kBlock>
  static void eval_prefix_parities_block(const DpfKey& key,
                                         const uint64_t* xs_sorted,
                                         size_t count,
                                         uint8_t* out_bits) {
    if (count == 0) return;
    std::array<uint64_t, kBlock> endpoints{};
    for (size_t i = 0; i < count; ++i) endpoints[i] = xs_sorted[i];
    for (size_t i = count; i < kBlock; ++i) endpoints[i] = xs_sorted[count - 1];
    auto res = grotto::prefix_parities(key, endpoints);
    const auto& parities = std::get<0>(res);
    for (size_t i = 0; i < count; ++i) out_bits[i] = parities[i] ? 1u : 0u;
  }

  void eval_lt_shares(const DcfEntry& entry,
                      int party,
                      const std::vector<u64>& xs,
                      std::vector<uint8_t>& out_bits) const {
    out_bits.assign(xs.size(), 0);
    if (xs.empty()) return;
    const uint64_t mask = mask_bits(entry.in_bits);
    std::vector<size_t> order(xs.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      return (xs[a] & mask) < (xs[b] & mask);
    });
    std::vector<uint64_t> xs_sorted(xs.size());
    for (size_t i = 0; i < xs.size(); ++i) xs_sorted[i] = xs[order[i]] & mask;
    std::vector<uint8_t> bits_sorted(xs.size(), 0);

    constexpr size_t kBlock = 128;
    const size_t blocks = (xs_sorted.size() + kBlock - 1) / kBlock;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (blocks >= 8)
#endif
    for (long long b = 0; b < static_cast<long long>(blocks); ++b) {
      size_t off = static_cast<size_t>(b) * kBlock;
      size_t take = std::min(kBlock, xs_sorted.size() - off);
      eval_prefix_parities_block<kBlock>((party == 0) ? entry.key0 : entry.key1,
                                         xs_sorted.data() + off,
                                         take,
                                         bits_sorted.data() + off);
    }
    if (party == 0) {
      for (auto& v : bits_sorted) v ^= 1u;
    }
    for (size_t i = 0; i < xs.size(); ++i) {
      out_bits[order[i]] = bits_sorted[i];
    }
  }

  Params params_{};
  SigmaFastBackend sigma_{};
  mutable std::unordered_map<u64, DcfEntry> dcf_;
  u64 next_id_ = 1;
};

#else

// Stub that falls back to SigmaFast when libdpf is unavailable.
class GrottoBackend final : public SigmaFastBackend {
 public:
  GrottoBackend() = default;
};

#endif  // SUF_HAVE_LIBDPF

}  // namespace proto
