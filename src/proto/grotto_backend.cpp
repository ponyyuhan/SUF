#include "proto/grotto_backend.hpp"

#ifdef SUF_HAVE_LIBDPF

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "proto/common.hpp"

#include "dpf/dpf_key.hpp"
#include "grotto/prefix_parity.hpp"

namespace proto {

struct GrottoBackend::Impl {
  using DpfKey = dpf::dpf_key<dpf::prg::aes128, dpf::prg::aes128, uint64_t, dpf::bit>;

  static inline void append_u16_le(std::vector<u8>& out, uint16_t v) {
    out.push_back(static_cast<u8>(v & 0xFFu));
    out.push_back(static_cast<u8>((v >> 8) & 0xFFu));
  }
  static inline uint16_t read_u16_le(const uint8_t* p) {
    return static_cast<uint16_t>(static_cast<uint16_t>(p[0]) |
                                 (static_cast<uint16_t>(p[1]) << 8));
  }
  static inline void append_u32_le(std::vector<u8>& out, uint32_t v) {
    out.push_back(static_cast<u8>(v & 0xFFu));
    out.push_back(static_cast<u8>((v >> 8) & 0xFFu));
    out.push_back(static_cast<u8>((v >> 16) & 0xFFu));
    out.push_back(static_cast<u8>((v >> 24) & 0xFFu));
  }
  static inline uint32_t read_u32_le(const uint8_t* p) {
    return static_cast<uint32_t>(static_cast<uint32_t>(p[0]) |
                                 (static_cast<uint32_t>(p[1]) << 8) |
                                 (static_cast<uint32_t>(p[2]) << 16) |
                                 (static_cast<uint32_t>(p[3]) << 24));
  }
  static inline void pad_to_align(std::vector<u8>& out, size_t align) {
    if (align == 0) return;
    size_t rem = out.size() % align;
    if (rem == 0) return;
    out.insert(out.end(), align - rem, 0u);
  }

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

  static uint8_t prefix_parity_single(const DpfKey& key, uint64_t x) {
    std::array<uint64_t, 1> endpoints{x};
    auto res = grotto::prefix_parities(key, endpoints);
    const auto& parities = std::get<0>(res);
    return parities[0] ? 1u : 0u;
  }

  uint8_t eval_lt_share(const DpfKey& key, int party, uint64_t x) const {
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

  Params params_{};
  SigmaFastBackend sigma_{};
  // Embedded key cache: hash(key_bytes) -> parsed DPF key (per-backend instance).
  mutable std::unordered_map<u64, std::shared_ptr<DpfKey>> dpf_cache_{};
  mutable std::mutex cache_mu_;

  static u64 hash64(const uint8_t* p, size_t n) {
    // FNV-1a 64-bit.
    u64 h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) {
      h ^= static_cast<u64>(p[i]);
      h *= 1099511628211ull;
    }
    return h;
  }

  static std::vector<u8> serialize_dpf_key(const DpfKey& k) {
    std::vector<u8> out;
    out.reserve(sizeof(typename DpfKey::interior_node) +
                sizeof(typename DpfKey::correction_words_array) +
                sizeof(typename DpfKey::correction_advice_array) +
                sizeof(std::tuple_element_t<0, typename DpfKey::leaf_tuple>) +
                sizeof(typename DpfKey::input_type));
    auto append_raw = [&](const void* ptr, size_t bytes) {
      const auto* b = reinterpret_cast<const uint8_t*>(ptr);
      out.insert(out.end(), b, b + bytes);
    };
    auto root = k.root();
    append_raw(&root, sizeof(root));
    auto cws = k.correction_words();
    append_raw(cws.data(), sizeof(cws));
    auto adv = k.correction_advice();
    append_raw(adv.data(), sizeof(adv));
    auto leaf0 = std::get<0>(k.leaf_nodes).get();
    append_raw(&leaf0, sizeof(leaf0));
    typename DpfKey::input_type offset_share{};
    append_raw(&offset_share, sizeof(offset_share));
    return out;
  }

  static DpfKey deserialize_dpf_key(const uint8_t* p, size_t n, size_t& off) {
    auto need = [&](size_t bytes) {
      if (off + bytes > n) throw std::runtime_error("GrottoBackend: dpf key truncated");
    };
    typename DpfKey::interior_node root{};
    need(sizeof(root));
    std::memcpy(&root, p + off, sizeof(root));
    off += sizeof(root);
    typename DpfKey::correction_words_array cws{};
    need(sizeof(cws));
    std::memcpy(cws.data(), p + off, sizeof(cws));
    off += sizeof(cws);
    typename DpfKey::correction_advice_array adv{};
    need(sizeof(adv));
    std::memcpy(adv.data(), p + off, sizeof(adv));
    off += sizeof(adv);
    using Leaf0 = std::tuple_element_t<0, typename DpfKey::leaf_tuple>;
    Leaf0 leaf0{};
    need(sizeof(leaf0));
    std::memcpy(&leaf0, p + off, sizeof(leaf0));
    off += sizeof(leaf0);
    typename DpfKey::input_type offset_share{};
    need(sizeof(offset_share));
    std::memcpy(&offset_share, p + off, sizeof(offset_share));
    off += sizeof(offset_share);
    typename DpfKey::leaf_tuple leaves{leaf0};
    typename DpfKey::beaver_tuple beavers{};
    return DpfKey(root, cws, adv, leaves, beavers, offset_share);
  }
};

GrottoBackend::GrottoBackend() : impl_(std::make_unique<Impl>()) {}
GrottoBackend::GrottoBackend(Params p) : impl_(std::make_unique<Impl>()) { impl_->params_ = p; }
GrottoBackend::~GrottoBackend() = default;

BitOrder GrottoBackend::bit_order() const { return BitOrder::MSB_FIRST; }

std::vector<u8> GrottoBackend::u64_to_bits_msb(u64 x, int in_bits) const {
  std::vector<u8> bits(in_bits);
  for (int i = 0; i < in_bits; i++) {
    int shift = (in_bits - 1 - i);
    bits[static_cast<size_t>(i)] = static_cast<u8>((x >> shift) & 1u);
  }
  return bits;
}

DcfKeyPair GrottoBackend::gen_dcf(int in_bits,
                                 const std::vector<u8>& alpha_bits,
                                 const std::vector<u8>& payload_bytes) {
  if (in_bits <= 0 || in_bits > 64) throw std::runtime_error("GrottoBackend: in_bits out of range");
  if (static_cast<int>(alpha_bits.size()) != in_bits) {
    throw std::runtime_error("GrottoBackend: alpha_bits size mismatch");
  }
  uint64_t alpha = Impl::bits_to_u64_msb(alpha_bits);
  alpha &= Impl::mask_bits(in_bits);
  auto kp = dpf::make_dpf<dpf::prg::aes128>(alpha);
  if (payload_bytes.size() > 0xFFFFu) throw std::runtime_error("GrottoBackend: payload too large");
  auto key_blob0 = Impl::serialize_dpf_key(kp.first);
  auto key_blob1 = Impl::serialize_dpf_key(kp.second);

  auto build = [&](int party, const std::vector<u8>& dpf_blob, const std::vector<u8>& payload) -> std::vector<u8> {
    std::vector<u8> out;
    out.reserve(4 + 1 + 1 + 2 + 4 + dpf_blob.size() + payload.size());
    out.insert(out.end(), {'G','D','C','1'});
    out.push_back(static_cast<u8>(party & 1));
    out.push_back(static_cast<u8>(in_bits));
    Impl::append_u16_le(out, static_cast<uint16_t>(payload.size()));
    Impl::append_u32_le(out, static_cast<uint32_t>(dpf_blob.size()));
    Impl::pad_to_align(out, 8);
    out.insert(out.end(), dpf_blob.begin(), dpf_blob.end());
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
  };
  DcfKeyPair out;
  out.k0.bytes = build(/*party=*/0, key_blob0, payload_bytes);
  out.k1.bytes = build(/*party=*/1, key_blob1, std::vector<u8>(payload_bytes.size(), 0u));
  return out;
}

std::vector<u8> GrottoBackend::eval_dcf(int in_bits,
                                       const FssKey& kb,
                                       const std::vector<u8>& x_bits) const {
  if (static_cast<int>(x_bits.size()) != in_bits) {
    throw std::runtime_error("GrottoBackend: eval_dcf x_bits size mismatch");
  }
  if (kb.bytes.size() < 4 || std::memcmp(kb.bytes.data(), "GDC1", 4) != 0) {
    throw std::runtime_error("GrottoBackend: unsupported key format (expected embedded GDC1)");
  }
  if (kb.bytes.size() < 4 + 1 + 1 + 2 + 4) throw std::runtime_error("GrottoBackend: key truncated");
  const int party = static_cast<int>(kb.bytes[4] & 1u);
  const int kb_bits = static_cast<int>(kb.bytes[5]);
  if (kb_bits != in_bits) throw std::runtime_error("GrottoBackend: in_bits mismatch");
  const uint16_t payload_len = Impl::read_u16_le(kb.bytes.data() + 6);
  const uint32_t blob_len = Impl::read_u32_le(kb.bytes.data() + 8);
  size_t off = 4 + 1 + 1 + 2 + 4;
  size_t pad = (8 - (off % 8)) & 7;
  off += pad;
  if (kb.bytes.size() < off + static_cast<size_t>(blob_len) + static_cast<size_t>(payload_len)) {
    throw std::runtime_error("GrottoBackend: key truncated body");
  }
  const uint8_t* blob = kb.bytes.data() + off;
  const uint8_t* payload_ptr = kb.bytes.data() + off + static_cast<size_t>(blob_len);
  const size_t payload_bytes = static_cast<size_t>(payload_len);
  const u64 key_hash = Impl::hash64(blob, static_cast<size_t>(blob_len));
  std::shared_ptr<Impl::DpfKey> key_sp;
  {
    std::lock_guard<std::mutex> lk(impl_->cache_mu_);
    auto it = impl_->dpf_cache_.find(key_hash);
    if (it != impl_->dpf_cache_.end()) {
      key_sp = it->second;
    }
  }
  if (!key_sp) {
    size_t boff = 0;
    auto key_obj = std::make_shared<Impl::DpfKey>(Impl::deserialize_dpf_key(blob, blob_len, boff));
    std::lock_guard<std::mutex> lk(impl_->cache_mu_);
    impl_->dpf_cache_[key_hash] = key_obj;
    key_sp = std::move(key_obj);
  }
  uint64_t x = Impl::bits_to_u64_msb(x_bits) & Impl::mask_bits(in_bits);
  uint8_t bit_share = impl_->eval_lt_share(*key_sp, party, x);
  std::vector<u8> out(payload_bytes, 0u);
  if ((bit_share & 1u) && payload_bytes) {
    std::memcpy(out.data(), payload_ptr, payload_bytes);
  }
  return out;
}

void GrottoBackend::eval_dcf_many_u64(int in_bits,
                                     size_t key_bytes,
                                     const uint8_t* keys_flat,
                                     const std::vector<u64>& xs_u64,
                                     int out_bytes,
                                     uint8_t* outs_flat) const {
  if (xs_u64.empty()) return;
  if (!keys_flat || !outs_flat) throw std::runtime_error("GrottoBackend: null buffers");
  if (key_bytes < 4) throw std::runtime_error("GrottoBackend: key_bytes too small");
  if (in_bits <= 0 || in_bits > 64) throw std::runtime_error("GrottoBackend: in_bits out of range");

  // Fast-path: embedded broadcast keys are extremely common; parse once and reuse.
  if (std::memcmp(keys_flat, "GDC1", 4) != 0) {
    throw std::runtime_error("GrottoBackend: unsupported key format (expected embedded GDC1)");
  }
  // If keys are not broadcast (rare), fall back to per-element evaluation.
  if (xs_u64.size() > 1 && std::memcmp(keys_flat, keys_flat + key_bytes, std::min<size_t>(key_bytes, 64)) != 0) {
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
  if (key_bytes < 4 + 1 + 1 + 2 + 4) throw std::runtime_error("GrottoBackend: key truncated");
  const int party = static_cast<int>(keys_flat[4] & 1u);
  const int kb_bits = static_cast<int>(keys_flat[5]);
  if (kb_bits != in_bits) throw std::runtime_error("GrottoBackend: in_bits mismatch");
  const uint16_t payload_len = Impl::read_u16_le(keys_flat + 6);
  const uint32_t blob_len = Impl::read_u32_le(keys_flat + 8);
  if (static_cast<int>(payload_len) != out_bytes) throw std::runtime_error("GrottoBackend: output size mismatch");
  size_t off = 4 + 1 + 1 + 2 + 4;
  size_t pad = (8 - (off % 8)) & 7;
  off += pad;
  if (key_bytes < off + static_cast<size_t>(blob_len) + static_cast<size_t>(payload_len)) {
    throw std::runtime_error("GrottoBackend: key truncated body");
  }
  const uint8_t* blob = keys_flat + off;
  const uint8_t* payload_ptr = keys_flat + off + static_cast<size_t>(blob_len);
  const u64 key_hash = Impl::hash64(blob, static_cast<size_t>(blob_len));
  std::shared_ptr<Impl::DpfKey> key_sp;
  {
    std::lock_guard<std::mutex> lk(impl_->cache_mu_);
    auto it = impl_->dpf_cache_.find(key_hash);
    if (it != impl_->dpf_cache_.end()) key_sp = it->second;
  }
  if (!key_sp) {
    size_t boff = 0;
    auto key_obj = std::make_shared<Impl::DpfKey>(Impl::deserialize_dpf_key(blob, blob_len, boff));
    std::lock_guard<std::mutex> lk(impl_->cache_mu_);
    impl_->dpf_cache_[key_hash] = key_obj;
    key_sp = std::move(key_obj);
  }

  std::vector<uint8_t> bit_shares;
  // Reuse the old batched sorter-based engine, but operating on a single party key.
  // NOTE: This keeps semantics identical to prefix_parities-based eval.
  bit_shares.assign(xs_u64.size(), 0);
  if (!xs_u64.empty()) {
    const uint64_t mask = Impl::mask_bits(in_bits);
    std::vector<size_t> order(xs_u64.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      return (xs_u64[a] & mask) < (xs_u64[b] & mask);
    });
    std::vector<uint64_t> xs_sorted(xs_u64.size());
    for (size_t i = 0; i < xs_u64.size(); ++i) xs_sorted[i] = xs_u64[order[i]] & mask;
    std::vector<uint8_t> bits_sorted(xs_u64.size(), 0);
    constexpr size_t kBlock = 128;
    const size_t blocks = (xs_sorted.size() + kBlock - 1) / kBlock;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (blocks >= 8)
#endif
    for (long long b = 0; b < static_cast<long long>(blocks); ++b) {
      size_t start = static_cast<size_t>(b) * kBlock;
      size_t take = std::min(kBlock, xs_sorted.size() - start);
      Impl::eval_prefix_parities_block<kBlock>(*key_sp, xs_sorted.data() + start, take, bits_sorted.data() + start);
    }
    if (party == 0) {
      for (auto& v : bits_sorted) v ^= 1u;
    }
    for (size_t i = 0; i < xs_u64.size(); ++i) bit_shares[order[i]] = bits_sorted[i];
  }

  for (size_t i = 0; i < xs_u64.size(); ++i) {
    uint8_t* out = outs_flat + i * static_cast<size_t>(out_bytes);
    if (bit_shares[i] & 1u) {
      std::memcpy(out, payload_ptr, static_cast<size_t>(payload_len));
    } else {
      std::memset(out, 0, static_cast<size_t>(payload_len));
    }
  }
}

PackedLtKeyPair GrottoBackend::gen_packed_lt(int in_bits, const std::vector<u64>& thresholds) {
  return impl_->sigma_.gen_packed_lt(in_bits, thresholds);
}

void GrottoBackend::eval_packed_lt_many(size_t key_bytes,
                                       const uint8_t* keys_flat,
                                       const std::vector<u64>& xs_u64,
                                       int in_bits,
                                       int out_words,
                                       u64* outs_bitmask) const {
  impl_->sigma_.eval_packed_lt_many(key_bytes, keys_flat, xs_u64, in_bits, out_words, outs_bitmask);
}

IntervalLutKeyPair GrottoBackend::gen_interval_lut(const IntervalLutDesc& desc) {
  return impl_->sigma_.gen_interval_lut(desc);
}

void GrottoBackend::eval_interval_lut_many_u64(size_t key_bytes,
                                              const uint8_t* keys_flat,
                                              const std::vector<u64>& xs_u64,
                                              int out_words,
                                              u64* outs_flat) const {
  impl_->sigma_.eval_interval_lut_many_u64(key_bytes, keys_flat, xs_u64, out_words, outs_flat);
}

}  // namespace proto

#endif  // SUF_HAVE_LIBDPF
