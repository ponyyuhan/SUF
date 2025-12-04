#pragma once

#include "proto/pfss_interval_lut_ext.hpp"
#include <array>
#include <random>
#include <stdexcept>
#include <vector>
#include <unordered_map>
#include <openssl/aes.h>
#if defined(__AES__) && defined(__SSE2__)
  #include <wmmintrin.h>
  #include <emmintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace proto {

struct PackedLtKeyPair { FssKey k0, k1; };

class SigmaFastBackend : public PfssIntervalLutExt {
public:
  struct Params {
    int lambda_bytes = 16;
    bool xor_bitmask = true; // if true, packed bits are XOR-shares
  };

  SigmaFastBackend() : params_() {}
  explicit SigmaFastBackend(Params p) : params_(p) {}

  // Base PfssBackend interface (fallback to un-packed storage for compatibility)
  DcfKeyPair gen_dcf(int in_bits, const std::vector<u8>& alpha_bits, const std::vector<u8>& payload_bytes) override {
    Program p;
    p.in_bits = in_bits;
    p.alpha_bits = alpha_bits;
    p.payload0 = payload_bytes;
    p.payload1.assign(payload_bytes.size(), 0u);
    u64 id = next_id_++;
    progs_[id] = std::move(p);
    DcfKeyPair kp;
    kp.k0.bytes = proto::pack_u64_le(id << 1);
    kp.k1.bytes = proto::pack_u64_le((id << 1) | 1ull);
    return kp;
  }
  std::vector<u8> eval_dcf(int in_bits, const FssKey& kb, const std::vector<u8>& x_bits) const override {
    if (static_cast<int>(x_bits.size()) != in_bits) throw std::runtime_error("SigmaFastBackend::eval_dcf size mismatch");
    u64 kid = proto::unpack_u64_le(kb.bytes.data());
    u64 id = kid >> 1;
    int party = static_cast<int>(kid & 1ull);
    auto it = progs_.find(id);
    if (it == progs_.end()) throw std::runtime_error("SigmaFastBackend: unknown key");
    const auto& p = it->second;
    if (p.in_bits != in_bits) throw std::runtime_error("SigmaFastBackend: in_bits mismatch");
    bool lt = false;
    for (int i = 0; i < in_bits; i++) {
      u8 xb = x_bits[static_cast<size_t>(i)] & 1u;
      u8 ab = p.alpha_bits[static_cast<size_t>(i)] & 1u;
      if (xb < ab) { lt = true; break; }
      if (xb > ab) { lt = false; break; }
    }
    if (lt) return (party == 0) ? p.payload0 : p.payload1;
    return std::vector<u8>(p.payload0.size(), 0u);
  }
  std::vector<u8> u64_to_bits_msb(u64 x, int in_bits) const override {
    std::vector<u8> bits(in_bits);
    for (int i = 0; i < in_bits; i++) bits[i] = static_cast<u8>((x >> (in_bits - 1 - i)) & 1u);
    return bits;
  }

  // Packed multi-threshold compare (CDPF-style) with AES-CTR masks (party seeds).
  PackedLtKeyPair gen_packed_lt(int in_bits, const std::vector<u64>& thresholds) {
    if (in_bits <= 0 || in_bits > 64) throw std::runtime_error("SigmaFastBackend: in_bits out of range");
    // Deterministic IDs; store thresholds + AES seeds (one per party).
    u64 id = next_id_++;
    PackedEntry pe;
    pe.in_bits = in_bits;
    pe.thresholds = thresholds;
    pe.mask = (in_bits == 64) ? ~0ull : ((u64(1) << in_bits) - 1);
    pe.thresholds_masked.resize(thresholds.size());
    for (size_t i = 0; i < thresholds.size(); i++) {
      pe.thresholds_masked[i] = thresholds[i] & pe.mask;
    }
    std::array<uint8_t,16> seed0{}, seed1{};
    std::mt19937_64 rng(id * 0x9e3779b97f4a7c15ull + thresholds.size());
    for (auto& b : seed0) b = static_cast<uint8_t>(rng() & 0xFFu);
    for (auto& b : seed1) b = static_cast<uint8_t>(rng() & 0xFFu);
    pe.seed0 = seed0;
    pe.seed1 = seed1;
    AES_set_encrypt_key(pe.seed0.data(), 128, &pe.aes0);
    AES_set_encrypt_key(pe.seed1.data(), 128, &pe.aes1);
    packed_[id] = std::move(pe);
    PackedLtKeyPair kp;
    kp.k0.bytes = proto::pack_u64_le(id << 1);        // party 0
    kp.k1.bytes = proto::pack_u64_le((id << 1) | 1u); // party 1
    return kp;
  }

  // Evaluate packed compare bundle for many inputs: outs_bitmask is [N][out_words] u64
  void eval_packed_lt_many(size_t key_bytes,
                           const uint8_t* keys_flat,
                           const std::vector<u64>& xs_u64,
                           int in_bits,
                           int out_words,
                           u64* outs_bitmask) const {
    if (key_bytes < 8) throw std::runtime_error("SigmaFastBackend: key size too small");
    if (in_bits <= 0 || in_bits > 64) throw std::runtime_error("SigmaFastBackend: in_bits out of range");
    u64 mask = (in_bits == 64) ? ~0ull : ((u64(1) << in_bits) - 1);
    size_t total = xs_u64.size();
    const size_t block = 1024;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (long long bb = 0; bb < static_cast<long long>((total + block - 1) / block); bb++) {
      size_t blk_start = static_cast<size_t>(bb) * block;
      size_t blk_end = std::min(blk_start + block, total);
      u64 kid = proto::unpack_u64_le(keys_flat + blk_start * key_bytes);
      u64 id = kid >> 1;
      int party = static_cast<int>(kid & 1ull);
      auto it = packed_.find(id);
      if (it == packed_.end()) throw std::runtime_error("SigmaFastBackend: unknown key");
      if (it->second.in_bits != in_bits) throw std::runtime_error("SigmaFastBackend: in_bits mismatch");
      const auto& thr = it->second.thresholds_masked.empty() ? it->second.thresholds : it->second.thresholds_masked;
      std::vector<u64> cmp_masks(static_cast<size_t>(out_words) * (blk_end - blk_start), 0);
      build_mask_block_words(xs_u64, blk_start, blk_end, thr, mask, static_cast<size_t>(out_words), cmp_masks.data());
      // Apply keystream/masking per element.
      std::vector<u64> ks(static_cast<size_t>(out_words), 0);
      for (size_t idx = blk_start; idx < blk_end; idx++) {
        fill_keystream_words(it->second.aes0, (id << 32) ^ static_cast<u64>(idx), ks.data(), ks.size());
        for (int w = 0; w < out_words; w++) {
          u64 cm = cmp_masks[(idx - blk_start) * static_cast<size_t>(out_words) + static_cast<size_t>(w)];
          u64 share = ks[static_cast<size_t>(w)];
          if (params_.xor_bitmask && party == 1) share ^= cm;
          if (!params_.xor_bitmask && party == 1) share = cm;
          if (!params_.xor_bitmask && party == 0) share = 0ull;
          outs_bitmask[idx * out_words + static_cast<size_t>(w)] = share;
        }
      }
    }
  }

  // Interval LUT (vector payload) API
  IntervalLutKeyPair gen_interval_lut(const IntervalLutDesc& desc) override {
    u64 id = next_id_++;
    IntervalEntry ie;
    ie.desc = desc;
    if (desc.in_bits <= 0 || desc.in_bits > 64) throw std::runtime_error("SigmaFastBackend: interval in_bits out of range");
    ie.mask = (desc.in_bits == 64) ? ~0ull : ((u64(1) << desc.in_bits) - 1);
    ie.boundaries_masked.clear();
    if (desc.cutpoints.size() >= 2) {
      for (size_t i = 1; i + 1 < desc.cutpoints.size(); i++) {
        ie.boundaries_masked.push_back(desc.cutpoints[i] & ie.mask);
      }
    }
    std::mt19937_64 rng(id * 0xdeadbeefULL + desc.cutpoints.size());
    for (auto& b : ie.seed0) b = static_cast<uint8_t>(rng() & 0xFFu);
    for (auto& b : ie.seed1) b = static_cast<uint8_t>(rng() & 0xFFu);
    AES_set_encrypt_key(ie.seed0.data(), 128, &ie.aes0);
    AES_set_encrypt_key(ie.seed1.data(), 128, &ie.aes1);
    // Pre-mask payloads into additive shares using AES keystream.
    size_t words_total = desc.payload_flat.size();
    ie.payload0.resize(words_total);
    ie.payload1.resize(words_total);
    size_t intervals = desc.cutpoints.size() > 0 ? desc.cutpoints.size() - 1 : 0;
    for (size_t iv = 0; iv < intervals; iv++) {
      size_t base = iv * static_cast<size_t>(desc.out_words);
      std::vector<u64> ks(static_cast<size_t>(desc.out_words), 0);
      fill_keystream_words(ie.aes0, (id << 24) ^ static_cast<u64>(iv), ks.data(), ks.size());
      for (int w = 0; w < desc.out_words; w++) {
        size_t idx = base + static_cast<size_t>(w);
        u64 payload = desc.payload_flat[idx];
        ie.payload0[idx] = ks[static_cast<size_t>(w)];
        ie.payload1[idx] = proto::sub_mod(payload, ie.payload0[idx]);
      }
    }
    intervals_[id] = std::move(ie);
    IntervalLutKeyPair kp;
    kp.k0.bytes = proto::pack_u64_le(id << 1);
    kp.k1.bytes = proto::pack_u64_le((id << 1) | 1ull);
    return kp;
  }

  void eval_interval_lut_many_u64(size_t key_bytes,
                                  const uint8_t* keys_flat,
                                  const std::vector<u64>& xs_u64,
                                  int out_words,
                                  u64* outs_flat) const override {
    if (key_bytes < 8) throw std::runtime_error("SigmaFastBackend: key size too small");
    size_t total = xs_u64.size();
    const size_t block = 1024;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (long long bb = 0; bb < static_cast<long long>((total + block - 1) / block); bb++) {
      size_t blk_start = static_cast<size_t>(bb) * block;
      size_t blk_end = std::min(blk_start + block, total);
      u64 kid = proto::unpack_u64_le(keys_flat + blk_start * key_bytes);
      u64 id = kid >> 1;
      int party = static_cast<int>(kid & 1ull);
      auto it = intervals_.find(id);
      if (it == intervals_.end()) throw std::runtime_error("SigmaFastBackend: unknown interval key");
      const auto& ie = it->second;
      const auto& d = ie.desc;
      if (out_words != d.out_words) throw std::runtime_error("SigmaFastBackend: interval out_words mismatch");
      size_t intervals = d.cutpoints.size() > 1 ? d.cutpoints.size() - 1 : 0;
      if (intervals == 0) continue;
      const auto& bounds = ie.boundaries_masked;
      size_t cut_words = bounds.empty() ? 0 : ((bounds.size() + 63) / 64);
      std::vector<u64> cmp_masks(cut_words * (blk_end - blk_start), 0);
      if (cut_words > 0) {
        build_mask_block_words(xs_u64, blk_start, blk_end, bounds, ie.mask,
                               cut_words, cmp_masks.data());
      }
      for (size_t idx_i = blk_start; idx_i < blk_end; idx_i++) {
        size_t local = idx_i - blk_start;
        size_t interval_idx = intervals - 1;
        if (cut_words > 0) {
          const u64* base = cmp_masks.data() + local * cut_words;
          for (size_t b = 0; b < bounds.size(); b++) {
            size_t w = b >> 6;
            size_t bit = b & 63;
            u64 word = base[w];
            if ((word >> bit) & 1ull) { interval_idx = b; break; }
          }
        } else {
          interval_idx = 0;
        }
        for (int j = 0; j < out_words; j++) {
          size_t pos = interval_idx * static_cast<size_t>(d.out_words) + static_cast<size_t>(j);
          u64 share = (party == 0) ? ie.payload0[pos] : ie.payload1[pos];
          outs_flat[idx_i * out_words + static_cast<size_t>(j)] = share;
        }
      }
    }
  }

private:
  // Minimal AES-128 (software) for PRG; not constant-time.
  static uint8_t xtime(uint8_t x) { return static_cast<uint8_t>((x << 1) ^ ((x >> 7) * 0x1B)); }
  static void sub_bytes(uint8_t s[16]) {
    static const uint8_t box[256] = {
      0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
      0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
      0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
      0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
      0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
      0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
      0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
      0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
      0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
      0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
      0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
      0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
      0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
      0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
      0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
      0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
    };
    for (int i = 0; i < 16; i++) s[i] = box[s[i]];
  }
  static void shift_rows(uint8_t s[16]) {
    uint8_t t[16];
    t[0]=s[0]; t[1]=s[5]; t[2]=s[10]; t[3]=s[15];
    t[4]=s[4]; t[5]=s[9]; t[6]=s[14]; t[7]=s[3];
    t[8]=s[8]; t[9]=s[13]; t[10]=s[2]; t[11]=s[7];
    t[12]=s[12]; t[13]=s[1]; t[14]=s[6]; t[15]=s[11];
    std::memcpy(s, t, 16);
  }
  static void mix_columns(uint8_t s[16]) {
    for (int c = 0; c < 4; c++) {
      int idx = c * 4;
      uint8_t a0 = s[idx], a1 = s[idx+1], a2 = s[idx+2], a3 = s[idx+3];
      uint8_t r0 = static_cast<uint8_t>(xtime(a0) ^ xtime(a1) ^ a1 ^ a2 ^ a3);
      uint8_t r1 = static_cast<uint8_t>(a0 ^ xtime(a1) ^ xtime(a2) ^ a2 ^ a3);
      uint8_t r2 = static_cast<uint8_t>(a0 ^ a1 ^ xtime(a2) ^ xtime(a3) ^ a3);
      uint8_t r3 = static_cast<uint8_t>(xtime(a0) ^ a0 ^ a1 ^ a2 ^ xtime(a3));
      s[idx]=r0; s[idx+1]=r1; s[idx+2]=r2; s[idx+3]=r3;
    }
  }
  static void add_round_key(uint8_t s[16], const uint8_t rk[16]) {
    for (int i = 0; i < 16; i++) s[i] ^= rk[i];
  }
  static void sub_word(uint8_t t[4]) {
    static const uint8_t box[256] = {
      0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
      0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
      0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
      0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
      0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
      0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
      0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
      0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
      0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
      0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
      0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
      0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
      0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
      0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
      0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
      0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
    };
    t[0]=box[t[0]]; t[1]=box[t[1]]; t[2]=box[t[2]]; t[3]=box[t[3]];
  }
  static void expand_key(const std::array<uint8_t,16>& key, uint8_t rks[11][16]) {
    static const uint8_t Rcon[10] = {0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1B,0x36};
    std::memcpy(rks[0], key.data(), 16);
    for (int i = 1; i <= 10; i++) {
      uint8_t t[4];
      t[0]=rks[i-1][13]; t[1]=rks[i-1][14]; t[2]=rks[i-1][15]; t[3]=rks[i-1][12];
      sub_bytes(t);
      t[0] ^= Rcon[i-1];
      for (int j = 0; j < 16; j++) {
        if (j < 4) rks[i][j] = rks[i-1][j] ^ t[j];
        else rks[i][j] = rks[i-1][j] ^ rks[i][j-1];
      }
    }
  }
  static std::array<uint8_t,16> aes_encrypt_block(const std::array<uint8_t,16>& key,
                                                  const std::array<uint8_t,16>& in) {
    // Software AES-128 reference (no AES-NI). Deterministic per key/counter.
    static const uint8_t sbox[256] = {
      0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
      0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
      0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
      0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
      0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
      0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
      0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
      0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
      0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
      0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
      0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
      0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
      0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
      0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
      0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
      0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
    };
    auto xtime = [](uint8_t x) { return static_cast<uint8_t>((x << 1) ^ ((x >> 7) * 0x1B)); };
    auto sub_bytes = [&](uint8_t s[16]) {
      for (int i = 0; i < 16; i++) s[i] = sbox[s[i]];
    };
    auto shift_rows = [](uint8_t s[16]) {
      uint8_t t[16];
      t[0]=s[0]; t[1]=s[5]; t[2]=s[10]; t[3]=s[15];
      t[4]=s[4]; t[5]=s[9]; t[6]=s[14]; t[7]=s[3];
      t[8]=s[8]; t[9]=s[13]; t[10]=s[2]; t[11]=s[7];
      t[12]=s[12]; t[13]=s[1]; t[14]=s[6]; t[15]=s[11];
      std::memcpy(s, t, 16);
    };
    auto mix_columns = [&](uint8_t s[16]) {
      for (int c = 0; c < 4; c++) {
        int idx = c * 4;
        uint8_t a0 = s[idx], a1 = s[idx+1], a2 = s[idx+2], a3 = s[idx+3];
        uint8_t r0 = static_cast<uint8_t>(xtime(a0) ^ xtime(a1) ^ a1 ^ a2 ^ a3);
        uint8_t r1 = static_cast<uint8_t>(a0 ^ xtime(a1) ^ xtime(a2) ^ a2 ^ a3);
        uint8_t r2 = static_cast<uint8_t>(a0 ^ a1 ^ xtime(a2) ^ xtime(a3) ^ a3);
        uint8_t r3 = static_cast<uint8_t>(xtime(a0) ^ a0 ^ a1 ^ a2 ^ xtime(a3));
        s[idx]=r0; s[idx+1]=r1; s[idx+2]=r2; s[idx+3]=r3;
      }
    };
    auto add_round_key = [](uint8_t s[16], const uint8_t rk[16]) {
      for (int i = 0; i < 16; i++) s[i] ^= rk[i];
    };
    uint8_t rks[11][16];
    std::memcpy(rks[0], key.data(), 16);
    for (int i = 1; i <= 10; i++) {
      uint8_t t[4];
      t[0]=rks[i-1][13]; t[1]=rks[i-1][14]; t[2]=rks[i-1][15]; t[3]=rks[i-1][12];
      for (int k = 0; k < 4; k++) t[k] = sbox[t[k]];
      t[0] ^= static_cast<uint8_t>(0x01u << (i-1));
      for (int j = 0; j < 16; j++) {
        if (j < 4) rks[i][j] = rks[i-1][j] ^ t[j];
        else rks[i][j] = rks[i-1][j] ^ rks[i][j-4];
      }
    }
    uint8_t state[16];
    std::memcpy(state, in.data(), 16);
    add_round_key(state, rks[0]);
    for (int r = 1; r < 10; r++) {
      sub_bytes(state);
      shift_rows(state);
      mix_columns(state);
      add_round_key(state, rks[r]);
    }
    sub_bytes(state);
    shift_rows(state);
    add_round_key(state, rks[10]);
    std::array<uint8_t,16> out{};
    std::memcpy(out.data(), state, 16);
    return out;
  }

  // Build packed comparison words for a block of inputs (SoA over thresholds).
  static void build_mask_block_words(const std::vector<u64>& xs_u64,
                                     size_t blk_start,
                                     size_t blk_end,
                                     const std::vector<u64>& thresholds,
                                     u64 mask,
                                     size_t out_words,
                                     u64* dst_words) {
    const size_t blk_size = blk_end - blk_start;
    if (blk_size == 0 || out_words == 0) return;
    for (size_t w = 0; w < out_words; w++) {
      size_t t_start = w * 64;
      size_t t_end = std::min(t_start + 64, thresholds.size());
      size_t span = t_end - t_start;
      size_t local_idx = 0;
      for (size_t idx = blk_start; idx + 3 < blk_end; idx += 4) {
        u64 x0 = xs_u64[idx] & mask;
        u64 x1 = xs_u64[idx + 1] & mask;
        u64 x2 = xs_u64[idx + 2] & mask;
        u64 x3 = xs_u64[idx + 3] & mask;
        u64 w0 = 0, w1 = 0, w2 = 0, w3 = 0;
        for (size_t b = 0; b < span; b++) {
          u64 thr = thresholds[t_start + b] & mask;
          u64 bit = u64(1) << b;
          if (x0 < thr) w0 |= bit;
          if (x1 < thr) w1 |= bit;
          if (x2 < thr) w2 |= bit;
          if (x3 < thr) w3 |= bit;
        }
        dst_words[(local_idx + 0) * out_words + w] = w0;
        dst_words[(local_idx + 1) * out_words + w] = w1;
        dst_words[(local_idx + 2) * out_words + w] = w2;
        dst_words[(local_idx + 3) * out_words + w] = w3;
        local_idx += 4;
      }
      for (size_t idx = blk_start + local_idx; idx < blk_end; idx++) {
        u64 x = xs_u64[idx] & mask;
        u64 word = 0;
        for (size_t b = 0; b < span; b++) {
          u64 thr = thresholds[t_start + b] & mask;
          word |= (static_cast<u64>(x < thr) << b);
        }
        dst_words[(idx - blk_start) * out_words + w] = word;
      }
    }
  }

  Params params_;
  mutable u64 next_id_ = 1;
  struct PackedEntry {
    int in_bits;
    std::vector<u64> thresholds;
    std::vector<u64> thresholds_masked;
    u64 mask = ~0ull;
    std::array<uint8_t,16> seed0;
    std::array<uint8_t,16> seed1;
    AES_KEY aes0;
    AES_KEY aes1;
  };
  struct IntervalEntry {
    IntervalLutDesc desc;
    u64 mask = ~0ull;
    std::array<uint8_t,16> seed0;
    std::array<uint8_t,16> seed1;
    AES_KEY aes0;
    AES_KEY aes1;
    std::vector<u64> boundaries_masked;
    std::vector<u64> payload0; // additive share for party 0 (flat)
    std::vector<u64> payload1; // additive share for party 1 (flat)
  };
  mutable std::unordered_map<u64, PackedEntry> packed_;
  mutable std::unordered_map<u64, IntervalEntry> intervals_;
  struct Program {
    int in_bits;
    std::vector<u8> alpha_bits;
    std::vector<u8> payload0;
    std::vector<u8> payload1;
  };
  mutable std::unordered_map<u64, Program> progs_;

  // Generate keystream words using AES-CTR (fixed key, little-endian counter).
  static inline void fill_keystream_words(const AES_KEY& key, u64 counter, u64* out, size_t words) {
#if defined(__AES__) && defined(__SSE2__)
    // AES-NI path: process 2 counters per iteration to amortize overhead.
    size_t produced = 0;
    while (produced < words) {
      __m128i ctr0 = _mm_set_epi64x(0, static_cast<long long>(counter));
      __m128i ctr1 = _mm_set_epi64x(0, static_cast<long long>(counter + 1));
      __m128i b0 = ctr0, b1 = ctr1;
      b0 = _mm_xor_si128(b0, ((__m128i*)key.rd_key)[0]);
      b1 = _mm_xor_si128(b1, ((__m128i*)key.rd_key)[0]);
      for (int r = 1; r < key.rounds; r++) {
        b0 = _mm_aesenc_si128(b0, ((__m128i*)key.rd_key)[r]);
        b1 = _mm_aesenc_si128(b1, ((__m128i*)key.rd_key)[r]);
      }
      b0 = _mm_aesenclast_si128(b0, ((__m128i*)key.rd_key)[key.rounds]);
      b1 = _mm_aesenclast_si128(b1, ((__m128i*)key.rd_key)[key.rounds]);
      alignas(16) uint64_t buf[4];
      _mm_store_si128((__m128i*)(buf + 0), b0);
      _mm_store_si128((__m128i*)(buf + 2), b1);
      for (int k = 0; k < 4 && produced < words; k++) {
        out[produced++] = buf[k];
      }
      counter += 2;
    }
#else
    std::array<uint8_t,16> in{};
    std::array<uint8_t,16> block{};
    size_t produced = 0;
    while (produced < words) {
      std::memcpy(in.data(), &counter, sizeof(u64));
      AES_encrypt(in.data(), block.data(), &key);
      uint64_t w0 = 0, w1 = 0;
      std::memcpy(&w0, block.data(), sizeof(uint64_t));
      std::memcpy(&w1, block.data() + 8, sizeof(uint64_t));
      out[produced++] = w0;
      if (produced < words) out[produced++] = w1;
      counter++;
    }
#endif
  }
};

}  // namespace proto
