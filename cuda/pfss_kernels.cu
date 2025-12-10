#include "pfss_cuda_api.hpp"
#include <cuda_runtime.h>
#include <cstdint>

// Lightweight AES-128 implementation (one block) for CUDA tests / PRG.
// Not optimized; correctness-first.
namespace {

__device__ __constant__ uint8_t kSBox[256] = {
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
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16};

__device__ inline uint8_t xtime(uint8_t x) { return uint8_t((x << 1) ^ (((x >> 7) & 1u) * 0x1b)); }

__device__ void mix_columns(uint8_t* s) {
  for (int c = 0; c < 4; c++) {
    uint8_t* col = s + 4 * c;
    uint8_t a0 = col[0], a1 = col[1], a2 = col[2], a3 = col[3];
    uint8_t t = a0 ^ a1 ^ a2 ^ a3;
    col[0] ^= t ^ xtime(a0 ^ a1);
    col[1] ^= t ^ xtime(a1 ^ a2);
    col[2] ^= t ^ xtime(a2 ^ a3);
    col[3] ^= t ^ xtime(a3 ^ a0);
  }
}

__device__ void sub_bytes_shift_rows(uint8_t* s) {
  uint8_t t[16];
  // SubBytes and ShiftRows combined.
  // Row0 unchanged
  t[0] = kSBox[s[0]];   t[4] = kSBox[s[4]];   t[8]  = kSBox[s[8]];  t[12] = kSBox[s[12]];
  // Row1 shift left by 1
  t[1] = kSBox[s[5]];   t[5] = kSBox[s[9]];   t[9]  = kSBox[s[13]]; t[13] = kSBox[s[1]];
  // Row2 shift left by 2
  t[2] = kSBox[s[10]];  t[6] = kSBox[s[14]];  t[10] = kSBox[s[2]];  t[14] = kSBox[s[6]];
  // Row3 shift left by 3
  t[3] = kSBox[s[15]];  t[7] = kSBox[s[3]];   t[11] = kSBox[s[7]];  t[15] = kSBox[s[11]];
  for (int i = 0; i < 16; i++) s[i] = t[i];
}

__device__ void add_round_key(uint8_t* s, const uint8_t* rk) {
  for (int i = 0; i < 16; i++) s[i] ^= rk[i];
}

__device__ void aes128_encrypt_block(uint8_t* state, const uint8_t* round_keys) {
  add_round_key(state, round_keys);  // round 0
  // rounds 1..9
  for (int r = 1; r <= 9; r++) {
    sub_bytes_shift_rows(state);
    mix_columns(state);
    add_round_key(state, round_keys + 16 * r);
  }
  // final round
  sub_bytes_shift_rows(state);
  add_round_key(state, round_keys + 160);
}

}  // namespace

extern "C" __global__ void aes128_ctr_kernel(uint8_t* out,
                                             const uint8_t* round_keys,
                                             uint64_t ctr_lo,
                                             uint64_t ctr_hi,
                                             size_t blocks) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= blocks) return;
  uint8_t state[16];
  uint64_t ctr = ctr_lo + idx;
  // little-endian counter in first 64 bits, high in next
  for (int i = 0; i < 8; i++) state[i] = static_cast<uint8_t>((ctr >> (8 * i)) & 0xFFu);
  for (int i = 0; i < 8; i++) state[8 + i] = static_cast<uint8_t>((ctr_hi >> (8 * i)) & 0xFFu);
  aes128_encrypt_block(state, round_keys);
  uint8_t* o = out + idx * 16;
  for (int i = 0; i < 16; i++) o[i] = state[i];
}

// Simple packed predicate kernel: for each x, compute a 64-bit mask where bit j = 1[x < thresholds[j]].
extern "C" __global__ void pred_mask_kernel(const uint64_t* thresholds,
                                            int num_thr,
                                            const uint64_t* xs,
                                            uint64_t* out_masks,
                                            size_t N) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  uint64_t x = xs[idx];
  uint64_t mask = 0;
  int limit = num_thr > 64 ? 64 : num_thr;
  for (int j = 0; j < limit; j++) {
    uint64_t bit = (x < thresholds[j]) ? 1ull : 0ull;
    mask |= (bit << j);
  }
  out_masks[idx] = mask;
}

struct __attribute__((packed)) PackedCmpKeyDev {
  uint16_t in_bits;
  uint16_t num_thr;
  uint8_t party;
  uint8_t reserved[3];
  uint64_t nonce;
  uint8_t seed[16];
  uint8_t round_keys[176];
};

struct __attribute__((packed)) IntervalKeyDev {
  uint16_t in_bits;
  uint16_t out_words;
  uint32_t intervals;
  uint8_t party;
  uint8_t reserved[3];
  uint64_t nonce;
  uint8_t seed[16];
  uint8_t round_keys[176];
};

__device__ inline uint64_t mask_bits_dev(int bits) {
  if (bits <= 0) return 0;
  if (bits >= 64) return ~0ull;
  return (1ull << bits) - 1ull;
}

// Packed compare using AES-derived masks (XOR shares) with keyed blob.
extern "C" __global__ void packed_cmp_kernel_keyed(const uint8_t* keys_flat,
                                                   size_t key_bytes,
                                                   const uint64_t* xs,
                                                   uint64_t* out_masks,
                                                   size_t N) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  auto* hdr = reinterpret_cast<const PackedCmpKeyDev*>(keys_flat + idx * key_bytes);
  const uint64_t* thresholds = reinterpret_cast<const uint64_t*>(keys_flat + idx * key_bytes + sizeof(PackedCmpKeyDev));
  int out_words = (static_cast<int>(hdr->num_thr) + 63) / 64;
  uint64_t mask = mask_bits_dev(hdr->in_bits);
  uint64_t x = xs[idx] & mask;
  // Process two mask words per AES block (low/high 64 bits) to reduce AES calls.
  for (int w = 0; w < out_words; w += 2) {
    int base0 = w * 64;
    int limit0 = (base0 + 64 < static_cast<int>(hdr->num_thr)) ? (base0 + 64) : static_cast<int>(hdr->num_thr);
    uint64_t word0 = 0;
    for (int b = base0; b < limit0; b++) {
      uint64_t thr = thresholds[b] & mask;
      if (x < thr) word0 |= (1ull << (b - base0));
    }
    uint64_t word1 = 0;
    if (w + 1 < out_words) {
      int base1 = (w + 1) * 64;
      int limit1 = (base1 + 64 < static_cast<int>(hdr->num_thr)) ? (base1 + 64) : static_cast<int>(hdr->num_thr);
      for (int b = base1; b < limit1; b++) {
        uint64_t thr = thresholds[b] & mask;
        if (x < thr) word1 |= (1ull << (b - base1));
      }
    }
    uint8_t ctr_block[16] = {0};
    uint64_t ctr = hdr->nonce ^ (static_cast<uint64_t>(idx) << 32) ^ static_cast<uint64_t>(w / 2);
    reinterpret_cast<uint64_t*>(ctr_block)[0] = ctr;
    reinterpret_cast<uint64_t*>(ctr_block)[1] = 0;
    aes128_encrypt_block(ctr_block, hdr->round_keys);
    uint64_t ks0 = reinterpret_cast<uint64_t*>(ctr_block)[0];
    uint64_t ks1 = reinterpret_cast<uint64_t*>(ctr_block)[1];
    uint64_t share0 = (hdr->party == 0) ? ks0 : (word0 ^ ks0);
    out_masks[idx * static_cast<size_t>(out_words) + static_cast<size_t>(w)] = share0;
    if (w + 1 < out_words) {
      uint64_t share1 = (hdr->party == 0) ? ks1 : (word1 ^ ks1);
      out_masks[idx * static_cast<size_t>(out_words) + static_cast<size_t>(w + 1)] = share1;
    }
  }
}

// Vector payload LUT using AES masks: party0=keystream, party1=payload-keystream (additive share).
extern "C" __global__ void vector_lut_kernel_keyed(const uint8_t* keys_flat,
                                                   size_t key_bytes,
                                                   const uint64_t* xs,
                                                   uint64_t* out,
                                                   size_t N) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  auto* hdr = reinterpret_cast<const IntervalKeyDev*>(keys_flat + idx * key_bytes);
  const uint64_t* cuts = reinterpret_cast<const uint64_t*>(keys_flat + idx * key_bytes + sizeof(IntervalKeyDev));
  const uint64_t* payload = cuts + (static_cast<size_t>(hdr->intervals) + 1);
  uint64_t mask = mask_bits_dev(hdr->in_bits);
  uint64_t x = xs[idx] & mask;
  int iv = static_cast<int>(hdr->intervals) - 1;
  for (uint32_t j = 0; j < hdr->intervals; j++) {
    uint64_t c0 = cuts[j] & mask;
    uint64_t c1 = cuts[j + 1] & mask;
    if (x >= c0 && x < c1) { iv = static_cast<int>(j); break; }
  }
  const uint64_t* row = payload + static_cast<size_t>(iv) * hdr->out_words;
  for (int w = 0; w < hdr->out_words; w += 2) {
    uint8_t ctr_block[16] = {0};
    uint64_t base_ctr = static_cast<uint64_t>(iv) * hdr->out_words + static_cast<uint64_t>(w);
    uint64_t ctr = hdr->nonce ^ (static_cast<uint64_t>(idx) << 32) ^ base_ctr;
    reinterpret_cast<uint64_t*>(ctr_block)[0] = ctr;
    reinterpret_cast<uint64_t*>(ctr_block)[1] = 0;
    aes128_encrypt_block(ctr_block, hdr->round_keys);
    uint64_t ks0 = reinterpret_cast<uint64_t*>(ctr_block)[0];
    uint64_t ks1 = reinterpret_cast<uint64_t*>(ctr_block)[1];
    uint64_t share0 = (hdr->party == 0) ? ks0 : (row[w] - ks0);
    out[idx * static_cast<size_t>(hdr->out_words) + static_cast<size_t>(w)] = share0;
    if (w + 1 < hdr->out_words) {
      uint64_t share1 = (hdr->party == 0) ? ks1 : (row[w + 1] - ks1);
      out[idx * static_cast<size_t>(hdr->out_words) + static_cast<size_t>(w + 1)] = share1;
    }
  }
}

// Unpack a dense bitstream of fixed-width (eff_bits) integers into u64 values.
extern "C" __global__ void unpack_eff_bits_kernel(const uint64_t* packed,
                                                  int eff_bits,
                                                  uint64_t* out,
                                                  size_t N) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  if (eff_bits <= 0 || eff_bits > 64) return;
  size_t bit_idx = idx * static_cast<size_t>(eff_bits);
  size_t word_idx = bit_idx >> 6;
  int bit_off = static_cast<int>(bit_idx & 63);
  uint64_t val = packed[word_idx] >> bit_off;
  int bits_used = 64 - bit_off;
  if (bits_used < eff_bits) {
    uint64_t hi = packed[word_idx + 1];
    val |= (hi << bits_used);
  }
  if (eff_bits < 64) {
    uint64_t mask = (eff_bits == 64) ? ~0ull : ((1ull << eff_bits) - 1ull);
    val &= mask;
  }
  out[idx] = val;
}

namespace cuda_pfss {

DeviceKey upload_key(const uint8_t* /*key_bytes*/, size_t /*key_len*/) {
  return {nullptr};
}

void free_key(DeviceKey) {}

void eval_batch_pred(int /*party*/, DeviceKey /*key*/,
                     const uint64_t* /*d_xhat*/, uint64_t* /*d_out*/,
                     size_t /*count*/) {}

void eval_batch_coeff(int /*party*/, DeviceKey /*key*/,
                      const uint64_t* /*d_xhat*/, uint64_t* /*d_out*/,
                      size_t /*count*/) {}

}  // namespace cuda_pfss
