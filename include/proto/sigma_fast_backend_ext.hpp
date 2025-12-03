#pragma once

#include "proto/pfss_interval_lut_ext.hpp"
#include <stdexcept>
#include <vector>

namespace proto {

struct PackedLtKeyPair { FssKey k0, k1; };

class SigmaFastBackend : public PfssIntervalLutExt {
public:
  struct Params {
    int lambda_bytes = 16;
    bool xor_bitmask = true; // if true, packed bits are XOR-shares
  };

  explicit SigmaFastBackend(Params p = Params{}) : params_(p) {}

  // Base PfssBackend interface stubs
  DcfKeyPair gen_dcf(int, const std::vector<u8>&, const std::vector<u8>&) override {
    throw std::runtime_error("SigmaFastBackend::gen_dcf: use packed/interval APIs");
  }
  std::vector<u8> eval_dcf(int, const FssKey&, const std::vector<u8>&) const override {
    throw std::runtime_error("SigmaFastBackend::eval_dcf: use packed/interval APIs");
  }
  std::vector<u8> u64_to_bits_msb(u64 x, int in_bits) const override {
    std::vector<u8> bits(in_bits);
    for (int i = 0; i < in_bits; i++) bits[i] = static_cast<u8>((x >> (in_bits - 1 - i)) & 1u);
    return bits;
  }

  // Packed multi-threshold compare (CDPF-style)
  PackedLtKeyPair gen_packed_lt(int in_bits, const std::vector<u64>& thresholds) {
    (void)in_bits; (void)thresholds;
    throw std::runtime_error("SigmaFastBackend::gen_packed_lt not implemented");
  }

  // Evaluate packed compare bundle for many inputs: outs_bitmask is [N][out_words] u64
  void eval_packed_lt_many(size_t key_bytes,
                           const uint8_t* keys_flat,
                           const std::vector<u64>& xs_u64,
                           int in_bits,
                           int out_words,
                           u64* outs_bitmask) const {
    (void)key_bytes; (void)keys_flat; (void)xs_u64; (void)in_bits; (void)out_words; (void)outs_bitmask;
    throw std::runtime_error("SigmaFastBackend::eval_packed_lt_many not implemented");
  }

  // Interval LUT (vector payload) API
  IntervalLutKeyPair gen_interval_lut(const IntervalLutDesc& desc) override {
    (void)desc;
    throw std::runtime_error("SigmaFastBackend::gen_interval_lut not implemented");
  }

  void eval_interval_lut_many_u64(size_t key_bytes,
                                  const uint8_t* keys_flat,
                                  const std::vector<u64>& xs_u64,
                                  int out_words,
                                  u64* outs_flat) const override {
    (void)key_bytes; (void)keys_flat; (void)xs_u64; (void)out_words; (void)outs_flat;
    throw std::runtime_error("SigmaFastBackend::eval_interval_lut_many_u64 not implemented");
  }

private:
  Params params_;
};

}  // namespace proto
