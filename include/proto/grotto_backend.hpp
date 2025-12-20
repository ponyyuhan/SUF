#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "proto/pfss_interval_lut_ext.hpp"
#include "proto/packed_backend.hpp"
#include "proto/sigma_fast_backend_ext.hpp"

namespace proto {

#ifdef SUF_HAVE_LIBDPF

// Grotto-backed PFSS adapter.
//
// Implementation is isolated in a .cpp file to avoid ODR violations stemming from
// libdpf headers that include non-inline definitions.
class GrottoBackend final : public PfssIntervalLutExt, public PackedLtBackend {
 public:
  struct Params {
    size_t block_elems = 128;
  };

  GrottoBackend();
  explicit GrottoBackend(Params p);
  ~GrottoBackend() override;

  BitOrder bit_order() const override;
  std::vector<u8> u64_to_bits_msb(u64 x, int in_bits) const override;

  DcfKeyPair gen_dcf(int in_bits,
                     const std::vector<u8>& alpha_bits,
                     const std::vector<u8>& payload_bytes) override;
  std::vector<u8> eval_dcf(int in_bits,
                           const FssKey& kb,
                           const std::vector<u8>& x_bits) const override;
  void eval_dcf_many_u64(int in_bits,
                         size_t key_bytes,
                         const uint8_t* keys_flat,
                         const std::vector<u64>& xs_u64,
                         int out_bytes,
                         uint8_t* outs_flat) const override;

  PackedLtKeyPair gen_packed_lt(int in_bits, const std::vector<u64>& thresholds) override;
  void eval_packed_lt_many(size_t key_bytes,
                           const uint8_t* keys_flat,
                           const std::vector<u64>& xs_u64,
                           int in_bits,
                           int out_words,
                           u64* outs_bitmask) const override;

  IntervalLutKeyPair gen_interval_lut(const IntervalLutDesc& desc) override;
  void eval_interval_lut_many_u64(size_t key_bytes,
                                  const uint8_t* keys_flat,
                                  const std::vector<u64>& xs_u64,
                                  int out_words,
                                  u64* outs_flat) const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

#else

// Stub that falls back to SigmaFast when libdpf is unavailable.
class GrottoBackend final : public SigmaFastBackend {
 public:
  GrottoBackend() = default;
};

#endif  // SUF_HAVE_LIBDPF

}  // namespace proto

