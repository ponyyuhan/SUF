#include <iostream>
#include <vector>
#include <cstring>
#include "proto/backend_gpu.hpp"
#include "proto/packed_backend.hpp"
#include "proto/pfss_interval_lut_ext.hpp"

int main() {
#ifndef SUF_HAVE_CUDA
  std::cout << "SUF_HAVE_CUDA not defined; skipping packed CUDA tests.\n";
  return 0;
#else
  auto gpu = proto::make_real_gpu_backend();
  auto* packed = dynamic_cast<proto::PackedLtBackend*>(gpu.get());
  auto* lut = dynamic_cast<proto::PfssIntervalLutExt*>(gpu.get());
  if (!packed || !lut) {
    std::cout << "Packed CUDA backend not available; skipping.\n";
    return 0;
  }

  // Packed compare -> XOR bitmask reconstruction.
  {
    std::vector<uint64_t> thresholds = {5, 12, 33, 48, 60};
    int in_bits = 6;
    auto kp = packed->gen_packed_lt(in_bits, thresholds);
    size_t key_bytes = kp.k0.bytes.size();
    std::vector<uint64_t> xs = {0, 15, 63};
    std::vector<uint8_t> keys0(xs.size() * key_bytes), keys1(xs.size() * key_bytes);
    for (size_t i = 0; i < xs.size(); i++) {
      std::memcpy(keys0.data() + i * key_bytes, kp.k0.bytes.data(), key_bytes);
      std::memcpy(keys1.data() + i * key_bytes, kp.k1.bytes.data(), key_bytes);
    }
    int out_words = static_cast<int>((thresholds.size() + 63) / 64);
    std::vector<uint64_t> out0(xs.size() * static_cast<size_t>(out_words), 0),
        out1(xs.size() * static_cast<size_t>(out_words), 0);
    packed->eval_packed_lt_many(key_bytes, keys0.data(), xs, in_bits, out_words, out0.data());
    packed->eval_packed_lt_many(key_bytes, keys1.data(), xs, in_bits, out_words, out1.data());
    for (size_t i = 0; i < xs.size(); i++) {
      uint64_t recon = out0[i] ^ out1[i];
      uint64_t expect = 0;
      for (size_t j = 0; j < thresholds.size(); j++) {
        uint64_t bit = (xs[i] < thresholds[j]) ? 1ull : 0ull;
        expect |= (bit << j);
      }
      if (recon != expect) {
        std::cerr << "Packed compare mismatch idx=" << i << " got=" << recon << " expect=" << expect << "\n";
        return 1;
      }
    }
  }

  // Vector payload LUT (interval) -> additive shares.
  {
    proto::IntervalLutDesc desc;
    desc.in_bits = 6;
    desc.out_words = 2;
    desc.cutpoints = {0, 20, 40, 64};
    desc.payload_flat = {100, 200, 300, 400, 500, 600};
    auto kp = lut->gen_interval_lut(desc);
    size_t key_bytes = kp.k0.bytes.size();
    std::vector<uint64_t> xs = {0, 25, 50};
    std::vector<uint8_t> keys0(xs.size() * key_bytes), keys1(xs.size() * key_bytes);
    for (size_t i = 0; i < xs.size(); i++) {
      std::memcpy(keys0.data() + i * key_bytes, kp.k0.bytes.data(), key_bytes);
      std::memcpy(keys1.data() + i * key_bytes, kp.k1.bytes.data(), key_bytes);
    }
    std::vector<uint64_t> out0(xs.size() * static_cast<size_t>(desc.out_words), 0),
        out1(xs.size() * static_cast<size_t>(desc.out_words), 0);
    lut->eval_interval_lut_many_u64(key_bytes, keys0.data(), xs, desc.out_words, out0.data());
    lut->eval_interval_lut_many_u64(key_bytes, keys1.data(), xs, desc.out_words, out1.data());
    auto payload_for = [&](uint64_t x, size_t word) {
      size_t iv = (x < 20) ? 0 : (x < 40 ? 1 : 2);
      return desc.payload_flat[iv * static_cast<size_t>(desc.out_words) + word];
    };
    for (size_t i = 0; i < xs.size(); i++) {
      for (int w = 0; w < desc.out_words; w++) {
        uint64_t recon = out0[i * static_cast<size_t>(desc.out_words) + static_cast<size_t>(w)] +
                         out1[i * static_cast<size_t>(desc.out_words) + static_cast<size_t>(w)];
        uint64_t expect = payload_for(xs[i], static_cast<size_t>(w));
        if (recon != expect) {
          std::cerr << "Vector LUT mismatch idx=" << i << " word=" << w << " got=" << recon
                    << " expect=" << expect << "\n";
          return 1;
        }
      }
    }
  }

  std::cout << "CUDA packed PFSS tests passed.\n";
  return 0;
#endif
}
