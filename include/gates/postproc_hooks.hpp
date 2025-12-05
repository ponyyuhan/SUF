#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <cstring>
#include <memory>
#include <stdexcept>
#include "proto/beaver_mul64.hpp"
#include "proto/channel.hpp"
#include "compiler/compiled_suf_gate.hpp"
#include "proto/bit_ring_ops.hpp"

namespace compiler { struct TruncationLoweringResult; }

namespace runtime {
class PfssSuperBatch;
}

namespace proto {
struct PfssBackendBatch;
}

namespace compiler {
struct TruncationLoweringResult;
}

namespace gates {

// Post-processing hook interface for gates that need extra arithmetic after PFSS (e.g., ReluARS/GeLU).
struct PostProcHook {
  virtual ~PostProcHook() = default;
  virtual void configure(const compiler::PortLayout&) {}
  virtual void run_batch(int party,
                         proto::IChannel& ch,
                         proto::BeaverMul64& mul,
                         const uint64_t* hatx_public,
                         const uint64_t* arith_share_in,
                         size_t arith_stride,
                         const uint64_t* bool_share_in,
                         size_t bool_stride,
                         size_t N,
                         uint64_t* haty_share_out) const = 0;
};

// No-op hook (default when no post-processing is needed).
struct NoopPostProc final : public PostProcHook {
  void run_batch(int,
                 proto::IChannel&,
                 proto::BeaverMul64&,
                 const uint64_t*,
                 const uint64_t* arith_share_in,
                 size_t arith_stride,
                 const uint64_t*,
                 size_t,
                 size_t N,
                 uint64_t* haty_share_out) const override {
    // Pass-through: copy arithmetic inputs to outputs.
    for (size_t i = 0; i < N; i++) {
      const uint64_t* row = arith_share_in + i * arith_stride;
      uint64_t* dst = haty_share_out + i * arith_stride;
      std::memcpy(dst, row, arith_stride * sizeof(uint64_t));
    }
  }
};

// Placeholder GeLU post-proc: y = x_plus + delta (already additive). No extra work here.
struct GeLUPostProc final : public PostProcHook {
  int idx_x = 0;
  int idx_delta = 1;
  void configure(const compiler::PortLayout& layout) override {
    auto find = [&](const std::string& name)->int {
      for (size_t i = 0; i < layout.arith_ports.size(); i++) if (layout.arith_ports[i] == name) return static_cast<int>(i);
      return -1;
    };
    int x_idx = find("x_plus");
    if (x_idx >= 0) idx_x = x_idx;
    int d_idx = find("delta");
    if (d_idx >= 0) idx_delta = d_idx;
  }
  GeLUPostProc() = default;
  GeLUPostProc(int x_idx, int delta_idx) : idx_x(x_idx), idx_delta(delta_idx) {}
  void run_batch(int,
                 proto::IChannel&,
                 proto::BeaverMul64&,
                 const uint64_t*,
                 const uint64_t* arith_share_in,
                 size_t arith_stride,
                 const uint64_t*,
                 size_t,
                 size_t N,
                 uint64_t* haty_share_out) const override {
    for (size_t i = 0; i < N; i++) {
      const uint64_t* row = arith_share_in + i * arith_stride;
      uint64_t x = row[idx_x];
      uint64_t d = (idx_delta < static_cast<int>(arith_stride)) ? row[idx_delta] : 0ull;
      haty_share_out[i] = proto::add_mod(x, d);
    }
  }
};

// Placeholder ReluARS post-proc: hook point for trunc/LUT; currently no-op.
struct ReluARSPostProc final : public PostProcHook {
  int idx_w = 0, idx_t = 1, idx_d = 2;
  int idx_y = 0;  // assumes arith holds final y
  int f = 12;
  std::vector<uint64_t> delta; // expected size 8
  uint64_t r_hi_share = 0;
  uint64_t wrap_sign_share = 0;
  void configure(const compiler::PortLayout& layout) override {
    auto findb = [&](const std::string& name)->int {
      for (size_t i = 0; i < layout.bool_ports.size(); i++) if (layout.bool_ports[i] == name) return static_cast<int>(i);
      return -1;
    };
    auto finda = [&](const std::string& name)->int {
      for (size_t i = 0; i < layout.arith_ports.size(); i++) if (layout.arith_ports[i] == name) return static_cast<int>(i);
      return -1;
    };
    int w_idx = findb("w");
    if (w_idx >= 0) idx_w = w_idx;
    int t_idx = findb("t");
    if (t_idx >= 0) idx_t = t_idx;
    int d_idx = findb("d");
    if (d_idx >= 0) idx_d = d_idx;
    int y_idx = finda("y0");
    if (y_idx < 0) y_idx = finda("y");
    if (y_idx >= 0) idx_y = y_idx;
  }
  ReluARSPostProc() = default;
  ReluARSPostProc(int w, int t, int d, int y_idx, int f_bits,
                  uint64_t r_hi, uint64_t wrap_bit,
                  std::vector<uint64_t> delta_tab)
      : idx_w(w), idx_t(t), idx_d(d), idx_y(y_idx), f(f_bits),
        delta(std::move(delta_tab)), r_hi_share(r_hi), wrap_sign_share(wrap_bit) {}
  void run_batch(int party,
                 proto::IChannel& ch,
                 proto::BeaverMul64& mul,
                 const uint64_t* hatx_public,
                 const uint64_t* arith_share_in,
                 size_t arith_stride,
                 const uint64_t* bool_share_in,
                 size_t bool_stride,
                 size_t N,
                 uint64_t* haty_share_out) const override {
    (void)ch;
    proto::BitRingOps B{party, mul};
    for (size_t i = 0; i < N; i++) {
      const uint64_t* arow = arith_share_in + i * arith_stride;
      const uint64_t* brow = bool_share_in + i * bool_stride;
      uint64_t w = (idx_w < static_cast<int>(bool_stride)) ? brow[idx_w] : 0ull;
      uint64_t tbit = (idx_t < static_cast<int>(bool_stride)) ? brow[idx_t] : 0ull;
      uint64_t dbit = (idx_d < static_cast<int>(bool_stride)) ? brow[idx_d] : 0ull;
      uint64_t hatx = hatx_public[i];
      uint64_t hatz = proto::add_mod(hatx, (f == 0) ? 0ull : (1ull << (f - 1)));
      uint64_t H = (f == 64) ? 0 : (hatz >> f);
      uint64_t q = proto::sub_mod(proto::sub_mod((party == 0) ? H : 0ull, r_hi_share), tbit);

      uint64_t y = mul.mul(w, q);
      if (delta.size() >= 8) {
        std::array<uint64_t,8> tab{};
        for (size_t k = 0; k < 8 && k < delta.size(); k++) tab[k] = delta[k];
        uint64_t corr = proto::lut8_select(B, w, tbit, dbit, tab);
        y = proto::add_mod(y, corr);
      }
      // arith row is expected to hold r_out share (or base y share) at idx_y.
      uint64_t base = (idx_y < static_cast<int>(arith_stride)) ? arow[idx_y] : 0ull;
      haty_share_out[i] = proto::add_mod(y, base);
    }
  }
};

// Faithful truncation (unsigned) post-proc: y = (hatx >> f) - r_hi - carry + r_out_mask.
struct FaithfulTruncPostProc final : public PostProcHook {
  int idx_carry = 0;
  int idx_y = 0;
  int f = 0;
  uint64_t r_hi_share = 0;
  uint64_t r_in = 0;

  FaithfulTruncPostProc() = default;
  FaithfulTruncPostProc(int carry_idx, int out_idx, int frac_bits, uint64_t r_hi)
      : idx_carry(carry_idx), idx_y(out_idx), f(frac_bits), r_hi_share(r_hi) {}

  void configure(const compiler::PortLayout& layout) override {
    auto findb = [&](const std::string& name)->int {
      for (size_t i = 0; i < layout.bool_ports.size(); i++) if (layout.bool_ports[i] == name) return static_cast<int>(i);
      return -1;
    };
    auto finda = [&](const std::string& name)->int {
      for (size_t i = 0; i < layout.arith_ports.size(); i++) if (layout.arith_ports[i] == name) return static_cast<int>(i);
      return -1;
    };
    int c = findb("carry");
    if (c >= 0) idx_carry = c;
    int y = finda("y0");
    if (y < 0) y = finda("y");
    if (y >= 0) idx_y = y;
  }

  void run_batch(int party,
                 proto::IChannel&,
                 proto::BeaverMul64&,
                 const uint64_t* hatx_public,
                 const uint64_t* arith_share_in,
                 size_t arith_stride,
      const uint64_t* bool_share_in,
      size_t bool_stride,
      size_t N,
      uint64_t* haty_share_out) const override {
    uint64_t modulus = (f <= 0 || f >= 64) ? 0ull : (uint64_t(1) << (64 - f));
    for (size_t i = 0; i < N; i++) {
      const uint64_t* arow = arith_share_in + i * arith_stride;
      const uint64_t* brow = bool_share_in + i * bool_stride;
      uint64_t base = (idx_y < static_cast<int>(arith_stride)) ? arow[idx_y] : 0ull;
      uint64_t carry = (idx_carry >= 0 && idx_carry < static_cast<int>(bool_stride)) ? brow[idx_carry] : 0ull;
      uint64_t top = (hatx_public != nullptr && f < 64 && party == 0) ? (hatx_public[i] >> f) : 0ull;
      uint64_t y = proto::add_mod(base, top);
      y = proto::sub_mod(y, r_hi_share);
      y = proto::sub_mod(y, carry);
      if (modulus != 0 && hatx_public != nullptr && (hatx_public[i] < r_in) && party == 0) {
        y = proto::add_mod(y, modulus);
      }
      haty_share_out[i] = y;
    }
  }
};

// Faithful ARS: trunc + sign extension.
struct FaithfulArsPostProc : public PostProcHook {
  int idx_carry = 0;
  int idx_sign = 1;
  int idx_y = 0;
  int f = 0;
  uint64_t r_hi_share = 0;
  uint64_t r_in = 0;

  FaithfulArsPostProc() = default;
  FaithfulArsPostProc(int carry_idx, int sign_idx, int out_idx, int frac_bits, uint64_t r_hi)
      : idx_carry(carry_idx), idx_sign(sign_idx), idx_y(out_idx), f(frac_bits), r_hi_share(r_hi) {}

  void configure(const compiler::PortLayout& layout) override {
    auto findb = [&](const std::string& name)->int {
      for (size_t i = 0; i < layout.bool_ports.size(); i++) if (layout.bool_ports[i] == name) return static_cast<int>(i);
      return -1;
    };
    auto finda = [&](const std::string& name)->int {
      for (size_t i = 0; i < layout.arith_ports.size(); i++) if (layout.arith_ports[i] == name) return static_cast<int>(i);
      return -1;
    };
    int c = findb("carry");
    if (c >= 0) idx_carry = c;
    int s = findb("sign");
    if (s >= 0) idx_sign = s;
    int y = finda("y0");
    if (y < 0) y = finda("y");
    if (y >= 0) idx_y = y;
  }

  void run_batch(int party,
                 proto::IChannel&,
                 proto::BeaverMul64&,
                 const uint64_t* hatx_public,
                 const uint64_t* arith_share_in,
                 size_t arith_stride,
      const uint64_t* bool_share_in,
      size_t bool_stride,
      size_t N,
      uint64_t* haty_share_out) const override {
    uint64_t sign_mask = (f <= 0) ? 0ull : (f >= 64 ? 0ull : (~uint64_t(0) << (64 - f)));
    uint64_t modulus = (f <= 0 || f >= 64) ? 0ull : (uint64_t(1) << (64 - f));
    for (size_t i = 0; i < N; i++) {
      const uint64_t* arow = arith_share_in + i * arith_stride;
      const uint64_t* brow = bool_share_in + i * bool_stride;
      uint64_t base = (idx_y < static_cast<int>(arith_stride)) ? arow[idx_y] : 0ull;
      uint64_t carry = (idx_carry >= 0 && idx_carry < static_cast<int>(bool_stride)) ? brow[idx_carry] : 0ull;
      uint64_t sign = (idx_sign >= 0 && idx_sign < static_cast<int>(bool_stride)) ? brow[idx_sign] : 0ull;
      uint64_t top = (hatx_public != nullptr && f < 64 && party == 0) ? (hatx_public[i] >> f) : 0ull;
      uint64_t y = proto::add_mod(base, top);
      y = proto::sub_mod(y, r_hi_share);
      y = proto::sub_mod(y, carry);
      if (modulus != 0 && hatx_public != nullptr && (hatx_public[i] < r_in) && party == 0) {
        y = proto::add_mod(y, modulus);
      }
      uint64_t sign_term = proto::mul_mod(sign, sign_mask);
      y = proto::add_mod(y, sign_term);
      haty_share_out[i] = y;
    }
  }
};

// GapARS fast-path hook: currently same as faithful ARS.
using GapArsPostProc = FaithfulArsPostProc;

// Horner eval for cubic polynomial using public x̂ and secret-shared coeffs.
// Intended for SiLU/nExp payloads where PFSS returns coeffs (c0..c3) and
// runtime handles the Q32→Q16 rescale explicitly (currently via local shift).
struct HornerCubicHook : public PostProcHook {
  int frac_bits = 0;
  uint64_t r_in_share = 0;  // to recover x share from public x̂
  proto::PfssBackendBatch* backend = nullptr;
  mutable compiler::TruncationLoweringResult* trunc_bundle = nullptr;
  int trunc_frac_bits = 0;
  mutable std::mt19937_64 rng{0};

  HornerCubicHook() = default;
  HornerCubicHook(int fb, uint64_t r_in) : frac_bits(fb), r_in_share(r_in), trunc_frac_bits(fb) {}
  ~HornerCubicHook();

  void configure(const compiler::PortLayout& layout) override {
    (void)layout;
    // arith ports are positional payload coeffs; nothing to configure.
  }

  void run_batch(int party,
                 proto::IChannel&,
                 proto::BeaverMul64& mul,
                 const uint64_t* hatx_public,
                 const uint64_t* arith_share_in,
                 size_t arith_stride,
                 const uint64_t*,
                 size_t,
                 size_t N,
                 uint64_t* haty_share_out) const override;

 private:
  const compiler::TruncationLoweringResult& ensure_trunc_bundle() const;
};

}  // namespace gates
