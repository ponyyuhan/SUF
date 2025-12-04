#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include "proto/beaver_mul64.hpp"
#include "proto/channel.hpp"
#include "compiler/compiled_suf_gate.hpp"

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
                 const uint64_t*,
                 size_t,
                 const uint64_t*,
                 size_t,
                 size_t N,
                 uint64_t* haty_share_out) const override {
    // Pass through; caller should have filled haty_share_out already.
    (void)N;
    (void)haty_share_out;
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

}  // namespace gates
