#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include "proto/beaver_mul64.hpp"
#include "proto/channel.hpp"

namespace gates {

// Post-processing hook interface for gates that need extra arithmetic after PFSS (e.g., ReluARS/GeLU).
struct PostProcHook {
  virtual ~PostProcHook() = default;
  virtual void run_batch(int party,
                         proto::IChannel& ch,
                         proto::BeaverMul64& mul,
                         const uint64_t* hatx_public,
                         const uint64_t* arith_share_in,
                         const uint64_t* bool_share_in,
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
                 const uint64_t*,
                 size_t N,
                 uint64_t* haty_share_out) const override {
    // Pass through; caller should have filled haty_share_out already.
    (void)N;
    (void)haty_share_out;
  }
};

// Placeholder GeLU post-proc: y = x_plus + delta (already additive). No extra work here.
struct GeLUPostProc final : public PostProcHook {
  void run_batch(int,
                 proto::IChannel&,
                 proto::BeaverMul64&,
                 const uint64_t*,
                 const uint64_t*,
                 const uint64_t*,
                 size_t,
                 uint64_t*) const override {
    // No-op placeholder; composite path already produced haty_share.
  }
};

// Placeholder ReluARS post-proc: hook point for trunc/LUT; currently no-op.
struct ReluARSPostProc final : public PostProcHook {
  void run_batch(int,
                 proto::IChannel&,
                 proto::BeaverMul64&,
                 const uint64_t*,
                 const uint64_t*,
                 const uint64_t*,
                 size_t,
                 uint64_t*) const override {
    // No-op placeholder; real ARS/LUT logic can be inserted here.
  }
};

}  // namespace gates
