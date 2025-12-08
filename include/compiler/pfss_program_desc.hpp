#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace compiler {

// Conservative interval for a value in Z2^64 (interpretable as signed or unsigned).
struct RangeInterval {
  int64_t lo = std::numeric_limits<int64_t>::min();
  int64_t hi = std::numeric_limits<int64_t>::max();
  bool is_signed = true;

  static RangeInterval whole(bool signed_flag = true) {
    RangeInterval r;
    r.is_signed = signed_flag;
    if (!signed_flag) {
      r.lo = 0;
      r.hi = std::numeric_limits<int64_t>::max();
    }
    return r;
  }
};

enum class RangeKind : uint8_t { Hint = 0, Proof = 1 };

struct GapCert {
  bool is_signed = true;
  int frac_bits = 0;
  uint64_t max_abs = std::numeric_limits<uint64_t>::max();   // bound on |x_int|
  uint64_t mask_abs = std::numeric_limits<uint64_t>::max();  // bound on |r_int| used by trunc bundle
  RangeKind kind = RangeKind::Hint;
};

struct AbsBound {
  bool is_signed = true;
  uint64_t max_abs = std::numeric_limits<uint64_t>::max();
  RangeKind kind = RangeKind::Hint;
};

enum class RawPredKind : uint8_t {
  kLtU64,
  kLtLow
};

struct RawPredQuery {
  RawPredKind kind = RawPredKind::kLtU64;
  uint8_t f = 64;       // only used for kLtLow
  uint64_t theta = 0;   // threshold in that domain
};

enum class PredOutMode : uint8_t {
  kU64PerBit         = 0,  // legacy additive bit shares
  kU64PerBit_Xor     = 0,  // alias: XOR bit in LSB of u64
  kU64PerBit_Add     = 1,
  kPackedMask_Xor    = 2
};

struct PredProgramDesc {
  int n = 64;
  PredOutMode out_mode = PredOutMode::kU64PerBit_Xor;
  std::vector<RawPredQuery> queries;  // deduplicated, stable order
};

enum class CoeffMode : uint8_t { kIntervalLut, kStepDcf };

struct IntervalPayload {
  uint64_t lo = 0;  // inclusive
  uint64_t hi = 0;  // exclusive, non-wrapping
  std::vector<uint64_t> payload_words; // size = out_words
};

struct CoeffProgramDesc {
  int n = 64;
  CoeffMode mode = CoeffMode::kStepDcf;
  int out_words = 0;  // e.g. r*(d+1)
  // Interval LUT mode
  std::vector<IntervalPayload> intervals;
  // Step-DCF mode
  std::vector<uint64_t> base_payload_words;         // size out_words
  std::vector<uint64_t> cutpoints_ge;               // sorted asc
  std::vector<std::vector<uint64_t>> deltas_words;  // each size out_words
};

enum class ShareSemantics : uint8_t {
  XorBytes,
  AddU64
};

enum class BitOrder : uint8_t {
  MSB_FIRST,
  LSB_FIRST
};

struct PredKeyMeta {
  int n = 64;
  PredOutMode out_mode = PredOutMode::kU64PerBit_Xor;
  BitOrder bit_order = BitOrder::MSB_FIRST;
  ShareSemantics sem = ShareSemantics::XorBytes;
  uint32_t out_words = 1;
};

struct CoeffKeyMeta {
  int n = 64;
  CoeffMode mode = CoeffMode::kStepDcf;
  ShareSemantics sem = ShareSemantics::AddU64;
  int out_words = 0;
};

enum class GateKind : uint8_t {
  SiLUSpline,
  NExp,
  Reciprocal,
  Rsqrt,
  SoftmaxBlock,
  LayerNormBlock,
  FaithfulTR,
  FaithfulARS,
  GapARS,
  AutoTrunc
};

struct GateParams {
  GateKind kind = GateKind::SiLUSpline;
  int frac_bits = 0;
  int nr_iters = 1;
  int segments = 0;
  size_t L = 0;
  double eps = 0.0;
  RangeInterval range_hint = RangeInterval::whole(true);
  AbsBound abs_hint;
  std::optional<GapCert> gap_hint;
};

} // namespace compiler
