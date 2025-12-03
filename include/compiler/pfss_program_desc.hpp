#pragma once

#include <cstdint>
#include <vector>

namespace compiler {

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
  kU64PerBit
};

struct PredProgramDesc {
  int n = 64;
  PredOutMode out_mode = PredOutMode::kU64PerBit;
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

} // namespace compiler
