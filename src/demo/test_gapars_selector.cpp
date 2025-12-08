#include <cassert>
#include <limits>

#include "compiler/range_analysis.hpp"

int main() {
  using compiler::AbsBound;
  using compiler::GapCert;
  using compiler::GateKind;
  using compiler::RangeKind;
  using compiler::default_mask_bound;
  using compiler::select_trunc_kind;
  using compiler::can_gapars;
  using compiler::gap_from_abs;

  // Hint only: should not enable GapARS.
  AbsBound hint;
  hint.is_signed = true;
  hint.max_abs = uint64_t(1) << 20;
  hint.kind = RangeKind::Hint;
  auto k_hint = select_trunc_kind(hint, /*frac_bits=*/8);
  assert(k_hint == GateKind::FaithfulARS);

  // Proof + small bound: GapARS allowed.
  AbsBound proof = hint;
  proof.kind = RangeKind::Proof;
  auto k_proof = select_trunc_kind(proof, /*frac_bits=*/8);
  assert(k_proof == GateKind::GapARS);

  // Unsigned values use faithful TR regardless of proof.
  AbsBound unsig;
  unsig.is_signed = false;
  unsig.max_abs = 16;
  unsig.kind = RangeKind::Proof;
  assert(select_trunc_kind(unsig, /*frac_bits=*/4) == GateKind::FaithfulTR);

  // Explicit certificate can drive selection even if abs_hint is weak.
  GapCert cert;
  cert.is_signed = true;
  cert.frac_bits = 6;
  cert.max_abs = 32;
  cert.mask_abs = default_mask_bound(6);
  cert.kind = RangeKind::Proof;
  AbsBound weak;
  weak.is_signed = true;
  weak.max_abs = std::numeric_limits<uint64_t>::max();
  weak.kind = RangeKind::Hint;
  assert(select_trunc_kind(weak, /*frac_bits=*/6, cert) == GateKind::GapARS);

  // can_gapars guard should fail near wrap.
  GapCert near_wrap;
  near_wrap.is_signed = true;
  near_wrap.frac_bits = 4;
  near_wrap.max_abs = (uint64_t(1) << 63) - 2;
  near_wrap.mask_abs = default_mask_bound(4);
  near_wrap.kind = RangeKind::Proof;
  assert(!can_gapars(near_wrap));

  // gap_from_abs helper mirrors proof-carrying bound.
  auto g = gap_from_abs(proof, /*frac_bits=*/8);
  assert(g.has_value());
  assert(can_gapars(*g));

  return 0;
}
