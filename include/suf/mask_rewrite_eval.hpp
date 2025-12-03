#pragma once

#include <cstdint>
#include "suf/mask_rewrite.hpp"

namespace suf {

inline bool eval_rot_recipe(const RotCmp64Recipe& r, uint64_t hatx) {
  bool a = hatx < r.theta0;
  bool b = hatx < r.theta1;
  if (r.wrap) {
    return (!a) || b;
  }
  return (!a) && b;
}

inline bool eval_rot_low_recipe(const RotLowRecipe& r, uint64_t hatx) {
  uint64_t mask = (r.f >= 64) ? ~uint64_t(0) : ((uint64_t(1) << r.f) - 1);
  uint64_t z = hatx & mask;
  bool a = z < r.theta0;
  bool b = z < r.theta1;
  if (r.wrap) return (!a) || b;
  return (!a) && b;
}

inline bool eval_msb_add_recipe(const RotCmp64Recipe& r, uint64_t hatx) {
  // Recipe encodes membership in length-2^63 interval; MSB(x+c)=1-h
  return !eval_rot_recipe(r, hatx);
}

}  // namespace suf
