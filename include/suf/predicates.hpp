#pragma once

#include <cstdint>
#include <variant>

namespace suf {

// Primitive predicates in the UNMASKED x-domain (your SUF definition)
struct Pred_X_lt_const { uint64_t beta; };                 // 1[x < beta]
struct Pred_X_mod2f_lt { int f; uint64_t gamma; };         // 1[x mod 2^f < gamma]
struct Pred_MSB_x {};                                      // MSB(x)
struct Pred_MSB_x_plus { uint64_t c; };                    // MSB(x + c)

using PrimitivePred = std::variant<
    Pred_X_lt_const,
    Pred_X_mod2f_lt,
    Pred_MSB_x,
    Pred_MSB_x_plus
>;

}  // namespace suf
