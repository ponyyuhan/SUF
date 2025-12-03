#pragma once

#include "core/ring.hpp"
#include <cstdint>

namespace mpc {

template<typename RingT>
struct AddShare { RingT s; };  // x = x0 + x1 mod 2^n

struct XorShare { uint8_t b; };  // bit share: b = b0 XOR b1

}  // namespace mpc
