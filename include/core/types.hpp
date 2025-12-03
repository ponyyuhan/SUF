#pragma once

#include <cstdint>
#include <vector>

namespace core {

// Simple byte container helper used in serialization-friendly structs.
using Bytes = std::vector<uint8_t>;

}  // namespace core
