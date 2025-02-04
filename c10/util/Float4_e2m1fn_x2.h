#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/TypeSafeSignMath.h>
#include <cstring>

namespace c10 {

struct alignas(1) Float4_e2m1fn_x2 {
  uint8_t x;

  Float4_e2m1fn_x2() = default;
  C10_HOST_DEVICE Float4_e2m1fn_x2(uint8_t val) : x(val) {}

  // Pack two 4-bit values into one byte
  C10_HOST_DEVICE static Float4_e2m1fn_x2 pack(uint8_t hi, uint8_t lo) {
    return Float4_e2m1fn_x2((hi << 4) | (lo & 0xF));
  }

  // Unpack one byte into two 4-bit values
  C10_HOST_DEVICE void unpack(uint8_t& hi, uint8_t& lo) const {
    hi = (x >> 4) & 0xF;
    lo = x & 0xF;
  }

  // Convert from float32 following MX spec
  C10_HOST_DEVICE static Float4_e2m1fn_x2 from_float(float val1, float val2) {
    // Implement MX spec conversion with RNE rounding and saturation
    // This is a placeholder - actual implementation needed
    return Float4_e2m1fn_x2(0);
  }

  // Convert to float32 following MX spec
  C10_HOST_DEVICE void to_float(float& val1, float& val2) const {
    // Implement MX spec conversion
    // This is a placeholder - actual implementation needed
    val1 = 0.0f;
    val2 = 0.0f;
  }
};

} // namespace c10 