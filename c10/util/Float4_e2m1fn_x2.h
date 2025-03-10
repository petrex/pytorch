#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/TypeSafeSignMath.h>
#include <cstring>
#include <cmath>

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

  // Helper function to convert a single float to 4-bit MX format
  C10_HOST_DEVICE static uint8_t float_to_mx4(float val) {
    // Extract components from float
    uint32_t bits;
    memcpy(&bits, &val, sizeof(float));
    
    int exp = ((bits >> 23) & 0xFF) - 127; // Extract biased exponent
    uint32_t mant = bits & 0x7FFFFF; // Extract mantissa
    bool sign = bits >> 31; // Extract sign
    
    // Handle special cases
    if (exp == 128 || exp == -127) { // inf, nan, or zero
      return 0; // Return zero as per spec (no inf/nan support)
    }
    
    // Normalize mantissa and adjust exponent
    if (mant != 0) {
      while ((mant & 0x800000) == 0) {
        mant <<= 1;
        exp--;
      }
    }
    
    // Apply bias for 2-bit exponent
    exp += 1; // MX-FP4 uses bias of 1
    
    // Clamp exponent
    if (exp > 2) exp = 2; // Saturate to max
    if (exp < -1) exp = -1; // Saturate to min
    
    // Pack into 4-bit format (1-bit mantissa, 2-bit exp, implied sign)
    uint8_t mx4 = ((exp + 1) << 1) | ((mant >> 23) & 0x1);
    
    return mx4;
  }

  // Helper function to convert 4-bit MX format to float
  C10_HOST_DEVICE static float mx4_to_float(uint8_t mx4) {
    // Extract components
    int exp = ((mx4 >> 1) & 0x3) - 1; // Remove bias
    uint32_t mant = mx4 & 0x1;
    
    // Build float32
    uint32_t bits = 0;
    if (mant != 0 || exp != -1) { // Not zero
      exp += 127; // Apply float32 bias
      bits = (uint32_t(exp) << 23) | (mant << 22);
    }
    
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
  }

  // Convert from float32 following MX spec
  C10_HOST_DEVICE static Float4_e2m1fn_x2 from_float(float val1, float val2) {
    uint8_t hi = float_to_mx4(val1);
    uint8_t lo = float_to_mx4(val2);
    return pack(hi, lo);
  }

  // Convert to float32 following MX spec
  C10_HOST_DEVICE void to_float(float& val1, float& val2) const {
    uint8_t hi, lo;
    unpack(hi, lo);
    val1 = mx4_to_float(hi);
    val2 = mx4_to_float(lo);
  }
};

} // namespace c10 