#pragma once

#include <ATen/Tensor.h>
#include <c10/util/Float4_e2m1fn_x2.h>

namespace at {
namespace native {

// Scaled matrix multiplication: C = alpha * (A @ B)
// Where A and B are Float4_e2m1fn_x2 tensors
Tensor& scaled_mm_impl(
    const Tensor& a,
    const Tensor& b,
    Tensor& c,
    float alpha);

} // namespace native
} // namespace at 