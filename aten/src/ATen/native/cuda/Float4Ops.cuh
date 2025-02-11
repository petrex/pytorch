#pragma once

#include <ATen/Tensor.h>
#include <c10/util/Float4_e2m1fn_x2.h>

namespace at {
namespace native {

Tensor& cast_to_float4_impl(const Tensor& src, Tensor& dst);
Tensor& cast_from_float4_impl(const Tensor& src, Tensor& dst);

} // namespace native
} // namespace at 