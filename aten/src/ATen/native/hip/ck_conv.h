#pragma once

#include <ATen/OpMathType.h>
#include <ATen/hip/HIPBlas.h>

namespace at::native {

template <typename Dtype>
inline void convolution_internal_ck(CUDABLAS_CONV_ARGTYPES(Dtype)) {
  static_assert(false && sizeof(Dtype), "at::cuda::convolution_internal_ck: not implemented");
}

template <>
void convolution_internal_ck<double>(CUDABLAS_CONV_ARGTYPES(double));
template <>
void convolution_internal_ck<float>(CUDABLAS_CONV_ARGTYPES(float));
template <>
void convolution_internal_ck<at::Half>(CUDABLAS_CONV_ARGTYPES(at::Half));
template <>
void convolution_internal_ck<at::BFloat16>(CUDABLAS_CONV_ARGTYPES(at::BFloat16));

} // namespace at::native
