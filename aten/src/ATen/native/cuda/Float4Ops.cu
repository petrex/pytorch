#include <ATen/native/cuda/Float4Ops.cuh>
#include <c10/util/Float4_e2m1fn_x2.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>

namespace at {
namespace native {

// Convert from float32/bfloat16 to float4_e2m1fn_x2
template <typename T>
__global__ void cast_to_float4_kernel(
    Float4_e2m1fn_x2* output,
    const T* input,
    int64_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size/2) { // Process 2 elements at a time
    float val1 = static_cast<float>(input[2*idx]);
    float val2 = static_cast<float>(input[2*idx + 1]);
    output[idx] = Float4_e2m1fn_x2::from_float(val1, val2);
  }
}

// Convert from float4_e2m1fn_x2 to float32/bfloat16
template <typename T>
__global__ void cast_from_float4_kernel(
    T* output,
    const Float4_e2m1fn_x2* input,
    int64_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size/2) {
    float val1, val2;
    input[idx].to_float(val1, val2);
    output[2*idx] = static_cast<T>(val1);
    output[2*idx + 1] = static_cast<T>(val2);
  }
}

template <typename T>
Tensor& cast_to_float4_impl_kernel(const Tensor& src, Tensor& dst) {
  auto N = src.numel();
  TORCH_CHECK(N % 2 == 0, "Input size must be even");
  
  dim3 block(256);
  dim3 grid((N + 511) / 512);  // Each thread handles 2 elements
  
  auto stream = at::cuda::getCurrentCUDAStream();
  cast_to_float4_kernel<T><<<grid, block, 0, stream>>>(
    reinterpret_cast<Float4_e2m1fn_x2*>(dst.data_ptr()),
    reinterpret_cast<const T*>(src.data_ptr()),
    N);
  
  return dst;
}

Tensor& cast_to_float4_impl(const Tensor& src, Tensor& dst) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half, at::ScalarType::BFloat16,
    src.scalar_type(), "cast_to_float4", [&]() {
      cast_to_float4_impl_kernel<scalar_t>(src, dst);
    });
  return dst;
}

template <typename T>
Tensor& cast_from_float4_impl_kernel(const Tensor& src, Tensor& dst) {
  auto N = dst.numel();
  TORCH_CHECK(N % 2 == 0, "Output size must be even");
  
  dim3 block(256);
  dim3 grid((N + 511) / 512);  // Each thread handles 2 elements
  
  auto stream = at::cuda::getCurrentCUDAStream();
  cast_from_float4_kernel<T><<<grid, block, 0, stream>>>(
    reinterpret_cast<T*>(dst.data_ptr()),
    reinterpret_cast<const Float4_e2m1fn_x2*>(src.data_ptr()),
    N);
  
  return dst;
}

Tensor& cast_from_float4_impl(const Tensor& src, Tensor& dst) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half, at::ScalarType::BFloat16,
    dst.scalar_type(), "cast_from_float4", [&]() {
      cast_from_float4_impl_kernel<scalar_t>(src, dst);
    });
  return dst;
}

// Register the ops
TORCH_LIBRARY_IMPL(aten, CUDA, m) {
  m.impl("_cast_Float4_e2m1fn_x2", cast_to_float4_impl);
  m.impl("_cast_from_Float4_e2m1fn_x2", cast_from_float4_impl);
}

} // namespace native
} // namespace at 