#include <ATen/native/cuda/Float4Ops.cuh>
#include <c10/util/Float4_e2m1fn_x2.h>

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

// Register the ops
TORCH_LIBRARY_IMPL(aten, CUDA, m) {
  m.impl("_cast_Float4_e2m1fn_x2", cast_to_float4_impl);
  m.impl("_cast_from_Float4_e2m1fn_x2", cast_from_float4_impl);
}

} // namespace native
} // namespace at 