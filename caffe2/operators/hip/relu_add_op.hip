#include "caffe2/core/hip/context_gpu.h"
#include "caffe2/operators/relu_add_op.h"
#include "hip/hip_runtime.h"

namespace caffe2 {

#ifdef __HIPCC__
typedef __half2 half2;
#endif

inline __device__ half2 greaterThanZero(half2 in) {
  return __hgt2(in, 0.0f16);
}

template <typename T>
__global__ void
FusedSumReluHIPKernel(const int N, const T* X, const T* Y, T* Z) {
  HIP_1D_KERNEL_LOOP(i, N) {
    Z[i] = fmaxf(0.0f, X[i] + Y[i]);
  }
}

__global__ void FusedAddReluHalf2HIPKernel(
    const int N,
    const half2* X,
    const half2* Y,
    half2* Z) {
  HIP_1D_KERNEL_LOOP(i, N) {
    half2 sum = __hadd2(X[i], Y[i]);
    Z[i] = __hmul2(greaterThanZero(sum), sum);
  }
}

template <typename T>
__global__ void FusedSumReluGradientHIPKernel(
    const int N,
    const T* X,
    const T* Y,
    const T* Z,
    T* dX) {
  HIP_1D_KERNEL_LOOP(i, N) {
    dX[i] = (Z[i] > 0.0f) ? (X[i] + Y[i]) : 0.0f;
  }
}

__global__ void FusedAddReluGradientHalf2HIPKernel(
    const int N,
    const half2* X,
    const half2* Y,
    const half2* Z,
    half2* dX) {
  HIP_1D_KERNEL_LOOP(i, N) {
    dX[i] = __hmul2(greaterThanZero(Z[i]), __hadd2(X[i], Y[i]));
  }
}

template <>
bool SumReluOp<float, HIPContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& Y = Input(1);
  const int N = X.dim32(0);

  CAFFE_ENFORCE_EQ(X.size(), Y.size());

  auto* Z = Output(0, Y.sizes(), at::dtype<float>());
  auto* Z_data = Z->mutable_data<float>();
  if (N == 0)
    return true;

  hipLaunchKernelGGL(
      FusedSumReluHIPKernel<float>,
      dim3(CAFFE_GET_BLOCKS(N)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context_.hip_stream(),
      N,
      X.data<float>(),
      Y.data<float>(),
      Z_data);
  return true;
}

/*
template <>
template <typename T>
bool AddReluFunctor<HIPContext>::
operator()(const int N, const T* X, const T* Y, T* Z, HIPContext* context) const
{ hipLaunchKernelGGL( FusedAddReluHIPKernel<T> , dim3(CAFFE_GET_BLOCKS(N)),
         dim3(CAFFE_HIP_NUM_THREADS),
         0,
         context->hip_stream(), N, X, Y, Z);
  return true;
}
*/

/*
template <>
template <>
bool AddReluFunctor<HIPContext>::operator()<at::Half>(
    const int N,
    const at::Half* X,
    const at::Half* Y,
    at::Half* Z,
    HIPContext* context) const {
   hipLaunchKernelGGL( FusedAddReluHalf2HIPKernel,
        dim3(CAFFE_GET_BLOCKS(N)),
        dim3(CAFFE_HIP_NUM_THREADS),
        0,
        context->hip_stream(),
        N,
        reinterpret_cast<const half2*>(X),
        reinterpret_cast<const half2*>(Y),
        reinterpret_cast<half2*>(Z));
  return true;
}
*/

template <>
bool SumReluGradientOp<float, HIPContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& Y = Input(1);
  auto& Z = Input(2);
  const int N = X.dim32(0);

  CAFFE_ENFORCE_EQ(X.size(), Y.size());
  CAFFE_ENFORCE_EQ(X.size(), Z.size());

  auto* dX = Output(0, X.sizes(), at::dtype<float>());
  auto* dX_data = dX->mutable_data<float>();

  if (N == 0)
    return true;

  hipLaunchKernelGGL(
      FusedSumReluGradientHIPKernel<float>,
      dim3(CAFFE_GET_BLOCKS(N)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context_.hip_stream(),
      N,
      X.data<float>(),
      Y.data<float>(),
      Z.data<float>(),
      dX_data);
  return true;
}

REGISTER_HIP_OPERATOR(SumRelu, SumReluOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(SumReluGradient, SumReluGradientOp<float, HIPContext>);

} // namespace caffe2
