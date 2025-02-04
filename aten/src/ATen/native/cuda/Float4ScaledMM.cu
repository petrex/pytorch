#include <ATen/native/cuda/Float4ScaledMM.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <ATen/cuda/CUDABlas.h>

namespace at {
namespace native {

namespace {

// Helper to convert Float4_e2m1fn_x2 to appropriate CUDA data type
hipblasDatatype_t getBlasDataType() {
#if defined(USE_ROCM) && ROCM_VERSION >= 60500
    return HIPBLAS_R_4F_E2M1_X2;
#else
    TORCH_CHECK(false, "Float4_e2m1fn_x2 scaled MM only supported on ROCm >= 6.5.0");
    return HIPBLAS_R_32F;  // Unreachable, just to satisfy compiler
#endif
}

// Helper to check tensor properties
void checkInputs(const Tensor& a, const Tensor& b, const Tensor& c) {
    TORCH_CHECK(a.scalar_type() == ScalarType::Float4_e2m1fn_x2,
               "Expected Float4_e2m1fn_x2 tensor for input A");
    TORCH_CHECK(b.scalar_type() == ScalarType::Float4_e2m1fn_x2,
               "Expected Float4_e2m1fn_x2 tensor for input B");
    TORCH_CHECK(c.scalar_type() == ScalarType::Float,
               "Expected Float tensor for output C");
    
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2,
               "Expected 2D matrices");
    TORCH_CHECK(a.size(1) == b.size(0),
               "Size mismatch for matrix multiplication");
    TORCH_CHECK(c.size(0) == a.size(0) && c.size(1) == b.size(1),
               "Output tensor size mismatch");
}

} // anonymous namespace

Tensor& scaled_mm_impl(
    const Tensor& a,
    const Tensor& b,
    Tensor& c,
    float alpha) {
    
    checkInputs(a, b, c);
    
    // Get dimensions
    auto m = a.size(0);
    auto k = a.size(1);
    auto n = b.size(1);
    
    // Get strides
    auto lda = a.stride(0);
    auto ldb = b.stride(0);
    auto ldc = c.stride(0);
    
    // Get pointers
    auto a_ptr = reinterpret_cast<const Float4_e2m1fn_x2*>(a.data_ptr());
    auto b_ptr = reinterpret_cast<const Float4_e2m1fn_x2*>(b.data_ptr());
    auto c_ptr = c.data_ptr<float>();
    
    // Get CUDA handles
    auto handle = at::cuda::getCurrentCUDABlasHandle();
    auto stream = at::cuda::getCurrentCUDAStream();
    
    // Set stream
    TORCH_CUDABLAS_CHECK(hipblasSetStream(handle, stream));
    
    // Get data type
    auto data_type = getBlasDataType();
    
    // Perform matrix multiplication
    float beta = 0.0f;  // C = alpha * (A @ B) + beta * C
    
    TORCH_CUDABLAS_CHECK(hipblasGemmEx(
        handle,
        HIPBLAS_OP_N,  // No transpose for A
        HIPBLAS_OP_N,  // No transpose for B
        m, n, k,       // Dimensions
        &alpha,        // Alpha scaling factor
        a_ptr,         // Input matrix A
        data_type,     // A's data type
        lda,           // Leading dimension of A
        b_ptr,         // Input matrix B
        data_type,     // B's data type
        ldb,           // Leading dimension of B
        &beta,         // Beta scaling factor
        c_ptr,         // Output matrix C
        HIPBLAS_R_32F, // C's data type (float)
        ldc,           // Leading dimension of C
        HIPBLAS_R_32F, // Computation type
        HIPBLAS_GEMM_DEFAULT  // Algorithm
    ));
    
    return c;
}

// Register the op
TORCH_LIBRARY_IMPL(aten, CUDA, m) {
    m.impl("_scaled_mm_float4", scaled_mm_impl);
}

} // namespace native
} // namespace at 