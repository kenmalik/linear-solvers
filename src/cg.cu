#include <cg_run/cg.h>
#include <cg_run/checks.h>
#include <cstdint>
#include <iostream>
#include <type_traits>

namespace {
template <typename T> struct Device_buffers {
    cudaDataType_t cuda_type =
        std::is_same_v<T, float> ? CUDA_R_32F : CUDA_R_64F;

    Device_buffers(std::int64_t n) {
        cudaMalloc(&r_d, sizeof(T) * n);

        cusparseCreateDnVec(&r, n, r_d, cuda_type);
    }

    ~Device_buffers() {
        cudaFree(r_d);

        cusparseDestroyDnVec(r);
    }

    cusparseDnVecDescr_t r;
    T *r_d;
};
} // namespace

namespace cg_run {
// R is lower triangular incomplete Cholesky where A ~= M = R^TR
int cg(cusparseHandle_t cusparse, cublasHandle_t cublas, cusparseSpMatDescr_t A,
       cusparseDnVecDescr_t b, cusparseDnVecDescr_t x, cusparseSpMatDescr_t L,
       double tolerance, int max_iterations, bool real_residual) {
    std::int64_t n = 0;
    double *x_d = nullptr;
    cudaDataType_t cuda_type;
    cusparseDnVecGet(x, &n, reinterpret_cast<void **>(&x_d), &cuda_type);

    double *b_d = nullptr;
    cusparseDnVecGetValues(b, reinterpret_cast<void **>(&b_d));

    Device_buffers<double> d{n};

    // TODO: Implement setup
    // r = b - A * x
    std::size_t bufsize_residual_MV = 0;
    constexpr double alpha_residual_MV = -1.0;
    constexpr double beta_residual_MV = 1.0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_residual_MV, A, x,
        &beta_residual_MV, d.r, cuda_type, CUSPARSE_SPMV_ALG_DEFAULT,
        &bufsize_residual_MV));

    int iterations = 0;
    while (iterations < max_iterations) {
        iterations += 1;
        // TODO: Implement solver loop
    }

    return iterations;
}
} // namespace cg_run