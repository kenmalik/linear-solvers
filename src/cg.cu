#include "cg_run/cg.h"
#include "cg_run/checks.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <type_traits>

#include <nvtx3/nvtx3.hpp>

namespace {
template <typename T> struct Device_buffers {
    cudaDataType_t cuda_type =
        std::is_same_v<T, float> ? CUDA_R_32F : CUDA_R_64F;

    Device_buffers(std::int64_t n) noexcept {
        CUDA_CHECK(cudaMalloc(&r_d, sizeof(T) * n));
        CUDA_CHECK(cudaMalloc(&s_d, sizeof(T) * n));
        CUDA_CHECK(cudaMalloc(&d_d, sizeof(T) * n));
        CUDA_CHECK(cudaMalloc(&q_d, sizeof(T) * n));

        CUSPARSE_CHECK(cusparseCreateDnVec(&r, n, r_d, cuda_type));
        CUSPARSE_CHECK(cusparseCreateDnVec(&s, n, s_d, cuda_type));
        CUSPARSE_CHECK(cusparseCreateDnVec(&d, n, d_d, cuda_type));
        CUSPARSE_CHECK(cusparseCreateDnVec(&q, n, q_d, cuda_type));
    }

    ~Device_buffers() noexcept {
        CUDA_CHECK(cudaFree(r_d));
        CUDA_CHECK(cudaFree(s_d));
        CUDA_CHECK(cudaFree(d_d));
        CUDA_CHECK(cudaFree(q_d));

        CUSPARSE_CHECK(cusparseDestroyDnVec(r));
        CUSPARSE_CHECK(cusparseDestroyDnVec(s));
        CUSPARSE_CHECK(cusparseDestroyDnVec(d));
        CUSPARSE_CHECK(cusparseDestroyDnVec(q));
    }

    cusparseDnVecDescr_t r;
    cusparseDnVecDescr_t s;
    cusparseDnVecDescr_t d;
    cusparseDnVecDescr_t q;

    T *r_d;
    T *s_d;
    T *d_d;
    T *q_d;
};
} // namespace

namespace cg_run {
// L is lower triangular incomplete Cholesky where A ~= M = L * L'
int cg(cusparseHandle_t cusparse, cublasHandle_t cublas, cusparseSpMatDescr_t A,
       cusparseDnVecDescr_t b, cusparseDnVecDescr_t x, cusparseSpMatDescr_t L,
       double tolerance, int max_iterations, bool real_residual) {
    NVTX3_FUNC_RANGE();

    std::int64_t n = 0;
    void *x_d = nullptr;
    cudaDataType_t cuda_type;
    CUSPARSE_CHECK(cusparseDnVecGet(x, &n, &x_d, &cuda_type));

    void *b_d_void = nullptr;
    CUSPARSE_CHECK(cusparseDnVecGetValues(b, &b_d_void));
    double *b_d = static_cast<double *>(b_d_void);

    Device_buffers<double> d{n};

    // b_norm = sqrt(b' * b)
    double b_norm = 0;
    CUBLAS_CHECK(cublasDnrm2_v2_64(cublas, n, b_d, 1, &b_norm));

    // r = b - A * x
    // Copy b into r
    CUBLAS_CHECK(cublasDcopy_v2_64(cublas, n, b_d, 1, d.r_d, 1));

    std::size_t bufsize_residual_MV = 0;
    constexpr double alpha_residual_MV = -1.0;
    constexpr double beta_residual_MV = 1.0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_residual_MV, A, x,
        &beta_residual_MV, d.r, cuda_type, CUSPARSE_SPMV_ALG_DEFAULT,
        &bufsize_residual_MV));

    void *buffer_residual_MV = 0;
    CUDA_CHECK(cudaMalloc(&buffer_residual_MV, bufsize_residual_MV));

    if (real_residual) {
        CUSPARSE_CHECK(cusparseSpMV_preprocess(
            cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_residual_MV, A,
            x, &beta_residual_MV, d.r, cuda_type, CUSPARSE_SPMV_ALG_DEFAULT,
            buffer_residual_MV));
    }

    CUSPARSE_CHECK(cusparseSpMV(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha_residual_MV, A, x, &beta_residual_MV,
                                d.r, cuda_type, CUSPARSE_SPMV_ALG_DEFAULT,
                                buffer_residual_MV));

    double residual_norm = 0;
    CUBLAS_CHECK(cublasDnrm2_v2_64(cublas, n, d.r_d, 1, &residual_norm));

    // d = L' (L \ r)
    // Solve M = L * L' which approximates A solve
    // Since SpSV supports in-place operations, we perform the solve like so:
    //   d = L \ r
    //   d = L' \ d
    void *buffer_SpSV_L = nullptr;
    void *buffer_SpSV_LT = nullptr;

    std::size_t bufsize_SpSV_L = 0;
    std::size_t bufsize_SpSV_LT = 0;

    cusparseSpSVDescr_t desc_SpSV_L;
    cusparseSpSVDescr_t desc_SpSV_LT;
    CUSPARSE_CHECK(cusparseSpSV_createDescr(&desc_SpSV_L));
    CUSPARSE_CHECK(cusparseSpSV_createDescr(&desc_SpSV_LT));

    constexpr double alpha_SpSM = 1.0;

    // Compute buffer sizes for the initial d solves
    CUSPARSE_CHECK(cusparseSpSV_bufferSize(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_SpSM, L, d.r, d.d,
        cuda_type, CUSPARSE_SPSV_ALG_DEFAULT, desc_SpSV_L, &bufsize_SpSV_L));
    CUSPARSE_CHECK(cusparseSpSV_bufferSize(
        cusparse, CUSPARSE_OPERATION_TRANSPOSE, &alpha_SpSM, L, d.d, d.d,
        cuda_type, CUSPARSE_SPSV_ALG_DEFAULT, desc_SpSV_LT, &bufsize_SpSV_LT));

    // Compute buffer sizes needed for the s solves and take max for each buffer
    std::size_t bufsize_SpSV_L_s = 0;
    std::size_t bufsize_SpSV_LT_s = 0;
    CUSPARSE_CHECK(cusparseSpSV_bufferSize(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_SpSM, L, d.r, d.s,
        cuda_type, CUSPARSE_SPSV_ALG_DEFAULT, desc_SpSV_L, &bufsize_SpSV_L_s));
    CUSPARSE_CHECK(cusparseSpSV_bufferSize(
        cusparse, CUSPARSE_OPERATION_TRANSPOSE, &alpha_SpSM, L, d.s, d.s,
        cuda_type, CUSPARSE_SPSV_ALG_DEFAULT, desc_SpSV_LT,
        &bufsize_SpSV_LT_s));

    bufsize_SpSV_L = std::max(bufsize_SpSV_L, bufsize_SpSV_L_s);
    bufsize_SpSV_LT = std::max(bufsize_SpSV_LT, bufsize_SpSV_LT_s);

    CUDA_CHECK(cudaMalloc(&buffer_SpSV_L, bufsize_SpSV_L));
    CUDA_CHECK(cudaMalloc(&buffer_SpSV_LT, bufsize_SpSV_LT));

    // Analysis for initial d
    CUSPARSE_CHECK(cusparseSpSV_analysis(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_SpSM, L, d.r, d.d,
        cuda_type, CUSPARSE_SPSV_ALG_DEFAULT, desc_SpSV_L, buffer_SpSV_L));
    CUSPARSE_CHECK(cusparseSpSV_analysis(
        cusparse, CUSPARSE_OPERATION_TRANSPOSE, &alpha_SpSM, L, d.d, d.d,
        cuda_type, CUSPARSE_SPSV_ALG_DEFAULT, desc_SpSV_LT, buffer_SpSV_LT));

    // Solve
    CUSPARSE_CHECK(cusparseSpSV_solve(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_SpSM, L, d.r, d.d,
        cuda_type, CUSPARSE_SPSV_ALG_DEFAULT, desc_SpSV_L));
    CUSPARSE_CHECK(cusparseSpSV_solve(cusparse, CUSPARSE_OPERATION_TRANSPOSE,
                                      &alpha_SpSM, L, d.d, d.d, cuda_type,
                                      CUSPARSE_SPSV_ALG_DEFAULT, desc_SpSV_LT));

    // Analysis for s (reuse the dedicated buffers)
    CUSPARSE_CHECK(cusparseSpSV_analysis(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_SpSM, L, d.r, d.s,
        cuda_type, CUSPARSE_SPSV_ALG_DEFAULT, desc_SpSV_L, buffer_SpSV_L));
    CUSPARSE_CHECK(cusparseSpSV_analysis(
        cusparse, CUSPARSE_OPERATION_TRANSPOSE, &alpha_SpSM, L, d.s, d.s,
        cuda_type, CUSPARSE_SPSV_ALG_DEFAULT, desc_SpSV_LT, buffer_SpSV_LT));

    // delta_new = r' * d
    double delta_old = 0;
    double delta_new = 0;
    CUBLAS_CHECK(cublasDdot_v2_64(cublas, n, d.r_d, 1, d.d_d, 1, &delta_new));

    // q = A * d setup
    void *buffer_MV_q = nullptr;
    std::size_t bufsize_MV_q = 0;

    constexpr double alpha_MV_q = 1.0;
    constexpr double beta_MV_q = 0.0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_MV_q, A, d.d,
        &beta_MV_q, d.q, cuda_type, CUSPARSE_SPMV_ALG_DEFAULT, &bufsize_MV_q));

    CUDA_CHECK(cudaMalloc(&buffer_MV_q, bufsize_MV_q));

    CUSPARSE_CHECK(cusparseSpMV_preprocess(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_MV_q, A, d.d,
        &beta_MV_q, d.q, cuda_type, CUSPARSE_SPMV_ALG_DEFAULT, buffer_MV_q));

    std::cout << residual_norm / b_norm << std::endl;
    int iterations = 0;
    while (iterations < max_iterations && residual_norm > tolerance * b_norm) {
        nvtx3::scoped_range iteration_range("iteration");

        iterations += 1;

        // q = A * d
        CUSPARSE_CHECK(cusparseSpMV(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha_MV_q, A, d.d, &beta_MV_q, d.q,
                                    cuda_type, CUSPARSE_SPMV_ALG_DEFAULT,
                                    buffer_MV_q));

        // alpha = delta_new / (d' * q)
        double d_dot_q = 0;
        CUBLAS_CHECK(cublasDdot_v2_64(cublas, n, d.d_d, 1, d.q_d, 1, &d_dot_q));
        assert(std::isfinite(d_dot_q));
        double alpha = delta_new / d_dot_q;
        assert(std::isfinite(alpha));

        // x = x + alpha * d
        CUBLAS_CHECK(cublasDaxpy_v2_64(cublas, n, &alpha, d.d_d, 1,
                                       static_cast<double *>(x_d), 1));

        if (real_residual) {
            // r = b - A * x
            CUBLAS_CHECK(cublasDcopy_v2_64(cublas, n, b_d, 1, d.r_d, 1));
            CUSPARSE_CHECK(cusparseSpMV(
                cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_residual_MV,
                A, x, &beta_residual_MV, d.r, cuda_type,
                CUSPARSE_SPMV_ALG_DEFAULT, buffer_residual_MV));
        } else {
            // r = r - alpha * q
            double neg_alpha = -alpha;
            CUBLAS_CHECK(
                cublasDaxpy_v2_64(cublas, n, &neg_alpha, d.q_d, 1, d.r_d, 1));
        }

        // Update residual norm
        CUBLAS_CHECK(cublasDnrm2_v2_64(cublas, n, d.r_d, 1, &residual_norm));
        std::cout << residual_norm / b_norm << std::endl;
        assert(std::isfinite(residual_norm));

        // s = L' \ (L \ r)
        CUSPARSE_CHECK(cusparseSpSV_solve(
            cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_SpSM, L, d.r,
            d.s, cuda_type, CUSPARSE_SPSV_ALG_DEFAULT, desc_SpSV_L));
        CUSPARSE_CHECK(cusparseSpSV_solve(
            cusparse, CUSPARSE_OPERATION_TRANSPOSE, &alpha_SpSM, L, d.s, d.s,
            cuda_type, CUSPARSE_SPSV_ALG_DEFAULT, desc_SpSV_LT));

        // delta_new = r' * s
        delta_old = delta_new;
        CUBLAS_CHECK(
            cublasDdot_v2_64(cublas, n, d.r_d, 1, d.s_d, 1, &delta_new));
        assert(std::isfinite(delta_new));
        assert(delta_new != 0);

        // beta = delta_new / delta_old
        double beta = delta_new / delta_old;
        assert(std::isfinite(beta));

        // d = s + beta * d
        // s is no longer needed this iteration so we can overwrite it here
        CUBLAS_CHECK(cublasDaxpy_v2_64(cublas, n, &beta, d.d_d, 1, d.s_d, 1));
        CUBLAS_CHECK(cublasDcopy_v2_64(cublas, n, d.s_d, 1, d.d_d, 1));
    }

    CUDA_CHECK(cudaFree(buffer_MV_q));
    CUSPARSE_CHECK(cusparseSpSV_destroyDescr(desc_SpSV_LT));
    CUSPARSE_CHECK(cusparseSpSV_destroyDescr(desc_SpSV_L));
    CUDA_CHECK(cudaFree(buffer_SpSV_L));
    CUDA_CHECK(cudaFree(buffer_SpSV_LT));
    CUDA_CHECK(cudaFree(buffer_residual_MV));

    return iterations;
}
} // namespace cg_run