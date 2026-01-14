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
        cudaMalloc(&s_d, sizeof(T) * n);
        cudaMalloc(&d_d, sizeof(T) * n);
        cudaMalloc(&q_d, sizeof(T) * n);

        cusparseCreateDnVec(&r, n, r_d, cuda_type);
        cusparseCreateDnVec(&s, n, s_d, cuda_type);
        cusparseCreateDnVec(&d, n, d_d, cuda_type);
        cusparseCreateDnVec(&q, n, q_d, cuda_type);
    }

    ~Device_buffers() {
        cudaFree(r_d);
        cudaFree(s_d);
        cudaFree(d_d);
        cudaFree(q_d);

        cusparseDestroyDnVec(r);
        cusparseDestroyDnVec(s);
        cusparseDestroyDnVec(d);
        cusparseDestroyDnVec(q);
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
    std::int64_t n = 0;
    double *x_d = nullptr;
    cudaDataType_t cuda_type;
    cusparseDnVecGet(x, &n, reinterpret_cast<void **>(&x_d), &cuda_type);

    double *b_d = nullptr;
    cusparseDnVecGetValues(b, reinterpret_cast<void **>(&b_d));

    Device_buffers<double> d{n};

    // b_norm = sqrt(b' * b)
    double b_norm = 0;
    CUBLAS_CHECK(cublasDnrm2_v2(cublas, n, b_d, 1, &b_norm));

    // TODO: Check _64 versions of cublas functions
    // TODO: Implement setup
    // r = b - A * x
    // Copy b into r
    CUBLAS_CHECK(cublasDcopy_v2(cublas, n, b_d, 1, d.r_d, 1));

    std::size_t bufsize_residual_MV = 0;
    constexpr double alpha_residual_MV = -1.0;
    constexpr double beta_residual_MV = 1.0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_residual_MV, A, x,
        &beta_residual_MV, d.r, cuda_type, CUSPARSE_SPMV_ALG_DEFAULT,
        &bufsize_residual_MV));

    void *buffer_residual_MV = 0;
    CUDA_CHECK(cudaMalloc(&buffer_residual_MV, bufsize_residual_MV));

    // TODO: Preprocess if using real residual since it'll be used again in loop

    CUSPARSE_CHECK(cusparseSpMV(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha_residual_MV, A, x, &beta_residual_MV,
                                d.r, cuda_type, CUSPARSE_SPMV_ALG_DEFAULT,
                                buffer_residual_MV));

    double residual_norm = 0;
    CUBLAS_CHECK(cublasDnrm2_v2(cublas, n, d.r_d, 1, &residual_norm));

    // d = L' (L \ r)
    // Solve M = L * L' which approximates A solve
    // Since SpSV supports in-place operations, we perform the solve like so:
    //   d = L \ r
    //   d = L' \ d
    void *buffer_SpSV = nullptr;

    std::size_t bufsize_SpSV_L = 0;
    std::size_t bufsize_SpSV_LT = 0;

    cusparseSpSVDescr_t desc_SpSV_L;
    cusparseSpSVDescr_t desc_SpSV_LT;
    CUSPARSE_CHECK(cusparseSpSV_createDescr(&desc_SpSV_L));
    CUSPARSE_CHECK(cusparseSpSV_createDescr(&desc_SpSV_LT));

    constexpr double alpha_SpSM = 1.0;

    // Reuse larger of two buffers
    CUSPARSE_CHECK(cusparseSpSV_bufferSize(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_SpSM, L, d.r, d.d,
        cuda_type, CUSPARSE_SPSV_ALG_DEFAULT, desc_SpSV_L, &bufsize_SpSV_L));
    CUSPARSE_CHECK(cusparseSpSV_bufferSize(
        cusparse, CUSPARSE_OPERATION_TRANSPOSE, &alpha_SpSM, L, d.d, d.d,
        cuda_type, CUSPARSE_SPSV_ALG_DEFAULT, desc_SpSV_LT, &bufsize_SpSV_LT));
    const std::size_t bufsize_SpSV_d =
        std::max(bufsize_SpSV_L, bufsize_SpSV_LT);
    CUDA_CHECK(cudaMalloc(&buffer_SpSV, bufsize_SpSV_d));

    // Analysis
    CUSPARSE_CHECK(cusparseSpSV_analysis(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_SpSM, L, d.r, d.d,
        cuda_type, CUSPARSE_SPSV_ALG_DEFAULT, desc_SpSV_L, buffer_SpSV));
    CUSPARSE_CHECK(cusparseSpSV_analysis(
        cusparse, CUSPARSE_OPERATION_TRANSPOSE, &alpha_SpSM, L, d.d, d.d,
        cuda_type, CUSPARSE_SPSV_ALG_DEFAULT, desc_SpSV_LT, buffer_SpSV));

    // Solve
    CUSPARSE_CHECK(cusparseSpSV_solve(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_SpSM, L, d.r, d.d,
        cuda_type, CUSPARSE_SPSV_ALG_DEFAULT, desc_SpSV_L));
    CUSPARSE_CHECK(cusparseSpSV_solve(cusparse, CUSPARSE_OPERATION_TRANSPOSE,
                                      &alpha_SpSM, L, d.d, d.d, cuda_type,
                                      CUSPARSE_SPSV_ALG_DEFAULT, desc_SpSV_LT));

    // Setup for s = L' \ (L \ r)
    CUSPARSE_CHECK(cusparseSpSV_bufferSize(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_SpSM, L, d.r, d.s,
        cuda_type, CUSPARSE_SPSV_ALG_DEFAULT, desc_SpSV_L, &bufsize_SpSV_L));
    CUSPARSE_CHECK(cusparseSpSV_bufferSize(
        cusparse, CUSPARSE_OPERATION_TRANSPOSE, &alpha_SpSM, L, d.s, d.s,
        cuda_type, CUSPARSE_SPSV_ALG_DEFAULT, desc_SpSV_LT, &bufsize_SpSV_LT));
    const std::size_t bufsize_SpSV_s =
        std::max(bufsize_SpSV_L, bufsize_SpSV_LT);

    if (bufsize_SpSV_s > bufsize_SpSV_d) {
        CUDA_CHECK(cudaFree(buffer_SpSV));
        CUDA_CHECK(cudaMalloc(&buffer_SpSV, bufsize_SpSV_s));
    }

    CUSPARSE_CHECK(cusparseSpSV_analysis(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_SpSM, L, d.r, d.s,
        cuda_type, CUSPARSE_SPSV_ALG_DEFAULT, desc_SpSV_L, buffer_SpSV));
    CUSPARSE_CHECK(cusparseSpSV_analysis(
        cusparse, CUSPARSE_OPERATION_TRANSPOSE, &alpha_SpSM, L, d.s, d.s,
        cuda_type, CUSPARSE_SPSV_ALG_DEFAULT, desc_SpSV_LT, buffer_SpSV));

    // delta_new = r' * d
    double delta_old = 0;
    double delta_new = 0;
    CUBLAS_CHECK(cublasDdot_v2(cublas, n, d.r_d, 1, d.d_d, 1, &delta_new));

    // q = A * d setup
    void *buffer_MV_q = nullptr;
    std::size_t bufsize_MV_q = 0;

    constexpr double alpha_MV_q = 1.0;
    constexpr double beta_MV_q = 0.0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_MV_q, A, d.d,
        &beta_MV_q, d.q, cuda_type, CUSPARSE_SPMV_ALG_DEFAULT, &bufsize_MV_q));

    CUDA_CHECK(cudaMalloc(&buffer_MV_q, bufsize_MV_q));

    // TODO: Preprocess MV_q since it'll be in the loop

    int iterations = 0;
    while (iterations < max_iterations && residual_norm > tolerance * b_norm) {
        iterations += 1;

        // TODO: Implement solver loop
        // q = A * d
        CUSPARSE_CHECK(cusparseSpMV(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha_MV_q, A, d.d, &beta_MV_q, d.q,
                                    cuda_type, CUSPARSE_SPMV_ALG_DEFAULT,
                                    buffer_MV_q));

        double d_dot_q = 0;
        CUBLAS_CHECK(cublasDdot_v2(cublas, n, d.d_d, 1, d.q_d, 1, &d_dot_q));

        double alpha = delta_new / d_dot_q;
        CUBLAS_CHECK(cublasDaxpy(cublas, n, &alpha, d.d_d, 1, x_d, 1));

        if (real_residual) {
            // r = b - A * x
            CUBLAS_CHECK(cublasDcopy_v2(cublas, n, b_d, 1, d.r_d, 1));
            CUSPARSE_CHECK(cusparseSpMV(
                cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_residual_MV,
                A, x, &beta_residual_MV, d.r, cuda_type,
                CUSPARSE_SPMV_ALG_DEFAULT, buffer_residual_MV));
        } else {
            // r = r - alpha * q
            double neg_alpha = -alpha;
            CUBLAS_CHECK(
                cublasDaxpy(cublas, n, &neg_alpha, d.q_d, 1, d.r_d, 1));
        }

        // Update residual norm
        CUBLAS_CHECK(cublasDnrm2_v2(cublas, n, d.r_d, 1, &residual_norm));

        // s = L' \ (L \ r)
        CUSPARSE_CHECK(cusparseSpSV_solve(
            cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_SpSM, L, d.r,
            d.s, cuda_type, CUSPARSE_SPSV_ALG_DEFAULT, desc_SpSV_L));
        CUSPARSE_CHECK(cusparseSpSV_solve(
            cusparse, CUSPARSE_OPERATION_TRANSPOSE, &alpha_SpSM, L, d.s, d.s,
            cuda_type, CUSPARSE_SPSV_ALG_DEFAULT, desc_SpSV_LT));

        // delta_new = r' * s
        delta_old = delta_new;
        CUBLAS_CHECK(cublasDdot_v2(cublas, n, d.r_d, 1, d.s_d, 1, &delta_new));

        // beta = delta_new / delta_old
        double beta = delta_new / delta_old;

        // d = s + beta * d
        // s is no longer needed this iteration so we can overwrite it here
        CUBLAS_CHECK(cublasDaxpy_v2(cublas, n, &beta, x_d, 1, d.s_d, 1));
        CUBLAS_CHECK(cublasDcopy_v2(cublas, n, d.s_d, 1, d.d_d, 1));
    }

    CUDA_CHECK(cudaFree(buffer_MV_q));
    CUSPARSE_CHECK(cusparseSpSV_destroyDescr(desc_SpSV_LT));
    CUSPARSE_CHECK(cusparseSpSV_destroyDescr(desc_SpSV_L));
    CUDA_CHECK(cudaFree(buffer_SpSV));
    CUDA_CHECK(cudaFree(buffer_residual_MV));

    return iterations;
}
} // namespace cg_run