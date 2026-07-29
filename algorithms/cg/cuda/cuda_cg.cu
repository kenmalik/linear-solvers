#include "cg/cuda.h"
#include "common/cuda_event_timer.h"
#include "common/log.h"

#include <cassert>
#include <cmath>

#include <nvtx3/nvtx3.hpp>

namespace cg::cuda {
namespace {
struct HandleStreamGuard {
    cusparseHandle_t cusparse;
    cublasHandle_t cublas;
    cudaStream_t cusparse_stream = nullptr;
    cudaStream_t cublas_stream = nullptr;

    HandleStreamGuard(cusparseHandle_t cusparse_handle,
                      cublasHandle_t cublas_handle,
                      cudaStream_t target_stream)
        : cusparse(cusparse_handle), cublas(cublas_handle) {
        CUSPARSE_CHECK(cusparseGetStream(cusparse, &cusparse_stream));
        CUBLAS_CHECK(cublasGetStream_v2(cublas, &cublas_stream));
        CUSPARSE_CHECK(cusparseSetStream(cusparse, target_stream));
        CUBLAS_CHECK(cublasSetStream_v2(cublas, target_stream));
    }

    HandleStreamGuard(const HandleStreamGuard &) = default;
    HandleStreamGuard(HandleStreamGuard &&) = delete;
    HandleStreamGuard &operator=(const HandleStreamGuard &) = default;
    HandleStreamGuard &operator=(HandleStreamGuard &&) = delete;

    ~HandleStreamGuard() {
        CUSPARSE_CHECK(cusparseSetStream(cusparse, cusparse_stream));
        CUBLAS_CHECK(cublasSetStream_v2(cublas, cublas_stream));
    }
};
} // namespace

int solve(cusparseHandle_t cusparse, cublasHandle_t cublas,
          cusparseSpMatDescr_t A, cusparseDnVecDescr_t b, // NOLINT
          cusparseDnVecDescr_t x, cusparseSpMatDescr_t L,
          Config config) {
    NVTX3_FUNC_RANGE();

    HandleStreamGuard handle_stream_guard(cusparse, cublas, config.stream);
    cils::detail::CudaTimerRange solve_range{cils::detail::g_event_timer, "solve", config.stream};

    std::int64_t n = 0;
    void *d_x = nullptr;
    cudaDataType_t cuda_type = CUDA_R_32F;
    CUSPARSE_CHECK(cusparseDnVecGet(x, &n, &d_x, &cuda_type));

    void *b_d_void = nullptr;
    CUSPARSE_CHECK(cusparseDnVecGetValues(b, &b_d_void));
    auto *d_b = static_cast<double *>(b_d_void);

    DeviceBuffers<double> d{n};

    // b_norm = sqrt(b' * b)
    double b_norm = 0;
    CUBLAS_CHECK(cublasDnrm2_v2_64(cublas, n, d_b, 1, &b_norm));

    // r = b - A * x
    std::size_t bufsize_residual_MV = 0;
    constexpr double alpha_residual_MV = -1.0;
    constexpr double beta_residual_MV = 1.0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_residual_MV, A, x,
        &beta_residual_MV, d.r, cuda_type, CUSPARSE_SPMV_ALG_DEFAULT,
        &bufsize_residual_MV));

    void *buffer_residual_MV = nullptr;
    CUDA_CHECK(cudaMalloc(&buffer_residual_MV, bufsize_residual_MV));

    if (config.real_residual) {
        CUSPARSE_CHECK(cusparseSpMV_preprocess(
            cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_residual_MV, A,
            x, &beta_residual_MV, d.r, cuda_type, CUSPARSE_SPMV_ALG_DEFAULT,
            buffer_residual_MV));
    }

    {
        cils::detail::CudaTimerRange er{cils::detail::g_event_timer, "r = b - A * x", config.stream};
        // Copy b into r
        CUBLAS_CHECK(cublasDcopy_v2_64(cublas, n, d_b, 1, d.d_r, 1));
        CUSPARSE_CHECK(cusparseSpMV(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha_residual_MV, A, x, &beta_residual_MV,
                                    d.r, cuda_type, CUSPARSE_SPMV_ALG_DEFAULT,
                                    buffer_residual_MV));
    }

    double residual_norm = 0;
    CUBLAS_CHECK(cublasDnrm2_v2_64(cublas, n, d.d_r, 1, &residual_norm));

    // d = L' (L \ r)
    // Solve M = L * L' which approximates A solve
    // Since SpSV supports in-place operations, we perform the solve like so:
    //   d = L \ r
    //   d = L' \ d
    void *buffer_SpSV_L = nullptr;
    void *buffer_SpSV_LT = nullptr;

    std::size_t bufsize_SpSV_L = 0;
    std::size_t bufsize_SpSV_LT = 0;

    cusparseSpSVDescr_t desc_SpSV_L = nullptr;
    cusparseSpSVDescr_t desc_SpSV_LT = nullptr;
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

    // d = M^{-1} * r
    {
        cils::detail::CudaTimerRange er{cils::detail::g_event_timer, "d = M^{-1} * r", config.stream};
        CUSPARSE_CHECK(cusparseSpSV_solve(
            cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_SpSM, L, d.r,
            d.d, cuda_type, CUSPARSE_SPSV_ALG_DEFAULT, desc_SpSV_L));
        CUSPARSE_CHECK(cusparseSpSV_solve(
            cusparse, CUSPARSE_OPERATION_TRANSPOSE, &alpha_SpSM, L, d.d, d.d,
            cuda_type, CUSPARSE_SPSV_ALG_DEFAULT, desc_SpSV_LT));
    }

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
    CUBLAS_CHECK(cublasDdot_v2_64(cublas, n, d.d_r, 1, d.d_d, 1, &delta_new));

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

    int iterations = 0;
    while (iterations < config.max_iterations && residual_norm > config.tolerance * b_norm) {
        nvtx3::scoped_range iteration_range{"iteration"};
        cils::detail::CudaTimerRange er{cils::detail::g_event_timer, "iteration", config.stream};

        cils::detail::log(residual_norm / b_norm);

        iterations += 1;

        // q = A * d
        {
            nvtx3::scoped_range r{"q = A * d"};
            cils::detail::CudaTimerRange er{cils::detail::g_event_timer, "q = A * d", config.stream};
            CUSPARSE_CHECK(
                cusparseSpMV(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             &alpha_MV_q, A, d.d, &beta_MV_q, d.q, cuda_type,
                             CUSPARSE_SPMV_ALG_DEFAULT, buffer_MV_q));
        }

        // alpha = delta_new / (d' * q)
        double alpha = NAN;
        {
            nvtx3::scoped_range r{"alpha = delta / d'q"};
            cils::detail::CudaTimerRange er{cils::detail::g_event_timer, "alpha = delta / d'q", config.stream};
            double d_dot_q = 0;
            CUBLAS_CHECK(
                cublasDdot_v2_64(cublas, n, d.d_d, 1, d.d_q, 1, &d_dot_q));
            assert(std::isfinite(d_dot_q));
            alpha = delta_new / d_dot_q;
            assert(std::isfinite(alpha));
        }

        // x = x + alpha * d
        {
            nvtx3::scoped_range r{"x = x + alpha * d"};
            cils::detail::CudaTimerRange er{cils::detail::g_event_timer, "x = x + alpha * d", config.stream};
            CUBLAS_CHECK(cublasDaxpy_v2_64(cublas, n, &alpha, d.d_d, 1,
                                           static_cast<double *>(d_x), 1));
        }

        if (config.real_residual) {
            // r = b - A * x
            nvtx3::scoped_range r{"r = b - A * x"};
            cils::detail::CudaTimerRange er{cils::detail::g_event_timer, "r = b - A * x", config.stream};
            CUBLAS_CHECK(cublasDcopy_v2_64(cublas, n, d_b, 1, d.d_r, 1));
            CUSPARSE_CHECK(cusparseSpMV(
                cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_residual_MV,
                A, x, &beta_residual_MV, d.r, cuda_type,
                CUSPARSE_SPMV_ALG_DEFAULT, buffer_residual_MV));
        } else {
            // r = r - alpha * q
            nvtx3::scoped_range r{"r = r - alpha * q"};
            cils::detail::CudaTimerRange er{cils::detail::g_event_timer, "r = r - alpha * q", config.stream};
            double neg_alpha = -alpha;
            CUBLAS_CHECK(
                cublasDaxpy_v2_64(cublas, n, &neg_alpha, d.d_q, 1, d.d_r, 1));
        }

        // residual_sq = r'r
        {
            nvtx3::scoped_range r{"residual_sq = r'r"};
            cils::detail::CudaTimerRange er{cils::detail::g_event_timer, "residual_sq = r'r", config.stream};
            CUBLAS_CHECK(
                cublasDnrm2_v2_64(cublas, n, d.d_r, 1, &residual_norm));
            assert(std::isfinite(residual_norm));
        }

        // s = M^{-1} * r
        {
            nvtx3::scoped_range r{"s = M^{-1} * r"};
            cils::detail::CudaTimerRange er{cils::detail::g_event_timer, "s = M^{-1} * r", config.stream};
            CUSPARSE_CHECK(cusparseSpSV_solve(
                cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_SpSM, L, d.r,
                d.s, cuda_type, CUSPARSE_SPSV_ALG_DEFAULT, desc_SpSV_L));
            CUSPARSE_CHECK(cusparseSpSV_solve(
                cusparse, CUSPARSE_OPERATION_TRANSPOSE, &alpha_SpSM, L, d.s,
                d.s, cuda_type, CUSPARSE_SPSV_ALG_DEFAULT, desc_SpSV_LT));
        }

        // beta = delta_new / delta_old
        double beta = NAN;
        {
            nvtx3::scoped_range r{"beta = delta_new / delta_old"};
            cils::detail::CudaTimerRange er{cils::detail::g_event_timer, "beta = delta_new / delta_old", config.stream};
            delta_old = delta_new;
            CUBLAS_CHECK(
                cublasDdot_v2_64(cublas, n, d.d_r, 1, d.d_s, 1, &delta_new));
            assert(std::isfinite(delta_new));
            assert(delta_new != 0);
            beta = delta_new / delta_old;
            assert(std::isfinite(beta));
        }

        // d = s + beta * d
        // s is no longer needed this iteration so we can overwrite it here
        {
            nvtx3::scoped_range r{"d = s + beta * d"};
            cils::detail::CudaTimerRange er{cils::detail::g_event_timer, "d = s + beta * d", config.stream};
            CUBLAS_CHECK(
                cublasDaxpy_v2_64(cublas, n, &beta, d.d_d, 1, d.d_s, 1));
            CUBLAS_CHECK(cublasDcopy_v2_64(cublas, n, d.d_s, 1, d.d_d, 1));
        }
    }

    CUDA_CHECK(cudaFree(buffer_MV_q));
    CUSPARSE_CHECK(cusparseSpSV_destroyDescr(desc_SpSV_LT));
    CUSPARSE_CHECK(cusparseSpSV_destroyDescr(desc_SpSV_L));
    CUDA_CHECK(cudaFree(buffer_SpSV_L));
    CUDA_CHECK(cudaFree(buffer_SpSV_LT));
    CUDA_CHECK(cudaFree(buffer_residual_MV));

    return iterations;
}
} // namespace cg::cuda
