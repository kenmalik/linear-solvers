#include "dr_bcg/cuda.h"
#include "dr_bcg/internal/device_buffer.h"
#include "dr_bcg/internal/math.h"

#include "common/cuda_checks.h"
#include "common/cuda_event_timer.h"
#include "common/log.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>

#include <nvtx3/nvtx3.hpp>

namespace {

std::pair<std::int64_t, std::int64_t> get_size(cusparseDnMatDescr_t mat) {
    std::int64_t n = 0;
    std::int64_t s = 0;
    std::int64_t ld = 0;
    void *vals = nullptr;
    cudaDataType_t data_type;
    cusparseOrder_t order;

    CUSPARSE_CHECK(
        cusparseDnMatGet(mat, &n, &s, &ld, &vals, &data_type, &order));

    return {n, s};
}

} // namespace

namespace dr_bcg::cuda {

Handles::Handles() {
    CUSPARSE_CHECK(cusparseCreate(&cusparse));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver));
    CUSOLVER_CHECK(cusolverDnCreateParams(&cusolver_params));
    CUBLAS_CHECK(cublasCreate_v2(&cublas));
}

Handles::~Handles() {
    CUSPARSE_CHECK(cusparseDestroy(cusparse));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver));
    CUSOLVER_CHECK(cusolverDnDestroyParams(cusolver_params));
    CUBLAS_CHECK(cublasDestroy_v2(cublas));
};

void Handles::set_stream(cudaStream_t stream) {
    CUSPARSE_CHECK(cusparseSetStream(cusparse, stream));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolver, stream));
    CUBLAS_CHECK(cublasSetStream_v2(cublas, stream));
}

int solve(Handles &handles, cusparseSpMatDescr_t A, cusparseDnMatDescr_t X,
          cusparseDnMatDescr_t B, double tolerance, int max_iterations,
          cudaStream_t stream) {
    NVTX3_FUNC_RANGE();

    auto [n, s] = get_size(B);
    Device_buffer<double> d(n, s);

    CudaTimerRange solve_range{g_event_timer, "solve", stream};

    handles.set_stream(stream);

    QrWorkspace<double> qr_ws;
    qr_ws.allocate(handles.cusolver, handles.cusolver_params,
                   static_cast<int>(n), static_cast<int>(s));
    LuWorkspace<double> lu_ws;
    lu_ws.allocate(handles.cusolver, handles.cusolver_params,
                   static_cast<int>(s));

    void *scratch_d = nullptr;

    cusparseDnMatDescr_t temp;
    CUSPARSE_CHECK(cusparseCreateDnMat(&temp, n, s, n, d.temp, CUDA_R_64F,
                                       CUSPARSE_ORDER_COL));

    double *d_X = nullptr;
    CUSPARSE_CHECK(cusparseDnMatGetValues(X, reinterpret_cast<void **>(&d_X)));

    // Precalculate B1 norm for conversion checks
    double *d_B = nullptr;
    CUSPARSE_CHECK(cusparseDnMatGetValues(B, reinterpret_cast<void **>(&d_B)));

    double *d_norm = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_norm, sizeof(double), stream));
    CUBLAS_CHECK(
        cublasSetPointerMode(handles.cublas, CUBLAS_POINTER_MODE_DEVICE));

    constexpr int incx = 1;
    CUBLAS_CHECK(cublasDnrm2_v2(handles.cublas, n, d_B, incx, d_norm));
    double B1_norm = 0;
    CUDA_CHECK(cudaMemcpyAsync(&B1_norm, d_norm, sizeof(double),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    double *d_R = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_R, sizeof(double) * n * s, stream));
    cusparseDnMatDescr_t R;
    CUSPARSE_CHECK(
        cusparseCreateDnMat(&R, n, s, n, d_R, CUDA_R_64F, CUSPARSE_ORDER_COL));

    {
        nvtx3::scoped_range R_range{"R = B - A * X"};
        CudaTimerRange er(g_event_timer, "R = B - A * X", stream);

        // R = B - A * X
        std::size_t buffer_size;
        constexpr double alpha = -1.0;
        constexpr double beta = 1.0;
        constexpr cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
        constexpr cudaDataType_t compute_type = CUDA_R_64F;
        constexpr cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT;

        void *d_B_ptr = nullptr;
        CUSPARSE_CHECK(cusparseDnMatGetValues(B, &d_B_ptr));
        CUDA_CHECK(cudaMemcpyAsync(d_R, d_B_ptr, sizeof(double) * n * s,
                                   cudaMemcpyDeviceToDevice, stream));

        CUSPARSE_CHECK(cusparseSpMM_bufferSize(handles.cusparse, op, op, &alpha,
                                               A, X, &beta, B, compute_type,
                                               alg, &buffer_size));

        CUDA_CHECK(cudaMallocAsync(&scratch_d, buffer_size, stream));

        CUSPARSE_CHECK(cusparseSpMM(handles.cusparse, op, op, &alpha, A, X,
                                    &beta, R, compute_type, alg, scratch_d));

        CUDA_CHECK(cudaFreeAsync(scratch_d, stream));
    }

    {
        nvtx3::scoped_range w_sigma_initial_range{"[w sigma] = QR(L^-1 * R)"};
        CudaTimerRange er(g_event_timer, "[w sigma] = QR(L^-1 * R)", stream);

        // [w, sigma] = qr(R, 'econ')
        qr_factorization(handles.cusolver, handles.cusolver_params, d.w,
                         d.sigma, n, s, d_R, qr_ws, stream);
    }

    CUDA_CHECK(cudaFreeAsync(d_R, stream));
    CUSPARSE_CHECK(cusparseDestroyDnMat(R));

    {
        nvtx3::scoped_range s_initial_range{"s = (L^-1)' * w"};
        CudaTimerRange er(g_event_timer, "s = (L^-1)' * w", stream);

        // s = w
        CUDA_CHECK(cudaMemcpyAsync(d.s, d.w, sizeof(double) * n * s,
                                   cudaMemcpyDeviceToDevice, stream));
    }

    cusparseDnMatDescr_t s_desc;
    CUSPARSE_CHECK(cusparseCreateDnMat(&s_desc, n, s, n, d.s, CUDA_R_64F,
                                       CUSPARSE_ORDER_COL));

    cusparseDnMatDescr_t w_desc;
    CUSPARSE_CHECK(cusparseCreateDnMat(&w_desc, n, s, n, d.w, CUDA_R_64F,
                                       CUSPARSE_ORDER_COL));

    cusparseDnVecDescr_t temp1;
    CUSPARSE_CHECK(cusparseCreateDnVec(&temp1, n, d.temp, CUDA_R_64F));

    cusparseDnVecDescr_t X1;
    CUSPARSE_CHECK(cusparseCreateDnVec(&X1, n, d_X, CUDA_R_64F));

    {
        constexpr double alpha_pos = 1.0;
        constexpr double beta_zero = 0.0;
        constexpr double alpha_neg = -1.0;
        constexpr double beta_pos = 1.0;
        constexpr cusparseOperation_t op_nt = CUSPARSE_OPERATION_NON_TRANSPOSE;
        std::size_t buf_xi = 0;
        std::size_t buf_rrn = 0;
        std::size_t buf_w_zeta = 0;
        CUSPARSE_CHECK(cusparseSpMM_bufferSize(
            handles.cusparse, op_nt, op_nt, &alpha_pos, A, s_desc, &beta_zero,
            temp, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &buf_xi));
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(
            handles.cusparse, op_nt, &alpha_neg, A, X1, &beta_pos, temp1,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_rrn));
        CUSPARSE_CHECK(cusparseSpMM_bufferSize(
            handles.cusparse, op_nt, op_nt, &alpha_neg, A, temp, &beta_pos,
            w_desc, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &buf_w_zeta));
        std::size_t scratch_size = std::max({buf_xi, buf_rrn, buf_w_zeta});
        if (scratch_size > 0) {
            CUDA_CHECK(cudaMallocAsync(&scratch_d, scratch_size, stream));
        }
    }

    int iterations = 0;
    while (iterations < max_iterations) {
        nvtx3::scoped_range iteration_range{"iteration"};
        CudaTimerRange iteration_event_range(g_event_timer, "iteration",
                                             stream);

        ++iterations;

        {
            nvtx3::scoped_range xi_range{"xi = (s' * As)^-1"};
            CudaTimerRange er(g_event_timer, "xi = (s' * As)^-1", stream);

            // xi = (s' * A * s)^-1
            constexpr double alpha = 1.0;
            constexpr double beta = 0.0;
            constexpr cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
            constexpr cudaDataType_t compute_type = CUDA_R_64F;
            constexpr cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT;

            CUSPARSE_CHECK(cusparseSpMM(handles.cusparse, op, op, &alpha, A,
                                        s_desc, &beta, temp, compute_type, alg,
                                        scratch_d));

            constexpr cublasOperation_t op_t = CUBLAS_OP_T;
            constexpr cublasOperation_t op_n = CUBLAS_OP_N;
            CUBLAS_CHECK(cublasDgemm_v2(handles.cublas, op_t, op_n, s, s, n,
                                        d.d_one, d.s, n, d.temp, n, d.d_zero,
                                        d.xi, s));

            invert_square_matrix(handles.cusolver, handles.cusolver_params,
                                 d.xi, s, lu_ws, stream);
        }

        {
            nvtx3::scoped_range X_range{"X = X + s * xi * sigma"};
            CudaTimerRange er(g_event_timer, "X = X + s * xi * sigma", stream);

            // X = X + s * xi * sigma
            CUBLAS_CHECK(cublasDgemm_v2(handles.cublas, CUBLAS_OP_N,
                                        CUBLAS_OP_N, s, s, s, d.d_one, d.xi, s,
                                        d.sigma, s, d.d_zero, d.temp, n));

            CUBLAS_CHECK(cublasDgemm_v2(handles.cublas, CUBLAS_OP_N,
                                        CUBLAS_OP_N, n, s, s, d.d_one, d.s, n,
                                        d.temp, n, d.d_one, d_X, n));
        }

        double relative_residual_norm = 0;
        {
            nvtx3::scoped_range rrn_range{"norm(B1 - A * X1) / norm(B1)"};
            CudaTimerRange er(g_event_timer, "norm(B1 - A * X1) / norm(B1)",
                              stream);

            // norm(B(:,1) - A * X(:,1)) / norm(B(:,1))
            constexpr cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
            constexpr double alpha = -1.0;
            constexpr double beta = 1.0;
            constexpr cudaDataType_t compute_type = CUDA_R_64F;
            constexpr cusparseSpMVAlg_t alg = CUSPARSE_SPMV_ALG_DEFAULT;

            CUDA_CHECK(cudaMemcpyAsync(d.temp, d_B, sizeof(double) * n,
                                       cudaMemcpyDeviceToDevice, stream));

            CUSPARSE_CHECK(cusparseSpMV(handles.cusparse, op, &alpha, A, X1,
                                        &beta, temp1, compute_type, alg,
                                        scratch_d));

            CUBLAS_CHECK(
                cublasDnrm2_v2(handles.cublas, n, d.temp, incx, d_norm));
            double residual_norm = 0;
            CUDA_CHECK(cudaMemcpyAsync(&residual_norm, d_norm, sizeof(double),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            if (*qr_ws.h_info < 0)
                throw std::runtime_error(std::to_string(-*qr_ws.h_info) +
                                         "-th parameter is wrong in QR\n");
            if (*lu_ws.h_info < 0)
                throw std::runtime_error(std::to_string(-*lu_ws.h_info) +
                                         "-th parameter is wrong in LU\n");

            relative_residual_norm = residual_norm / B1_norm;
            LOG_TRACE(relative_residual_norm);
        }

        if (relative_residual_norm < tolerance) {
            break;
        }

        {
            nvtx3::scoped_range w_zeta_range{
                "[w zeta] = QR(w - L^{-1} * A * s * xi)"};
            CudaTimerRange er(g_event_timer,
                              "[w zeta] = QR(w - L^{-1} * A * s * xi)", stream);

            // [w, zeta] = qr(w - A * s * xi, 'econ')
            constexpr cublasOperation_t op = CUBLAS_OP_N;
            CUBLAS_CHECK(cublasDgemm_v2(handles.cublas, op, op, n, s, s,
                                        d.d_one, d.s, n, d.xi, s,
                                        d.d_zero, d.temp, n));

            constexpr cusparseOperation_t spmm_op =
                CUSPARSE_OPERATION_NON_TRANSPOSE;
            constexpr double spmm_alpha = -1.0;
            constexpr double spmm_beta = 1.0;
            constexpr cudaDataType_t compute_type = CUDA_R_64F;
            constexpr cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT;

            CUSPARSE_CHECK(cusparseSpMM(handles.cusparse, spmm_op, spmm_op,
                                        &spmm_alpha, A, temp, &spmm_beta,
                                        w_desc, compute_type, alg, scratch_d));

            qr_factorization(handles.cusolver, handles.cusolver_params, d.w,
                             d.zeta, n, s, d.w, qr_ws, stream);
        }

        {
            nvtx3::scoped_range s_range{"s = (L^-1)' * w + s * zeta'"};
            CudaTimerRange er(g_event_timer, "s = (L^-1)' * w + s * zeta'",
                              stream);

            // s = w + s * zeta'
            constexpr cublasSideMode_t side = CUBLAS_SIDE_RIGHT;
            constexpr cublasFillMode_t fill_mode = CUBLAS_FILL_MODE_UPPER;
            constexpr cublasDiagType_t diag_type = CUBLAS_DIAG_NON_UNIT;
            constexpr cublasOperation_t op_zeta = CUBLAS_OP_T;

            CUBLAS_CHECK(cublasDtrmm_v2(handles.cublas, side, fill_mode,
                                        op_zeta, diag_type, n, s, d.d_one,
                                        d.zeta, s, d.s, n, d.s, n));

            constexpr cublasOperation_t sgeam_op = CUBLAS_OP_N;
            CUBLAS_CHECK(cublasDgeam(handles.cublas, sgeam_op, sgeam_op, n, s,
                                     d.d_one, d.s, n, d.d_one, d.w, n,
                                     d.s, n));
        }

        {
            nvtx3::scoped_range sigma_range{"sigma = zeta * sigma"};
            CudaTimerRange er(g_event_timer, "sigma = zeta * sigma", stream);

            // sigma = zeta * sigma
            constexpr cublasSideMode_t side = CUBLAS_SIDE_LEFT;
            constexpr cublasFillMode_t fill_mode = CUBLAS_FILL_MODE_UPPER;
            constexpr cublasDiagType_t diag_type = CUBLAS_DIAG_NON_UNIT;
            constexpr cublasOperation_t op_zeta = CUBLAS_OP_N;

            CUBLAS_CHECK(cublasDtrmm_v2(handles.cublas, side, fill_mode,
                                        op_zeta, diag_type, s, s, d.d_one,
                                        d.zeta, s, d.sigma, s, d.sigma, s));
        }
    }

    CUDA_CHECK(cudaFreeAsync(d_norm, stream));
    CUDA_CHECK(cudaFreeAsync(scratch_d, stream));
    CUSPARSE_CHECK(cusparseDestroyDnMat(s_desc));
    CUSPARSE_CHECK(cusparseDestroyDnMat(w_desc));
    CUSPARSE_CHECK(cusparseDestroyDnVec(temp1));
    CUSPARSE_CHECK(cusparseDestroyDnVec(X1));

    return iterations;
}

// Preconditioned double-precision variant
int solve(Handles &handles, cusparseSpMatDescr_t A, cusparseDnMatDescr_t X,
          cusparseDnMatDescr_t B, cusparseSpMatDescr_t L, double tolerance,
          int max_iterations, cudaStream_t stream) {
    NVTX3_FUNC_RANGE();

    auto [n, s] = get_size(B);
    Device_buffer<double> d(n, s);

    CudaTimerRange solve_range{g_event_timer, "solve", stream};

    handles.set_stream(stream);

    QrWorkspace<double> qr_ws;
    qr_ws.allocate(handles.cusolver, handles.cusolver_params,
                   static_cast<int>(n), static_cast<int>(s));
    LuWorkspace<double> lu_ws;
    lu_ws.allocate(handles.cusolver, handles.cusolver_params,
                   static_cast<int>(s));

    void *scratch_d = nullptr;

    cusparseDnMatDescr_t temp;
    CUSPARSE_CHECK(cusparseCreateDnMat(&temp, n, s, n, d.temp, CUDA_R_64F, CUSPARSE_ORDER_COL));
    cusparseDnMatDescr_t s_desc;
    CUSPARSE_CHECK(cusparseCreateDnMat(&s_desc, n, s, n, d.s, CUDA_R_64F,
                                       CUSPARSE_ORDER_COL));
    cusparseDnMatDescr_t w_desc;
    CUSPARSE_CHECK(cusparseCreateDnMat(&w_desc, n, s, n, d.w, CUDA_R_64F,
                                       CUSPARSE_ORDER_COL));

    SpsmCache<double> spsm_nt;
    spsm_nt.analyze(handles.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, L,
                    w_desc, temp);

    SpsmCache<double> spsm_t;
    spsm_t.analyze(handles.cusparse, CUSPARSE_OPERATION_TRANSPOSE, L, w_desc,
                   temp);

    double *d_X = nullptr;
    CUSPARSE_CHECK(cusparseDnMatGetValues(X, reinterpret_cast<void **>(&d_X)));

    // Precalculate B1 norm for conversion checks
    double *d_B = nullptr;
    CUSPARSE_CHECK(cusparseDnMatGetValues(B, reinterpret_cast<void **>(&d_B)));

    double *d_norm = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_norm, sizeof(double), stream));
    CUBLAS_CHECK(
        cublasSetPointerMode(handles.cublas, CUBLAS_POINTER_MODE_DEVICE));

    constexpr int incx = 1;
    CUBLAS_CHECK(cublasDnrm2_v2(handles.cublas, n, d_B, incx, d_norm));
    double B1_norm = 0;
    CUDA_CHECK(cudaMemcpyAsync(&B1_norm, d_norm, sizeof(double),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    double *d_R = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_R, sizeof(double) * n * s, stream));
    cusparseDnMatDescr_t R;
    CUSPARSE_CHECK(
        cusparseCreateDnMat(&R, n, s, n, d_R, CUDA_R_64F, CUSPARSE_ORDER_COL));

    {
        nvtx3::scoped_range R_range{"R = B - A * X"};
        CudaTimerRange er(g_event_timer, "R = B - A * X", stream);

        // R = B - A * X
        std::size_t buffer_size;
        constexpr double alpha = -1.0;
        constexpr double beta = 1.0;
        constexpr cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
        constexpr cudaDataType_t compute_type = CUDA_R_64F;
        constexpr cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT;

        void *d_B_ptr = nullptr;
        CUSPARSE_CHECK(cusparseDnMatGetValues(B, &d_B_ptr));
        CUDA_CHECK(cudaMemcpyAsync(d_R, d_B_ptr, sizeof(double) * n * s,
                                   cudaMemcpyDeviceToDevice, stream));

        CUSPARSE_CHECK(cusparseSpMM_bufferSize(handles.cusparse, op, op, &alpha,
                                               A, X, &beta, B, compute_type,
                                               alg, &buffer_size));

        CUDA_CHECK(cudaMallocAsync(&scratch_d, buffer_size, stream));

        CUSPARSE_CHECK(cusparseSpMM(handles.cusparse, op, op, &alpha, A, X,
                                    &beta, R, compute_type, alg, scratch_d));

        CUDA_CHECK(cudaFreeAsync(scratch_d, stream));
    }

    // We break [w sigma] = QR(L^-1 * R) into two steps for timing purposes:
    // 1. temp = L^-1 * R
    // 2. [w sigma] = QR(temp)
    {
        nvtx3::scoped_range w_sigma_initial_range{"temp = L^-1 * R"};
        CudaTimerRange er(g_event_timer, "temp = L^-1 * R", stream);

        sptri_solve<double>(handles.cusparse, temp,
                            CUSPARSE_OPERATION_NON_TRANSPOSE, L, R, spsm_nt);
    }

    {
        nvtx3::scoped_range w_sigma_initial_range{"[w sigma] = QR(temp)"};
        CudaTimerRange er(g_event_timer, "[w sigma] = QR(temp)", stream);

        qr_factorization(handles.cusolver, handles.cusolver_params, d.w,
                         d.sigma, n, s, d.temp, qr_ws, stream);
    }

    CUDA_CHECK(cudaFreeAsync(d_R, stream));
    CUSPARSE_CHECK(cusparseDestroyDnMat(R));

    {
        nvtx3::scoped_range s_initial_range{"s = (L^-1)' * w"};
        CudaTimerRange er(g_event_timer, "s = (L^-1)' * w", stream);

        // s = (L^-1)' * w
        CUDA_CHECK(cudaMemcpyAsync(d.s, d.w, sizeof(double) * n * s,
                                   cudaMemcpyDeviceToDevice, stream));

        sptri_solve<double>(handles.cusparse, s_desc,
                            CUSPARSE_OPERATION_TRANSPOSE, L, w_desc, spsm_t);
    }

    cusparseDnVecDescr_t temp1;
    CUSPARSE_CHECK(cusparseCreateDnVec(&temp1, n, d.temp, CUDA_R_64F));

    cusparseDnVecDescr_t X1;
    CUSPARSE_CHECK(cusparseCreateDnVec(&X1, n, d_X, CUDA_R_64F));

    {
        constexpr double alpha_pos = 1.0;
        constexpr double beta_zero = 0.0;
        constexpr double alpha_neg = -1.0;
        constexpr double beta_pos = 1.0;
        constexpr cusparseOperation_t op_nt = CUSPARSE_OPERATION_NON_TRANSPOSE;
        std::size_t buf_xi = 0;
        std::size_t buf_rrn = 0;
        CUSPARSE_CHECK(cusparseSpMM_bufferSize(
            handles.cusparse, op_nt, op_nt, &alpha_pos, A, s_desc, &beta_zero,
            temp, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &buf_xi));
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(
            handles.cusparse, op_nt, &alpha_neg, A, X1, &beta_pos, temp1,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_rrn));
        std::size_t scratch_size = std::max(buf_xi, buf_rrn);
        if (scratch_size > 0) {
            CUDA_CHECK(cudaMallocAsync(&scratch_d, scratch_size, stream));
        }
    }

    int iterations = 0;
    while (iterations < max_iterations) {
        nvtx3::scoped_range iteration_range{"iteration"};
        CudaTimerRange iteration_event_range(g_event_timer, "iteration",
                                             stream);

        ++iterations;

        {
            nvtx3::scoped_range xi_range{"xi = (s' * As)^-1"};
            CudaTimerRange er(g_event_timer, "xi = (s' * As)^-1", stream);

            // xi = (s' * A * s)^-1
            constexpr double alpha = 1.0;
            constexpr double beta = 0.0;
            constexpr cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
            constexpr cudaDataType_t compute_type = CUDA_R_64F;
            constexpr cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT;

            CUSPARSE_CHECK(cusparseSpMM(handles.cusparse, op, op, &alpha, A,
                                        s_desc, &beta, temp, compute_type, alg,
                                        scratch_d));

            constexpr cublasOperation_t op_t = CUBLAS_OP_T;
            constexpr cublasOperation_t op_n = CUBLAS_OP_N;
            CUBLAS_CHECK(cublasDgemm_v2(handles.cublas, op_t, op_n, s, s, n,
                                        d.d_one, d.s, n, d.temp, n, d.d_zero,
                                        d.xi, s));

            invert_square_matrix(handles.cusolver, handles.cusolver_params,
                                 d.xi, s, lu_ws, stream);
        }

        {
            nvtx3::scoped_range X_range{"X = X + s * xi * sigma"};
            CudaTimerRange er(g_event_timer, "X = X + s * xi * sigma", stream);

            // X = X + s * xi * sigma
            CUBLAS_CHECK(cublasDgemm_v2(handles.cublas, CUBLAS_OP_N,
                                        CUBLAS_OP_N, s, s, s, d.d_one, d.xi, s,
                                        d.sigma, s, d.d_zero, d.temp, n));

            CUBLAS_CHECK(cublasDgemm_v2(handles.cublas, CUBLAS_OP_N,
                                        CUBLAS_OP_N, n, s, s, d.d_one, d.s, n,
                                        d.temp, n, d.d_one, d_X, n));
        }

        double relative_residual_norm = 0;
        {
            nvtx3::scoped_range rrn_range{"norm(B1 - A * X1) / norm(B1)"};
            CudaTimerRange er(g_event_timer, "norm(B1 - A * X1) / norm(B1)",
                              stream);

            // norm(B(:,1) - A * X(:,1)) / norm(B(:,1))
            constexpr cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
            constexpr double alpha = -1.0;
            constexpr double beta = 1.0;
            constexpr cudaDataType_t compute_type = CUDA_R_64F;
            constexpr cusparseSpMVAlg_t alg = CUSPARSE_SPMV_ALG_DEFAULT;

            CUDA_CHECK(cudaMemcpyAsync(d.temp, d_B, sizeof(double) * n,
                                       cudaMemcpyDeviceToDevice, stream));

            CUSPARSE_CHECK(cusparseSpMV(handles.cusparse, op, &alpha, A, X1,
                                        &beta, temp1, compute_type, alg,
                                        scratch_d));

            CUBLAS_CHECK(
                cublasDnrm2_v2(handles.cublas, n, d.temp, incx, d_norm));
            double residual_norm = 0;
            CUDA_CHECK(cudaMemcpyAsync(&residual_norm, d_norm, sizeof(double),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            if (*qr_ws.h_info < 0)
                throw std::runtime_error(std::to_string(-*qr_ws.h_info) +
                                         "-th parameter is wrong in QR\n");
            if (*lu_ws.h_info < 0)
                throw std::runtime_error(std::to_string(-*lu_ws.h_info) +
                                         "-th parameter is wrong in LU\n");

            relative_residual_norm = residual_norm / B1_norm;
            LOG_TRACE(relative_residual_norm);
        }

        if (relative_residual_norm < tolerance) {
            break;
        }

        // We break [w zeta] = QR(w - L^-1 * A * s * xi) into two steps for timing purposes:
        // 1. w = w - L^-1 * A * s * xi
        // 2. [w zeta] = QR(w)
        {
            nvtx3::scoped_range w_zeta_range{"w = w - L^-1 * A * s * xi"};
            CudaTimerRange er(g_event_timer, "w = w - L^-1 * A * s * xi", stream);

            // temp = A * s
            constexpr cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
            constexpr double alpha = 1.0;
            constexpr double beta = 0.0;
            constexpr cudaDataType compute_type = CUDA_R_64F;
            constexpr cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT;

            CUSPARSE_CHECK(cusparseSpMM(handles.cusparse, op, op, &alpha, A,
                                        s_desc, &beta, temp, compute_type, alg,
                                        scratch_d));

            // temp = L^-1 * temp
            sptri_solve<double>(handles.cusparse, temp, op, L, temp, spsm_nt);

            // w = w - temp * xi
            constexpr cublasOperation_t sgemm_op = CUBLAS_OP_N;
            CUBLAS_CHECK(cublasDgemm_v2(handles.cublas, sgemm_op, sgemm_op, n,
                                        s, s, d.d_neg_one, d.temp, n, d.xi, s,
                                        d.d_one, d.w, n));
        }

        {
            nvtx3::scoped_range w_zeta_range{"[w zeta] = QR(w)"};
            CudaTimerRange er(g_event_timer, "[w zeta] = QR(w)", stream);

            // [w, zeta] = qr(w)
            qr_factorization(handles.cusolver, handles.cusolver_params, d.w,
                             d.zeta, n, s, d.w, qr_ws, stream);
        }

        {
            nvtx3::scoped_range s_range{"s = (L^-1)' * w + s * zeta'"};
            CudaTimerRange er(g_event_timer, "s = (L^-1)' * w + s * zeta'",
                              stream);

            // s = (L^-1)' * w + s * zeta'
            constexpr cublasSideMode_t side = CUBLAS_SIDE_RIGHT;
            constexpr cublasFillMode_t fill_mode = CUBLAS_FILL_MODE_UPPER;
            constexpr cublasDiagType_t diag_type = CUBLAS_DIAG_NON_UNIT;
            constexpr cublasOperation_t op_zeta = CUBLAS_OP_T;

            CUBLAS_CHECK(cublasDtrmm_v2(handles.cublas, side, fill_mode,
                                        op_zeta, diag_type, n, s, d.d_one,
                                        d.zeta, s, d.s, n, d.s, n));

            sptri_solve<double>(handles.cusparse, temp,
                                CUSPARSE_OPERATION_TRANSPOSE, L, w_desc,
                                spsm_t);

            constexpr cublasOperation_t sgeam_op = CUBLAS_OP_N;
            CUBLAS_CHECK(cublasDgeam(handles.cublas, sgeam_op, sgeam_op, n, s,
                                     d.d_one, d.s, n, d.d_one, d.temp,
                                     n, d.s, n));
        }

        {
            nvtx3::scoped_range sigma_range{"sigma = zeta * sigma"};
            CudaTimerRange er(g_event_timer, "sigma = zeta * sigma", stream);

            // sigma = zeta * sigma
            constexpr cublasSideMode_t side = CUBLAS_SIDE_LEFT;
            constexpr cublasFillMode_t fill_mode = CUBLAS_FILL_MODE_UPPER;
            constexpr cublasDiagType_t diag_type = CUBLAS_DIAG_NON_UNIT;
            constexpr cublasOperation_t op_zeta = CUBLAS_OP_N;

            CUBLAS_CHECK(cublasDtrmm_v2(handles.cublas, side, fill_mode,
                                        op_zeta, diag_type, s, s, d.d_one,
                                        d.zeta, s, d.sigma, s, d.sigma, s));
        }
    }

    CUDA_CHECK(cudaFreeAsync(d_norm, stream));
    CUDA_CHECK(cudaFreeAsync(scratch_d, stream));
    CUSPARSE_CHECK(cusparseDestroyDnVec(temp1));
    CUSPARSE_CHECK(cusparseDestroyDnVec(X1));

    return iterations;
}

} // namespace dr_bcg::cuda
