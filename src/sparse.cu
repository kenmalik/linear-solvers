#include "dr_bcg/helper.h"
#include "dr_bcg/internal/device_buffer.h"
#include "dr_bcg/internal/math.h"
#include "dr_bcg/sparse.h"

#include <cstdint>
#include <functional>
#include <iostream>

namespace {
struct Handles {
    cusparseHandle_t cusparse;
    cusolverDnHandle_t cusolver;
    cusolverDnParams_t cusolver_params;
    cublasHandle_t cublas;

    Handles() {
        CUSPARSE_CHECK(cusparseCreate(&cusparse));
        CUSOLVER_CHECK(cusolverDnCreate(&cusolver));
        CUSOLVER_CHECK(cusolverDnCreateParams(&cusolver_params));
        CUBLAS_CHECK(cublasCreate_v2(&cublas));
    }

    ~Handles() {
        CUSPARSE_CHECK(cusparseDestroy(cusparse));
        CUSOLVER_CHECK(cusolverDnDestroy(cusolver));
        CUSOLVER_CHECK(cusolverDnDestroyParams(cusolver_params));
        CUBLAS_CHECK(cublasDestroy_v2(cublas));
    };

    void set_stream(cudaStream_t stream) {
        CUSPARSE_CHECK(cusparseSetStream(cusparse, stream));
        CUSOLVER_CHECK(cusolverDnSetStream(cusolver, stream));
        CUBLAS_CHECK(cublasSetStream_v2(cublas, stream));
    }
};

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

int dr_bcg::dr_bcg(cusparseSpMatDescr_t A, cusparseDnMatDescr_t X,
                   cusparseDnMatDescr_t B, float tolerance, int max_iterations,
                   std::function<void(int, float)> residual_callback) {
    auto [n, s] = get_size(B);
    Device_buffer<float> d(n, s);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    Handles handles;
    handles.set_stream(stream);

    void *scratch_d = nullptr;

    cusparseDnMatDescr_t temp;
    CUSPARSE_CHECK(cusparseCreateDnMat(&temp, n, s, n, d.temp, CUDA_R_32F,
                                       CUSPARSE_ORDER_COL));

    float *d_X = nullptr;
    CUSPARSE_CHECK(cusparseDnMatGetValues(X, reinterpret_cast<void **>(&d_X)));

    // Precalculate B1 norm for conversion checks
    float *d_B = nullptr;
    CUSPARSE_CHECK(cusparseDnMatGetValues(B, reinterpret_cast<void **>(&d_B)));

    constexpr int incx = 1;
    float B1_norm = 0;
    cublasSnrm2_v2(handles.cublas, n, d_B, incx, &B1_norm);

    float *d_R = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_R, sizeof(float) * n * s, stream));
    cusparseDnMatDescr_t R;
    CUSPARSE_CHECK(
        cusparseCreateDnMat(&R, n, s, n, d_R, CUDA_R_32F, CUSPARSE_ORDER_COL));

    {
        // R = B - A * X
        std::size_t buffer_size;
        constexpr float alpha = -1.0f;
        constexpr float beta = 1.0f;
        constexpr cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
        constexpr cudaDataType_t compute_type = CUDA_R_32F;
        constexpr cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT;

        void *d_B = nullptr;
        CUSPARSE_CHECK(cusparseDnMatGetValues(B, &d_B));
        CUDA_CHECK(cudaMemcpyAsync(d_R, d_B, sizeof(float) * n * s,
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
        // [w, sigma] = qr(R, 'econ')
        qr_factorization(handles.cusolver, handles.cusolver_params, d.w,
                         d.sigma, n, s, d_R);
    }

    CUDA_CHECK(cudaFreeAsync(d_R, stream));
    CUSPARSE_CHECK(cusparseDestroyDnMat(R));

    {
        // s = w
        CUDA_CHECK(cudaMemcpyAsync(d.s, d.w, sizeof(float) * n * s,
                                   cudaMemcpyDeviceToDevice, stream));
    }

    int iterations = 0;
    while (iterations < max_iterations) {
        ++iterations;

        {
            // xi = (s' * A * s)^-1
            std::size_t buffer_size;
            constexpr float alpha = 1.0f;
            constexpr float beta = 0.0f;
            constexpr cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
            constexpr cudaDataType_t compute_type = CUDA_R_32F;
            constexpr cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT;

            cusparseDnMatDescr_t s_mat;
            CUSPARSE_CHECK(cusparseCreateDnMat(&s_mat, n, s, n, d.s, CUDA_R_32F,
                                               CUSPARSE_ORDER_COL));

            CUSPARSE_CHECK(cusparseSpMM_bufferSize(
                handles.cusparse, op, op, &alpha, A, s_mat, &beta, temp,
                compute_type, alg, &buffer_size));

            CUDA_CHECK(cudaMallocAsync(&scratch_d, buffer_size, stream));

            CUSPARSE_CHECK(cusparseSpMM(handles.cusparse, op, op, &alpha, A,
                                        s_mat, &beta, temp, compute_type, alg,
                                        scratch_d));

            constexpr cublasOperation_t op_t = CUBLAS_OP_T;
            constexpr cublasOperation_t op_n = CUBLAS_OP_N;
            CUBLAS_CHECK(cublasSgemm_v2(handles.cublas, op_t, op_n, s, s, n,
                                        &alpha, d.s, n, d.temp, n, &beta, d.xi,
                                        s));

            invert_square_matrix(handles.cusolver, handles.cusolver_params,
                                 d.xi, s);

            CUDA_CHECK(cudaFreeAsync(scratch_d, stream));
        }

        {
            // X = X + s * xi * sigma
            constexpr float alpha_1 = 1.0f;
            constexpr float beta_1 = 0.0f;
            CUBLAS_CHECK(cublasSgemm_v2(handles.cublas, CUBLAS_OP_N,
                                        CUBLAS_OP_N, s, s, s, &alpha_1, d.xi, s,
                                        d.sigma, s, &beta_1, d.temp, n));

            constexpr float alpha_2 = 1.0f;
            constexpr float beta_2 = 1.0f;
            CUBLAS_CHECK(cublasSgemm_v2(handles.cublas, CUBLAS_OP_N,
                                        CUBLAS_OP_N, n, s, s, &alpha_2, d.s, n,
                                        d.temp, n, &beta_2, d_X, n));
        }

        float relative_residual_norm = 0;
        {
            // norm(B(:,1) - A * X(:,1)) / norm(B(:,1))
            constexpr cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
            constexpr float alpha = -1.0f;
            constexpr float beta = 1.0f;
            constexpr cudaDataType_t compute_type = CUDA_R_32F;
            constexpr cusparseSpMVAlg_t alg = CUSPARSE_SPMV_ALG_DEFAULT;

            CUDA_CHECK(cudaMemcpyAsync(d.temp, d_B, sizeof(float) * n,
                                       cudaMemcpyDeviceToDevice, stream));

            cusparseDnVecDescr_t temp1;
            CUSPARSE_CHECK(cusparseCreateDnVec(&temp1, n, d.temp, CUDA_R_32F));

            cusparseDnVecDescr_t X1;
            CUSPARSE_CHECK(cusparseCreateDnVec(&X1, n, d_X, CUDA_R_32F));

            std::size_t buffer_size = 0;
            CUSPARSE_CHECK(cusparseSpMV_bufferSize(
                handles.cusparse, op, &alpha, A, X1, &beta, temp1, compute_type,
                alg, &buffer_size));

            CUDA_CHECK(cudaMallocAsync(&scratch_d, buffer_size, stream));

            CUSPARSE_CHECK(cusparseSpMV(handles.cusparse, op, &alpha, A, X1,
                                        &beta, temp1, compute_type, alg,
                                        scratch_d));

            CUDA_CHECK(cudaFreeAsync(scratch_d, stream));

            constexpr int incx = 1;
            float residual_norm = 0;
            CUBLAS_CHECK(cublasSnrm2_v2(handles.cublas, n, d.temp, incx,
                                        &residual_norm));

            relative_residual_norm = residual_norm / B1_norm;
        }

        if (residual_callback) {
            residual_callback(iterations, relative_residual_norm);
        }
        if (relative_residual_norm < tolerance) {
            break;
        }

        {
            // [w, zeta] = qr(w - A * s * xi, 'econ')
            constexpr cublasOperation_t op = CUBLAS_OP_N;
            constexpr float sgemm_alpha = 1.0f;
            constexpr float sgemm_beta = 0.0f;
            CUBLAS_CHECK(cublasSgemm_v2(handles.cublas, op, op, n, s, s,
                                        &sgemm_alpha, d.s, n, d.xi, s,
                                        &sgemm_beta, d.temp, n));

            constexpr cusparseOperation_t spmm_op =
                CUSPARSE_OPERATION_NON_TRANSPOSE;
            constexpr float spmm_alpha = -1.0f;
            constexpr float spmm_beta = 1.0f;
            constexpr cudaDataType_t compute_type = CUDA_R_32F;
            constexpr cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT;

            cusparseDnMatDescr_t w;
            CUSPARSE_CHECK(cusparseCreateDnMat(&w, n, s, n, d.w, CUDA_R_32F,
                                               CUSPARSE_ORDER_COL));

            std::size_t buffer_size = 0;
            CUSPARSE_CHECK(cusparseSpMM_bufferSize(
                handles.cusparse, spmm_op, spmm_op, &spmm_alpha, A, temp,
                &spmm_beta, w, compute_type, alg, &buffer_size));

            CUDA_CHECK(cudaMallocAsync(&scratch_d, buffer_size, stream));

            CUSPARSE_CHECK(cusparseSpMM(handles.cusparse, spmm_op, spmm_op,
                                        &spmm_alpha, A, temp, &spmm_beta, w,
                                        compute_type, alg, scratch_d));

            CUDA_CHECK(cudaFreeAsync(scratch_d, stream));

            qr_factorization(handles.cusolver, handles.cusolver_params, d.w,
                             d.zeta, n, s, d.w);
        }

        {
            // s = w + s * zeta'
            constexpr float alpha = 1.0f;
            constexpr cublasSideMode_t side = CUBLAS_SIDE_RIGHT;
            constexpr cublasFillMode_t fill_mode = CUBLAS_FILL_MODE_UPPER;
            constexpr cublasDiagType_t diag_type = CUBLAS_DIAG_NON_UNIT;
            constexpr cublasOperation_t op_zeta = CUBLAS_OP_T;

            CUBLAS_CHECK(cublasStrmm_v2(handles.cublas, side, fill_mode,
                                        op_zeta, diag_type, n, s, &alpha,
                                        d.zeta, s, d.s, n, d.s, n));

            constexpr cublasOperation_t sgeam_op = CUBLAS_OP_N;
            constexpr float sgeam_alpha = 1.0f;
            constexpr float sgeam_beta = 1.0f;
            CUBLAS_CHECK(cublasSgeam(handles.cublas, sgeam_op, sgeam_op, n, s,
                                     &sgeam_alpha, d.s, n, &sgeam_beta, d.w, n,
                                     d.s, n));
        }

        {
            // sigma = zeta * sigma
            constexpr float alpha = 1.0f;
            constexpr cublasSideMode_t side = CUBLAS_SIDE_LEFT;
            constexpr cublasFillMode_t fill_mode = CUBLAS_FILL_MODE_UPPER;
            constexpr cublasDiagType_t diag_type = CUBLAS_DIAG_NON_UNIT;
            constexpr cublasOperation_t op_zeta = CUBLAS_OP_N;

            CUBLAS_CHECK(cublasStrmm_v2(handles.cublas, side, fill_mode,
                                        op_zeta, diag_type, s, s, &alpha,
                                        d.zeta, s, d.sigma, s, d.sigma, s));
        }
    }

    return iterations;
}

// Double-precision variant: same algorithm but using double/cuBLAS/cuSPARSE D
// APIs
int dr_bcg::dr_bcg(cusparseSpMatDescr_t A, cusparseDnMatDescr_t X,
                   cusparseDnMatDescr_t B, double tolerance, int max_iterations,
                   std::function<void(int, double)> residual_callback) {
    auto [n, s] = get_size(B);
    Device_buffer<double> d(n, s);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    Handles handles;
    handles.set_stream(stream);

    void *scratch_d = nullptr;

    cusparseDnMatDescr_t temp;
    CUSPARSE_CHECK(cusparseCreateDnMat(&temp, n, s, n, d.temp, CUDA_R_64F,
                                       CUSPARSE_ORDER_COL));

    double *d_X = nullptr;
    CUSPARSE_CHECK(cusparseDnMatGetValues(X, reinterpret_cast<void **>(&d_X)));

    // Precalculate B1 norm for conversion checks
    double *d_B = nullptr;
    CUSPARSE_CHECK(cusparseDnMatGetValues(B, reinterpret_cast<void **>(&d_B)));

    constexpr int incx = 1;
    double B1_norm = 0;
    cublasDnrm2_v2(handles.cublas, n, d_B, incx, &B1_norm);

    double *d_R = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_R, sizeof(double) * n * s, stream));
    cusparseDnMatDescr_t R;
    CUSPARSE_CHECK(
        cusparseCreateDnMat(&R, n, s, n, d_R, CUDA_R_64F, CUSPARSE_ORDER_COL));

    {
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
        // [w, sigma] = qr(R, 'econ')
        qr_factorization(handles.cusolver, handles.cusolver_params, d.w,
                         d.sigma, n, s, d_R);
    }

    CUDA_CHECK(cudaFreeAsync(d_R, stream));
    CUSPARSE_CHECK(cusparseDestroyDnMat(R));

    {
        // s = w
        CUDA_CHECK(cudaMemcpyAsync(d.s, d.w, sizeof(double) * n * s,
                                   cudaMemcpyDeviceToDevice, stream));
    }

    int iterations = 0;
    while (iterations < max_iterations) {
        ++iterations;

        {
            // xi = (s' * A * s)^-1
            std::size_t buffer_size;
            constexpr double alpha = 1.0;
            constexpr double beta = 0.0;
            constexpr cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
            constexpr cudaDataType_t compute_type = CUDA_R_64F;
            constexpr cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT;

            cusparseDnMatDescr_t s_mat;
            CUSPARSE_CHECK(cusparseCreateDnMat(&s_mat, n, s, n, d.s, CUDA_R_64F,
                                               CUSPARSE_ORDER_COL));

            CUSPARSE_CHECK(cusparseSpMM_bufferSize(
                handles.cusparse, op, op, &alpha, A, s_mat, &beta, temp,
                compute_type, alg, &buffer_size));

            CUDA_CHECK(cudaMallocAsync(&scratch_d, buffer_size, stream));

            CUSPARSE_CHECK(cusparseSpMM(handles.cusparse, op, op, &alpha, A,
                                        s_mat, &beta, temp, compute_type, alg,
                                        scratch_d));

            constexpr cublasOperation_t op_t = CUBLAS_OP_T;
            constexpr cublasOperation_t op_n = CUBLAS_OP_N;
            CUBLAS_CHECK(cublasDgemm_v2(handles.cublas, op_t, op_n, s, s, n,
                                        &alpha, d.s, n, d.temp, n, &beta, d.xi,
                                        s));

            invert_square_matrix(handles.cusolver, handles.cusolver_params,
                                 d.xi, s);

            CUDA_CHECK(cudaFreeAsync(scratch_d, stream));
        }

        {
            // X = X + s * xi * sigma
            constexpr double alpha_1 = 1.0;
            constexpr double beta_1 = 0.0;
            CUBLAS_CHECK(cublasDgemm_v2(handles.cublas, CUBLAS_OP_N,
                                        CUBLAS_OP_N, s, s, s, &alpha_1, d.xi, s,
                                        d.sigma, s, &beta_1, d.temp, n));

            constexpr double alpha_2 = 1.0;
            constexpr double beta_2 = 1.0;
            CUBLAS_CHECK(cublasDgemm_v2(handles.cublas, CUBLAS_OP_N,
                                        CUBLAS_OP_N, n, s, s, &alpha_2, d.s, n,
                                        d.temp, n, &beta_2, d_X, n));
        }

        double relative_residual_norm = 0;
        {
            // norm(B(:,1) - A * X(:,1)) / norm(B(:,1))
            constexpr cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
            constexpr double alpha = -1.0;
            constexpr double beta = 1.0;
            constexpr cudaDataType_t compute_type = CUDA_R_64F;
            constexpr cusparseSpMVAlg_t alg = CUSPARSE_SPMV_ALG_DEFAULT;

            CUDA_CHECK(cudaMemcpyAsync(d.temp, d_B, sizeof(double) * n,
                                       cudaMemcpyDeviceToDevice, stream));

            cusparseDnVecDescr_t temp1;
            CUSPARSE_CHECK(cusparseCreateDnVec(&temp1, n, d.temp, CUDA_R_64F));

            cusparseDnVecDescr_t X1;
            CUSPARSE_CHECK(cusparseCreateDnVec(&X1, n, d_X, CUDA_R_64F));

            std::size_t buffer_size = 0;
            CUSPARSE_CHECK(cusparseSpMV_bufferSize(
                handles.cusparse, op, &alpha, A, X1, &beta, temp1, compute_type,
                alg, &buffer_size));

            CUDA_CHECK(cudaMallocAsync(&scratch_d, buffer_size, stream));

            CUSPARSE_CHECK(cusparseSpMV(handles.cusparse, op, &alpha, A, X1,
                                        &beta, temp1, compute_type, alg,
                                        scratch_d));

            CUDA_CHECK(cudaFreeAsync(scratch_d, stream));

            constexpr int incx = 1;
            double residual_norm = 0;
            CUBLAS_CHECK(cublasDnrm2_v2(handles.cublas, n, d.temp, incx,
                                        &residual_norm));

            relative_residual_norm = residual_norm / B1_norm;
        }

        if (residual_callback) {
            residual_callback(iterations, relative_residual_norm);
        }
        if (relative_residual_norm < tolerance) {
            break;
        }

        {
            // [w, zeta] = qr(w - A * s * xi, 'econ')
            constexpr cublasOperation_t op = CUBLAS_OP_N;
            constexpr double sgemm_alpha = 1.0;
            constexpr double sgemm_beta = 0.0;
            CUBLAS_CHECK(cublasDgemm_v2(handles.cublas, op, op, n, s, s,
                                        &sgemm_alpha, d.s, n, d.xi, s,
                                        &sgemm_beta, d.temp, n));

            constexpr cusparseOperation_t spmm_op =
                CUSPARSE_OPERATION_NON_TRANSPOSE;
            constexpr double spmm_alpha = -1.0;
            constexpr double spmm_beta = 1.0;
            constexpr cudaDataType_t compute_type = CUDA_R_64F;
            constexpr cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT;

            cusparseDnMatDescr_t w;
            CUSPARSE_CHECK(cusparseCreateDnMat(&w, n, s, n, d.w, CUDA_R_64F,
                                               CUSPARSE_ORDER_COL));

            std::size_t buffer_size = 0;
            CUSPARSE_CHECK(cusparseSpMM_bufferSize(
                handles.cusparse, spmm_op, spmm_op, &spmm_alpha, A, temp,
                &spmm_beta, w, compute_type, alg, &buffer_size));

            CUDA_CHECK(cudaMallocAsync(&scratch_d, buffer_size, stream));

            CUSPARSE_CHECK(cusparseSpMM(handles.cusparse, spmm_op, spmm_op,
                                        &spmm_alpha, A, temp, &spmm_beta, w,
                                        compute_type, alg, scratch_d));

            CUDA_CHECK(cudaFreeAsync(scratch_d, stream));

            qr_factorization(handles.cusolver, handles.cusolver_params, d.w,
                             d.zeta, n, s, d.w);
        }

        {
            // s = w + s * zeta'
            constexpr double alpha = 1.0;
            constexpr cublasSideMode_t side = CUBLAS_SIDE_RIGHT;
            constexpr cublasFillMode_t fill_mode = CUBLAS_FILL_MODE_UPPER;
            constexpr cublasDiagType_t diag_type = CUBLAS_DIAG_NON_UNIT;
            constexpr cublasOperation_t op_zeta = CUBLAS_OP_T;

            CUBLAS_CHECK(cublasDtrmm_v2(handles.cublas, side, fill_mode,
                                        op_zeta, diag_type, n, s, &alpha,
                                        d.zeta, s, d.s, n, d.s, n));

            constexpr cublasOperation_t sgeam_op = CUBLAS_OP_N;
            constexpr double sgeam_alpha = 1.0;
            constexpr double sgeam_beta = 1.0;
            CUBLAS_CHECK(cublasDgeam(handles.cublas, sgeam_op, sgeam_op, n, s,
                                     &sgeam_alpha, d.s, n, &sgeam_beta, d.w, n,
                                     d.s, n));
        }

        {
            // sigma = zeta * sigma
            constexpr double alpha = 1.0;
            constexpr cublasSideMode_t side = CUBLAS_SIDE_LEFT;
            constexpr cublasFillMode_t fill_mode = CUBLAS_FILL_MODE_UPPER;
            constexpr cublasDiagType_t diag_type = CUBLAS_DIAG_NON_UNIT;
            constexpr cublasOperation_t op_zeta = CUBLAS_OP_N;

            CUBLAS_CHECK(cublasDtrmm_v2(handles.cublas, side, fill_mode,
                                        op_zeta, diag_type, s, s, &alpha,
                                        d.zeta, s, d.sigma, s, d.sigma, s));
        }
    }

    return iterations;
}

// Preconditioned double-precision variant
int dr_bcg::dr_bcg(cusparseSpMatDescr_t A, cusparseDnMatDescr_t X,
                   cusparseDnMatDescr_t B, cusparseSpMatDescr_t L,
                   double tolerance, int max_iterations,
                   std::function<void(int, double)> residual_callback) {
    auto [n, s] = get_size(B);
    Device_buffer<double> d(n, s);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    Handles handles;
    handles.set_stream(stream);

    void *scratch_d = nullptr;

    cusparseDnMatDescr_t temp;
    CUSPARSE_CHECK(cusparseCreateDnMat(&temp, n, s, n, d.temp, CUDA_R_64F,
                                       CUSPARSE_ORDER_COL));
    cusparseDnMatDescr_t s_desc;
    CUSPARSE_CHECK(cusparseCreateDnMat(&s_desc, n, s, n, d.s, CUDA_R_64F,
                                       CUSPARSE_ORDER_COL));
    cusparseDnMatDescr_t w_desc;
    CUSPARSE_CHECK(cusparseCreateDnMat(&w_desc, n, s, n, d.w, CUDA_R_64F,
                                       CUSPARSE_ORDER_COL));

    double *d_X = nullptr;
    CUSPARSE_CHECK(cusparseDnMatGetValues(X, reinterpret_cast<void **>(&d_X)));

    // Precalculate B1 norm for conversion checks
    double *d_B = nullptr;
    CUSPARSE_CHECK(cusparseDnMatGetValues(B, reinterpret_cast<void **>(&d_B)));

    constexpr int incx = 1;
    double B1_norm = 0;
    cublasDnrm2_v2(handles.cublas, n, d_B, incx, &B1_norm);

    double *d_R = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_R, sizeof(double) * n * s, stream));
    cusparseDnMatDescr_t R;
    CUSPARSE_CHECK(
        cusparseCreateDnMat(&R, n, s, n, d_R, CUDA_R_64F, CUSPARSE_ORDER_COL));

    {
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
        // [w, sigma] = qr(L^-1 * R, 'econ')
        sptri_left_multiply<double>(handles.cusparse, temp,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE, L, R);

        qr_factorization(handles.cusolver, handles.cusolver_params, d.w,
                         d.sigma, n, s, d.temp);
    }

    CUDA_CHECK(cudaFreeAsync(d_R, stream));
    CUSPARSE_CHECK(cusparseDestroyDnMat(R));

    {
        // s = (L^-1)' * w
        CUDA_CHECK(cudaMemcpyAsync(d.s, d.w, sizeof(double) * n * s,
                                   cudaMemcpyDeviceToDevice, stream));

        sptri_left_multiply<double>(handles.cusparse, s_desc,
                                    CUSPARSE_OPERATION_TRANSPOSE, L, w_desc);
    }

    int iterations = 0;
    while (iterations < max_iterations) {
        ++iterations;

        {
            // xi = (s' * A * s)^-1
            std::size_t buffer_size;
            constexpr double alpha = 1.0;
            constexpr double beta = 0.0;
            constexpr cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
            constexpr cudaDataType_t compute_type = CUDA_R_64F;
            constexpr cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT;

            cusparseDnMatDescr_t s_mat;
            CUSPARSE_CHECK(cusparseCreateDnMat(&s_mat, n, s, n, d.s, CUDA_R_64F,
                                               CUSPARSE_ORDER_COL));

            CUSPARSE_CHECK(cusparseSpMM_bufferSize(
                handles.cusparse, op, op, &alpha, A, s_mat, &beta, temp,
                compute_type, alg, &buffer_size));

            CUDA_CHECK(cudaMallocAsync(&scratch_d, buffer_size, stream));

            CUSPARSE_CHECK(cusparseSpMM(handles.cusparse, op, op, &alpha, A,
                                        s_mat, &beta, temp, compute_type, alg,
                                        scratch_d));

            constexpr cublasOperation_t op_t = CUBLAS_OP_T;
            constexpr cublasOperation_t op_n = CUBLAS_OP_N;
            CUBLAS_CHECK(cublasDgemm_v2(handles.cublas, op_t, op_n, s, s, n,
                                        &alpha, d.s, n, d.temp, n, &beta, d.xi,
                                        s));

            invert_square_matrix(handles.cusolver, handles.cusolver_params,
                                 d.xi, s);

            CUDA_CHECK(cudaFreeAsync(scratch_d, stream));
        }

        {
            // X = X + s * xi * sigma
            constexpr double alpha_1 = 1.0;
            constexpr double beta_1 = 0.0;
            CUBLAS_CHECK(cublasDgemm_v2(handles.cublas, CUBLAS_OP_N,
                                        CUBLAS_OP_N, s, s, s, &alpha_1, d.xi, s,
                                        d.sigma, s, &beta_1, d.temp, n));

            constexpr double alpha_2 = 1.0;
            constexpr double beta_2 = 1.0;
            CUBLAS_CHECK(cublasDgemm_v2(handles.cublas, CUBLAS_OP_N,
                                        CUBLAS_OP_N, n, s, s, &alpha_2, d.s, n,
                                        d.temp, n, &beta_2, d_X, n));
        }

        double relative_residual_norm = 0;
        {
            // norm(B(:,1) - A * X(:,1)) / norm(B(:,1))
            constexpr cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
            constexpr double alpha = -1.0;
            constexpr double beta = 1.0;
            constexpr cudaDataType_t compute_type = CUDA_R_64F;
            constexpr cusparseSpMVAlg_t alg = CUSPARSE_SPMV_ALG_DEFAULT;

            CUDA_CHECK(cudaMemcpyAsync(d.temp, d_B, sizeof(double) * n,
                                       cudaMemcpyDeviceToDevice, stream));

            cusparseDnVecDescr_t temp1;
            CUSPARSE_CHECK(cusparseCreateDnVec(&temp1, n, d.temp, CUDA_R_64F));

            cusparseDnVecDescr_t X1;
            CUSPARSE_CHECK(cusparseCreateDnVec(&X1, n, d_X, CUDA_R_64F));

            std::size_t buffer_size = 0;
            CUSPARSE_CHECK(cusparseSpMV_bufferSize(
                handles.cusparse, op, &alpha, A, X1, &beta, temp1, compute_type,
                alg, &buffer_size));

            CUDA_CHECK(cudaMallocAsync(&scratch_d, buffer_size, stream));

            CUSPARSE_CHECK(cusparseSpMV(handles.cusparse, op, &alpha, A, X1,
                                        &beta, temp1, compute_type, alg,
                                        scratch_d));

            CUDA_CHECK(cudaFreeAsync(scratch_d, stream));

            constexpr int incx = 1;
            double residual_norm = 0;
            CUBLAS_CHECK(cublasDnrm2_v2(handles.cublas, n, d.temp, incx,
                                        &residual_norm));

            relative_residual_norm = residual_norm / B1_norm;
        }

        if (residual_callback) {
            residual_callback(iterations, relative_residual_norm);
        }
        if (relative_residual_norm < tolerance) {
            break;
        }

        {
            // [w, zeta] = qr(w - L^-1 * A * s * xi, 'econ')

            // temp = A * s
            constexpr cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
            constexpr double alpha = 1.0;
            constexpr double beta = 0.0;
            constexpr cudaDataType compute_type = CUDA_R_64F;
            constexpr cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT;

            void *buffer = nullptr;
            std::size_t buffer_size = 0;

            CUSPARSE_CHECK(cusparseSpMM_bufferSize(
                handles.cusparse, op, op, &alpha, A, s_desc, &beta, temp,
                compute_type, alg, &buffer_size));

            if (buffer_size > 0) {
                CUDA_CHECK(cudaMalloc(&buffer, buffer_size));
            }

            CUSPARSE_CHECK(cusparseSpMM(handles.cusparse, op, op, &alpha, A,
                                        s_desc, &beta, temp, compute_type, alg,
                                        buffer));

            if (buffer) {
                CUDA_CHECK(cudaFree(buffer));
                buffer = nullptr;
            }

            // temp = L^-1 * temp
            sptri_left_multiply<double>(handles.cusparse, temp, op, L, temp);

            // w = w - temp * xi
            constexpr cublasOperation_t sgemm_op = CUBLAS_OP_N;
            constexpr double sgemm_alpha = -1.0;
            constexpr double sgemm_beta = 1.0;
            CUBLAS_CHECK(cublasDgemm_v2(handles.cublas, sgemm_op, sgemm_op, n,
                                        s, s, &sgemm_alpha, d.temp, n, d.xi, s,
                                        &sgemm_beta, d.w, n));

            // [w, zeta] = qr(w)
            qr_factorization(handles.cusolver, handles.cusolver_params, d.w,
                             d.zeta, n, s, d.w);
        }

        {
            // s = (L^-1)' * w + s * zeta'
            constexpr double alpha = 1.0;
            constexpr cublasSideMode_t side = CUBLAS_SIDE_RIGHT;
            constexpr cublasFillMode_t fill_mode = CUBLAS_FILL_MODE_UPPER;
            constexpr cublasDiagType_t diag_type = CUBLAS_DIAG_NON_UNIT;
            constexpr cublasOperation_t op_zeta = CUBLAS_OP_T;

            CUBLAS_CHECK(cublasDtrmm_v2(handles.cublas, side, fill_mode,
                                        op_zeta, diag_type, n, s, &alpha,
                                        d.zeta, s, d.s, n, d.s, n));

            sptri_left_multiply<double>(handles.cusparse, temp,
                                        CUSPARSE_OPERATION_TRANSPOSE, L,
                                        w_desc);

            constexpr cublasOperation_t sgeam_op = CUBLAS_OP_N;
            constexpr double sgeam_alpha = 1.0;
            constexpr double sgeam_beta = 1.0;
            CUBLAS_CHECK(cublasDgeam(handles.cublas, sgeam_op, sgeam_op, n, s,
                                     &sgeam_alpha, d.s, n, &sgeam_beta, d.temp,
                                     n, d.s, n));
        }

        {
            // sigma = zeta * sigma
            constexpr double alpha = 1.0;
            constexpr cublasSideMode_t side = CUBLAS_SIDE_LEFT;
            constexpr cublasFillMode_t fill_mode = CUBLAS_FILL_MODE_UPPER;
            constexpr cublasDiagType_t diag_type = CUBLAS_DIAG_NON_UNIT;
            constexpr cublasOperation_t op_zeta = CUBLAS_OP_N;

            CUBLAS_CHECK(cublasDtrmm_v2(handles.cublas, side, fill_mode,
                                        op_zeta, diag_type, s, s, &alpha,
                                        d.zeta, s, d.sigma, s, d.sigma, s));
        }
    }

    return iterations;
}

// Preconditioned single-precision variant
int dr_bcg::dr_bcg(cusparseSpMatDescr_t A, cusparseDnMatDescr_t X,
                   cusparseDnMatDescr_t B, cusparseSpMatDescr_t L,
                   float tolerance, int max_iterations,
                   std::function<void(int, float)> residual_callback) {
    auto [n, s] = get_size(B);
    Device_buffer<float> d(n, s);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    Handles handles;
    handles.set_stream(stream);

    void *scratch_d = nullptr;

    cusparseDnMatDescr_t temp;
    CUSPARSE_CHECK(cusparseCreateDnMat(&temp, n, s, n, d.temp, CUDA_R_32F,
                                       CUSPARSE_ORDER_COL));
    cusparseDnMatDescr_t s_desc;
    CUSPARSE_CHECK(cusparseCreateDnMat(&s_desc, n, s, n, d.s, CUDA_R_32F,
                                       CUSPARSE_ORDER_COL));
    cusparseDnMatDescr_t w_desc;
    CUSPARSE_CHECK(cusparseCreateDnMat(&w_desc, n, s, n, d.w, CUDA_R_32F,
                                       CUSPARSE_ORDER_COL));

    float *d_X = nullptr;
    CUSPARSE_CHECK(cusparseDnMatGetValues(X, reinterpret_cast<void **>(&d_X)));

    // Precalculate B1 norm for conversion checks
    float *d_B = nullptr;
    CUSPARSE_CHECK(cusparseDnMatGetValues(B, reinterpret_cast<void **>(&d_B)));

    constexpr int incx = 1;
    float B1_norm = 0;
    cublasSnrm2_v2(handles.cublas, n, d_B, incx, &B1_norm);

    float *d_R = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_R, sizeof(float) * n * s, stream));
    cusparseDnMatDescr_t R;
    CUSPARSE_CHECK(
        cusparseCreateDnMat(&R, n, s, n, d_R, CUDA_R_32F, CUSPARSE_ORDER_COL));

    {
        // R = B - A * X
        std::size_t buffer_size;
        constexpr float alpha = -1.0;
        constexpr float beta = 1.0;
        constexpr cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
        constexpr cudaDataType_t compute_type = CUDA_R_32F;
        constexpr cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT;

        void *d_B_ptr = nullptr;
        CUSPARSE_CHECK(cusparseDnMatGetValues(B, &d_B_ptr));
        CUDA_CHECK(cudaMemcpyAsync(d_R, d_B_ptr, sizeof(float) * n * s,
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
        // [w, sigma] = qr(L^-1 * R, 'econ')
        sptri_left_multiply<float>(handles.cusparse, temp,
                                   CUSPARSE_OPERATION_NON_TRANSPOSE, L, R);

        qr_factorization(handles.cusolver, handles.cusolver_params, d.w,
                         d.sigma, n, s, d.temp);
    }

    CUDA_CHECK(cudaFreeAsync(d_R, stream));
    CUSPARSE_CHECK(cusparseDestroyDnMat(R));

    {
        // s = (L^-1)' * w
        CUDA_CHECK(cudaMemcpyAsync(d.s, d.w, sizeof(float) * n * s,
                                   cudaMemcpyDeviceToDevice, stream));

        sptri_left_multiply<float>(handles.cusparse, s_desc,
                                   CUSPARSE_OPERATION_TRANSPOSE, L, w_desc);
    }

    int iterations = 0;
    while (iterations < max_iterations) {
        ++iterations;

        {
            // xi = (s' * A * s)^-1
            std::size_t buffer_size;
            constexpr float alpha = 1.0;
            constexpr float beta = 0.0;
            constexpr cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
            constexpr cudaDataType_t compute_type = CUDA_R_32F;
            constexpr cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT;

            cusparseDnMatDescr_t s_mat;
            CUSPARSE_CHECK(cusparseCreateDnMat(&s_mat, n, s, n, d.s, CUDA_R_32F,
                                               CUSPARSE_ORDER_COL));

            CUSPARSE_CHECK(cusparseSpMM_bufferSize(
                handles.cusparse, op, op, &alpha, A, s_mat, &beta, temp,
                compute_type, alg, &buffer_size));

            CUDA_CHECK(cudaMallocAsync(&scratch_d, buffer_size, stream));

            CUSPARSE_CHECK(cusparseSpMM(handles.cusparse, op, op, &alpha, A,
                                        s_mat, &beta, temp, compute_type, alg,
                                        scratch_d));

            constexpr cublasOperation_t op_t = CUBLAS_OP_T;
            constexpr cublasOperation_t op_n = CUBLAS_OP_N;
            CUBLAS_CHECK(cublasSgemm_v2(handles.cublas, op_t, op_n, s, s, n,
                                        &alpha, d.s, n, d.temp, n, &beta, d.xi,
                                        s));

            invert_square_matrix(handles.cusolver, handles.cusolver_params,
                                 d.xi, s);

            CUDA_CHECK(cudaFreeAsync(scratch_d, stream));
        }

        {
            // X = X + s * xi * sigma
            constexpr float alpha_1 = 1.0;
            constexpr float beta_1 = 0.0;
            CUBLAS_CHECK(cublasSgemm_v2(handles.cublas, CUBLAS_OP_N,
                                        CUBLAS_OP_N, s, s, s, &alpha_1, d.xi, s,
                                        d.sigma, s, &beta_1, d.temp, n));

            constexpr float alpha_2 = 1.0;
            constexpr float beta_2 = 1.0;
            CUBLAS_CHECK(cublasSgemm_v2(handles.cublas, CUBLAS_OP_N,
                                        CUBLAS_OP_N, n, s, s, &alpha_2, d.s, n,
                                        d.temp, n, &beta_2, d_X, n));
        }

        float relative_residual_norm = 0;
        {
            // norm(B(:,1) - A * X(:,1)) / norm(B(:,1))
            constexpr cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
            constexpr float alpha = -1.0;
            constexpr float beta = 1.0;
            constexpr cudaDataType_t compute_type = CUDA_R_32F;
            constexpr cusparseSpMVAlg_t alg = CUSPARSE_SPMV_ALG_DEFAULT;

            CUDA_CHECK(cudaMemcpyAsync(d.temp, d_B, sizeof(float) * n,
                                       cudaMemcpyDeviceToDevice, stream));

            cusparseDnVecDescr_t temp1;
            CUSPARSE_CHECK(cusparseCreateDnVec(&temp1, n, d.temp, CUDA_R_32F));

            cusparseDnVecDescr_t X1;
            CUSPARSE_CHECK(cusparseCreateDnVec(&X1, n, d_X, CUDA_R_32F));

            std::size_t buffer_size = 0;
            CUSPARSE_CHECK(cusparseSpMV_bufferSize(
                handles.cusparse, op, &alpha, A, X1, &beta, temp1, compute_type,
                alg, &buffer_size));

            CUDA_CHECK(cudaMallocAsync(&scratch_d, buffer_size, stream));

            CUSPARSE_CHECK(cusparseSpMV(handles.cusparse, op, &alpha, A, X1,
                                        &beta, temp1, compute_type, alg,
                                        scratch_d));

            CUDA_CHECK(cudaFreeAsync(scratch_d, stream));

            constexpr int incx = 1;
            float residual_norm = 0;
            CUBLAS_CHECK(cublasSnrm2_v2(handles.cublas, n, d.temp, incx,
                                        &residual_norm));

            relative_residual_norm = residual_norm / B1_norm;
        }

        if (residual_callback) {
            residual_callback(iterations, relative_residual_norm);
        }
        if (relative_residual_norm < tolerance) {
            break;
        }

        {
            // [w, zeta] = qr(w - L^-1 * A * s * xi, 'econ')

            // temp = A * s
            constexpr cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
            constexpr float alpha = 1.0;
            constexpr float beta = 0.0;
            constexpr cudaDataType compute_type = CUDA_R_32F;
            constexpr cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT;

            void *buffer = nullptr;
            std::size_t buffer_size = 0;

            CUSPARSE_CHECK(cusparseSpMM_bufferSize(
                handles.cusparse, op, op, &alpha, A, s_desc, &beta, temp,
                compute_type, alg, &buffer_size));

            if (buffer_size > 0) {
                CUDA_CHECK(cudaMalloc(&buffer, buffer_size));
            }

            CUSPARSE_CHECK(cusparseSpMM(handles.cusparse, op, op, &alpha, A,
                                        s_desc, &beta, temp, compute_type, alg,
                                        buffer));

            if (buffer) {
                CUDA_CHECK(cudaFree(buffer));
                buffer = nullptr;
            }

            // temp = L^-1 * temp
            sptri_left_multiply<float>(handles.cusparse, temp, op, L, temp);

            // w = w - temp * xi
            constexpr cublasOperation_t sgemm_op = CUBLAS_OP_N;
            constexpr float sgemm_alpha = -1.0;
            constexpr float sgemm_beta = 1.0;
            CUBLAS_CHECK(cublasSgemm_v2(handles.cublas, sgemm_op, sgemm_op, n,
                                        s, s, &sgemm_alpha, d.temp, n, d.xi, s,
                                        &sgemm_beta, d.w, n));

            // [w, zeta] = qr(w)
            qr_factorization(handles.cusolver, handles.cusolver_params, d.w,
                             d.zeta, n, s, d.w);
        }

        {
            // s = (L^-1)' * w + s * zeta'
            constexpr float alpha = 1.0;
            constexpr cublasSideMode_t side = CUBLAS_SIDE_RIGHT;
            constexpr cublasFillMode_t fill_mode = CUBLAS_FILL_MODE_UPPER;
            constexpr cublasDiagType_t diag_type = CUBLAS_DIAG_NON_UNIT;
            constexpr cublasOperation_t op_zeta = CUBLAS_OP_T;

            CUBLAS_CHECK(cublasStrmm_v2(handles.cublas, side, fill_mode,
                                        op_zeta, diag_type, n, s, &alpha,
                                        d.zeta, s, d.s, n, d.s, n));

            sptri_left_multiply<float>(handles.cusparse, temp,
                                       CUSPARSE_OPERATION_TRANSPOSE, L, w_desc);

            constexpr cublasOperation_t sgeam_op = CUBLAS_OP_N;
            constexpr float sgeam_alpha = 1.0;
            constexpr float sgeam_beta = 1.0;
            CUBLAS_CHECK(cublasSgeam(handles.cublas, sgeam_op, sgeam_op, n, s,
                                     &sgeam_alpha, d.s, n, &sgeam_beta, d.temp,
                                     n, d.s, n));
        }

        {
            // sigma = zeta * sigma
            constexpr float alpha = 1.0;
            constexpr cublasSideMode_t side = CUBLAS_SIDE_LEFT;
            constexpr cublasFillMode_t fill_mode = CUBLAS_FILL_MODE_UPPER;
            constexpr cublasDiagType_t diag_type = CUBLAS_DIAG_NON_UNIT;
            constexpr cublasOperation_t op_zeta = CUBLAS_OP_N;

            CUBLAS_CHECK(cublasStrmm_v2(handles.cublas, side, fill_mode,
                                        op_zeta, diag_type, s, s, &alpha,
                                        d.zeta, s, d.sigma, s, d.sigma, s));
        }
    }

    return iterations;
}
