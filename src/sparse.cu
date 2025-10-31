#include "dr_bcg/device_buffer.h"
#include "dr_bcg/helper.h"
#include "dr_bcg/sparse.h"

#include <iostream>
#include <tuple>

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

int dr_bcg(cusparseSpMatDescr_t A, cusparseDnMatDescr_t X,
           cusparseDnMatDescr_t B, float tolerance, int max_iterations) {
    auto [n, s] = get_size(B);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    Handles handles;
    handles.set_stream(stream);

    void *scratch_d = nullptr;

    float *d_temp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_temp, sizeof(float) * n * s));
    cusparseDnMatDescr_t temp;
    CUSPARSE_CHECK(cusparseCreateDnMat(&temp, n, s, n, d_temp, CUDA_R_32F,
                                       CUSPARSE_ORDER_COL));

    void *X_vals = nullptr;
    CUSPARSE_CHECK(cusparseDnMatGetValues(X, &X_vals));
    float *d_X = static_cast<float *>(X_vals);

    float *d_R = nullptr;
    CUDA_CHECK(cudaMalloc(&d_R, sizeof(float) * n * s));
    cusparseDnMatDescr_t R;
    CUSPARSE_CHECK(
        cusparseCreateDnMat(&R, n, s, n, d_R, CUDA_R_32F, CUSPARSE_ORDER_COL));

    DeviceBuffer d(n, s);

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
        CUDA_CHECK(cudaDeviceSynchronize());

        CUSPARSE_CHECK(cusparseSpMM_bufferSize(handles.cusparse, op, op, &alpha,
                                               A, X, &beta, B, compute_type,
                                               alg, &buffer_size));

        CUDA_CHECK(cudaMallocAsync(&scratch_d, buffer_size, stream));

        CUSPARSE_CHECK(cusparseSpMM_preprocess(handles.cusparse, op, op, &alpha,
                                               A, X, &beta, B, compute_type,
                                               alg, scratch_d));

        CUSPARSE_CHECK(cusparseSpMM(handles.cusparse, op, op, &alpha, A, X,
                                    &beta, B, compute_type, alg, scratch_d));

        CUDA_CHECK(cudaFreeAsync(scratch_d, stream));
    }

    {
        // [w, sigma] = qr(R, 'econ')
        qr_factorization(handles.cusolver, handles.cusolver_params, d.w,
                         d.sigma, n, s, d_R);
    }

    {
        // s = w
        CUDA_CHECK(cudaMemcpyAsync(d.s, d.w, sizeof(float) * n * s,
                                   cudaMemcpyDeviceToDevice, stream));
    }

    int iterations;
    for (iterations = 1; iterations < max_iterations; ++iterations) {
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

            CUSPARSE_CHECK(cusparseSpMM_preprocess(
                handles.cusparse, op, op, &alpha, A, s_mat, &beta, temp,
                compute_type, alg, scratch_d));

            CUSPARSE_CHECK(cusparseSpMM(handles.cusparse, op, op, &alpha, A,
                                        s_mat, &beta, temp, compute_type, alg,
                                        scratch_d));

            constexpr cublasOperation_t op_t = CUBLAS_OP_T;
            constexpr cublasOperation_t op_n = CUBLAS_OP_N;
            CUBLAS_CHECK(cublasSgemm_v2(handles.cublas, op_t, op_n, s, s, n,
                                        &alpha, d.s, n, d_temp, n, &beta, d.xi,
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
            CUBLAS_CHECK(cublasSgemm_v2(
                handles.cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, s, s, &alpha_2,
                d.s, n, d.temp, n, &beta_2, static_cast<float *>(d_X), n));
        }
    }

    return iterations;
}
