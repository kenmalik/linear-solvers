#include "dr_bcg/dense.h"
#include "dr_bcg/device_buffer.h"
#include "dr_bcg/helper.h"
#include "dr_bcg/internal/math.h"

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <nvtx3/nvtx3.hpp>

#include <iostream>

namespace {
struct Handles {
    cublasHandle_t cublas;
    cusolverDnHandle_t cusolver;
    cusolverDnParams_t cusolver_params;

    Handles() {
        CUBLAS_CHECK(cublasCreate_v2(&cublas));
        CUSOLVER_CHECK(cusolverDnCreate(&cusolver));
        CUSOLVER_CHECK(cusolverDnCreateParams(&cusolver_params));
    }

    ~Handles() {
        CUBLAS_CHECK(cublasDestroy_v2(cublas));
        CUSOLVER_CHECK(cusolverDnDestroy(cusolver));
        CUSOLVER_CHECK(cusolverDnDestroyParams(cusolver_params));
    }

    void set_stream(cudaStream_t stream) {
        CUBLAS_CHECK(cublasSetStream_v2(cublas, stream));
        CUSOLVER_CHECK(cusolverDnSetStream(cusolver, stream));
    }
};
} // namespace

int dr_bcg::dr_bcg(float *d_A, float *d_X, float *d_B, std::int64_t n,
                   std::int64_t s, float tolerance,
                   std::int64_t max_iterations) {
    NVTX3_FUNC_RANGE();

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    Handles h;
    h.set_stream(stream);

    DeviceBuffer<float> d(n, s);

    float *d_R = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_R, sizeof(float) * n * s, stream));

    float B1_norm = 0.0f;
    {
        // Get B's first column norm for use in relative residual norm
        // calculation
        constexpr int incx = 1;
        CUBLAS_CHECK(cublasSnrm2_v2(h.cublas, n, d_B, incx, &B1_norm));
    }

    {
        // R = B - A * X
        constexpr cublasOperation_t op = CUBLAS_OP_N;
        constexpr float alpha = -1.0f;
        constexpr float beta = 1.0f;

        CUDA_CHECK(cudaMemcpyAsync(d_R, d_B, sizeof(float) * n * s,
                                   cudaMemcpyDeviceToDevice, stream));

        CUBLAS_CHECK(cublasSgemm_v2(h.cublas, op, op, n, s, n, &alpha, d_A, n,
                                    d_X, n, &beta, d_R, n));
    }

    {
        // [w, sigma] = qr(R,'econ')
        qr_factorization(h.cusolver, h.cusolver_params, d.w, d.sigma, n, s,
                         d_R);
    }

    CUDA_CHECK(cudaFreeAsync(d_R, stream));

    {
        // s = w
        CUDA_CHECK(cudaMemcpyAsync(d.s, d.w, sizeof(float) * n * s,
                                   cudaMemcpyDeviceToDevice, stream));
    }

    int iterations = 0;
    while (iterations < max_iterations) {
        nvtx3::scoped_range loop{"iteration"};
        ++iterations;

        {
            // xi = (s' * A * s)^-1
            constexpr float alpha = 1.0f;
            constexpr float beta = 0.0f;

            // temp = A * s
            CUBLAS_CHECK(cublasSgemm_v2(h.cublas, CUBLAS_OP_N, CUBLAS_OP_N, n,
                                        s, n, &alpha, d_A, n, d.s, n, &beta,
                                        d.temp, n));

            // xi = s' * temp
            CUBLAS_CHECK(cublasSgemm_v2(h.cublas, CUBLAS_OP_T, CUBLAS_OP_N, s,
                                        s, n, &alpha, d.s, n, d.temp, n, &beta,
                                        d.xi, s));

            invert_square_matrix(h.cusolver, h.cusolver_params, d.xi, s);
        }

        {
            // X = X + s * xi * sigma
            constexpr cublasOperation_t op = CUBLAS_OP_N;

            // temp = s * xi
            constexpr float alpha_temp = 1.0f;
            constexpr float beta_temp = 0.0f;
            CUBLAS_CHECK(cublasSgemm_v2(h.cublas, op, op, n, s, s, &alpha_temp,
                                        d.s, n, d.xi, s, &beta_temp, d.temp,
                                        n));

            // X = X + temp * sigma
            constexpr float alpha_X = 1.0f;
            constexpr float beta_X = 1.0f;
            CUBLAS_CHECK(cublasSgemm_v2(h.cublas, op, op, n, s, s, &alpha_X,
                                        d.temp, n, d.sigma, s, &beta_X, d_X,
                                        n));
        }

        float relative_residual_norm = 0;
        {
            // rrn = norm(B(:,1) - A * X(:,1)) / norm(B(:,1))
            constexpr float alpha = -1.0f;
            constexpr float beta = 1.0f;
            constexpr int inc = 1;

            CUDA_CHECK(cudaMemcpyAsync(d.temp, d_B, sizeof(float) * n,
                                       cudaMemcpyDeviceToDevice, stream));

            CUBLAS_CHECK(cublasSgemv_v2(h.cublas, CUBLAS_OP_N, n, n, &alpha,
                                        d_A, n, d_X, inc, &beta, d.temp, inc));
            CUBLAS_CHECK(cublasSnrm2_v2(h.cublas, n, d.temp, inc,
                                        &relative_residual_norm));

            relative_residual_norm /= B1_norm;
        }

        std::cerr << iterations << ' ' << relative_residual_norm << std::endl;
        if (relative_residual_norm < tolerance) {
            break;
        }

        {
            // [w, zeta] = qr(w - A * s * xi, 'econ')
            constexpr cublasOperation_t op = CUBLAS_OP_N;

            // temp = s * xi
            constexpr float alpha_temp = 1.0f;
            constexpr float beta_temp = 0.0f;
            CUBLAS_CHECK(cublasSgemm_v2(h.cublas, op, op, n, s, s, &alpha_temp,
                                        d.s, n, d.xi, s, &beta_temp, d.temp,
                                        n));

            // w = w - A * temp
            constexpr float alpha_w = -1.0f;
            constexpr float beta_w = 1.0f;
            CUBLAS_CHECK(cublasSgemm_v2(h.cublas, op, op, n, s, n, &alpha_w,
                                        d_A, n, d.temp, n, &beta_w, d.w, n));

            qr_factorization(h.cusolver, h.cusolver_params, d.w, d.zeta, n, s,
                             d.w);
        }

        {
            // s = w + s * zeta'
            constexpr cublasSideMode_t side = CUBLAS_SIDE_RIGHT;
            constexpr cublasFillMode_t fill = CUBLAS_FILL_MODE_UPPER;
            constexpr cublasOperation_t op_zeta = CUBLAS_OP_T;
            constexpr cublasDiagType_t diag = CUBLAS_DIAG_NON_UNIT;
            constexpr float alpha = 1.0f;

            // temp = s * zeta'
            CUBLAS_CHECK(cublasStrmm_v2(h.cublas, side, fill, op_zeta, diag, n,
                                        s, &alpha, d.zeta, s, d.s, n, d.temp,
                                        n));

            // s = w + temp
            constexpr cublasOperation_t op_s = CUBLAS_OP_N;
            constexpr float multiplier = 1.0f;
            CUBLAS_CHECK(cublasSgeam(h.cublas, op_s, op_s, n, s, &multiplier,
                                     d.w, n, &multiplier, d.temp, n, d.s, n));
        }

        {
            // sigma = zeta * sigma
            constexpr cublasSideMode_t side = CUBLAS_SIDE_LEFT;
            constexpr cublasFillMode_t fill = CUBLAS_FILL_MODE_UPPER;
            constexpr cublasOperation_t op_zeta = CUBLAS_OP_N;
            constexpr cublasDiagType_t diag = CUBLAS_DIAG_NON_UNIT;
            constexpr float alpha = 1.0f;

            CUBLAS_CHECK(cublasStrmm_v2(h.cublas, side, fill, op_zeta, diag, s,
                                        s, &alpha, d.zeta, s, d.sigma, s,
                                        d.sigma, s));
        }
    }

    return iterations;
}

int dr_bcg::dr_bcg(double *d_A, double *d_X, double *d_B, std::int64_t n,
                   std::int64_t s, double tolerance,
                   std::int64_t max_iterations) {
    NVTX3_FUNC_RANGE();

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    Handles h;
    h.set_stream(stream);

    DeviceBuffer<double> d(n, s);

    double *d_R = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_R, sizeof(double) * n * s, stream));

    double B1_norm = 0.0;
    {
        // Get B's first column norm for use in relative residual norm
        // calculation
        constexpr int incx = 1;
        CUBLAS_CHECK(cublasDnrm2_v2(h.cublas, n, d_B, incx, &B1_norm));
    }

    {
        // R = B - A * X
        constexpr cublasOperation_t op = CUBLAS_OP_N;
        constexpr double alpha = -1.0;
        constexpr double beta = 1.0;

        CUDA_CHECK(cudaMemcpyAsync(d_R, d_B, sizeof(double) * n * s,
                                   cudaMemcpyDeviceToDevice, stream));

        CUBLAS_CHECK(cublasDgemm_v2(h.cublas, op, op, n, s, n, &alpha, d_A, n,
                                    d_X, n, &beta, d_R, n));
    }

    {
        // [w, sigma] = qr(R,'econ')
        qr_factorization(h.cusolver, h.cusolver_params, d.w, d.sigma, n, s,
                         d_R);
    }

    CUDA_CHECK(cudaFreeAsync(d_R, stream));

    {
        // s = w
        CUDA_CHECK(cudaMemcpyAsync(d.s, d.w, sizeof(double) * n * s,
                                   cudaMemcpyDeviceToDevice, stream));
    }

    int iterations = 0;
    while (iterations < max_iterations) {
        nvtx3::scoped_range loop{"iteration"};
        ++iterations;

        {
            // xi = (s' * A * s)^-1
            constexpr double alpha = 1.0;
            constexpr double beta = 0.0;

            // temp = A * s
            CUBLAS_CHECK(cublasDgemm_v2(h.cublas, CUBLAS_OP_N, CUBLAS_OP_N, n,
                                        s, n, &alpha, d_A, n, d.s, n, &beta,
                                        d.temp, n));

            // xi = s' * temp
            CUBLAS_CHECK(cublasDgemm_v2(h.cublas, CUBLAS_OP_T, CUBLAS_OP_N, s,
                                        s, n, &alpha, d.s, n, d.temp, n, &beta,
                                        d.xi, s));

            invert_square_matrix(h.cusolver, h.cusolver_params, d.xi, s);
        }

        {
            // X = X + s * xi * sigma
            constexpr cublasOperation_t op = CUBLAS_OP_N;

            // temp = s * xi
            constexpr double alpha_temp = 1.0;
            constexpr double beta_temp = 0.0;
            CUBLAS_CHECK(cublasDgemm_v2(h.cublas, op, op, n, s, s, &alpha_temp,
                                        d.s, n, d.xi, s, &beta_temp, d.temp,
                                        n));

            // X = X + temp * sigma
            constexpr double alpha_X = 1.0;
            constexpr double beta_X = 1.0;
            CUBLAS_CHECK(cublasDgemm_v2(h.cublas, op, op, n, s, s, &alpha_X,
                                        d.temp, n, d.sigma, s, &beta_X, d_X,
                                        n));
        }

        double relative_residual_norm = 0;
        {
            // rrn = norm(B(:,1) - A * X(:,1)) / norm(B(:,1))
            constexpr double alpha = -1.0;
            constexpr double beta = 1.0;
            constexpr int inc = 1;

            CUDA_CHECK(cudaMemcpyAsync(d.temp, d_B, sizeof(double) * n,
                                       cudaMemcpyDeviceToDevice, stream));

            CUBLAS_CHECK(cublasDgemv_v2(h.cublas, CUBLAS_OP_N, n, n, &alpha,
                                        d_A, n, d_X, inc, &beta, d.temp, inc));
            CUBLAS_CHECK(cublasDnrm2_v2(h.cublas, n, d.temp, inc,
                                        &relative_residual_norm));

            relative_residual_norm /= B1_norm;
        }

        std::cerr << iterations << ' ' << relative_residual_norm << std::endl;
        if (relative_residual_norm < tolerance) {
            break;
        }

        {
            // [w, zeta] = qr(w - A * s * xi, 'econ')
            constexpr cublasOperation_t op = CUBLAS_OP_N;

            // temp = s * xi
            constexpr double alpha_temp = 1.0;
            constexpr double beta_temp = 0.0;
            CUBLAS_CHECK(cublasDgemm_v2(h.cublas, op, op, n, s, s, &alpha_temp,
                                        d.s, n, d.xi, s, &beta_temp, d.temp,
                                        n));

            // w = w - A * temp
            constexpr double alpha_w = -1.0;
            constexpr double beta_w = 1.0;
            CUBLAS_CHECK(cublasDgemm_v2(h.cublas, op, op, n, s, n, &alpha_w,
                                        d_A, n, d.temp, n, &beta_w, d.w, n));

            qr_factorization(h.cusolver, h.cusolver_params, d.w, d.zeta, n, s,
                             d.w);
        }

        {
            // s = w + s * zeta'
            constexpr cublasSideMode_t side = CUBLAS_SIDE_RIGHT;
            constexpr cublasFillMode_t fill = CUBLAS_FILL_MODE_UPPER;
            constexpr cublasOperation_t op_zeta = CUBLAS_OP_T;
            constexpr cublasDiagType_t diag = CUBLAS_DIAG_NON_UNIT;
            constexpr double alpha = 1.0;

            // temp = s * zeta'
            CUBLAS_CHECK(cublasDtrmm_v2(h.cublas, side, fill, op_zeta, diag, n,
                                        s, &alpha, d.zeta, s, d.s, n, d.temp,
                                        n));

            // s = w + temp
            constexpr cublasOperation_t op_s = CUBLAS_OP_N;
            constexpr double multiplier = 1.0;
            CUBLAS_CHECK(cublasDgeam(h.cublas, op_s, op_s, n, s, &multiplier,
                                     d.w, n, &multiplier, d.temp, n, d.s, n));
        }

        {
            // sigma = zeta * sigma
            constexpr cublasSideMode_t side = CUBLAS_SIDE_LEFT;
            constexpr cublasFillMode_t fill = CUBLAS_FILL_MODE_UPPER;
            constexpr cublasOperation_t op_zeta = CUBLAS_OP_N;
            constexpr cublasDiagType_t diag = CUBLAS_DIAG_NON_UNIT;
            constexpr double alpha = 1.0;

            CUBLAS_CHECK(cublasDtrmm_v2(h.cublas, side, fill, op_zeta, diag, s,
                                        s, &alpha, d.zeta, s, d.sigma, s,
                                        d.sigma, s));
        }
    }

    return iterations;
}
