#pragma once

#include "dr_bcg/convergence_check.cuh"
#include "dr_bcg/device_buffer.cuh"
#include "dr_bcg/handles.cuh"
#include "dr_bcg/iteration.cuh"
#include "dr_bcg/math.h"
#include "dr_bcg/qr.cuh"

#include "common/cuda_checks.h"
#include "common/cuda_event_timer.h"
#include "common/type_info.h"

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusparse_v2.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <nvtx3/nvtx3.hpp>

// TODO: Figure out why LU workspace check was in the sigma convergence block

namespace dr_bcg::cuda {

inline std::pair<std::int64_t, std::int64_t> get_size(cusparseDnMatDescr_t mat) {
    std::int64_t n = 0;
    std::int64_t s = 0;
    std::int64_t ld = 0;
    void *vals = nullptr;
    cudaDataType_t data_type;
    cusparseOrder_t order;

    CUSPARSE_CHECK(cusparseDnMatGet(mat, &n, &s, &ld, &vals, &data_type, &order));

    return {n, s};
}

template <SupportedType T, QrPolicy<T> Qr = HouseholderQr<T>>
int solve(Handles &handles, cusparseSpMatDescr_t A, cusparseDnMatDescr_t X,
          cusparseDnMatDescr_t B, T tolerance, int max_iterations,
          cudaStream_t stream) {
    static_assert(std::is_same_v<T, double>, "currently only double supported");
    NVTX3_FUNC_RANGE();

    auto [n, s] = get_size(B);
    DeviceBuffer<T> d(n, s);

    CudaTimerRange solve_range{g_event_timer, "solve", stream};

    handles.set_stream(stream);

    Qr qr{handles.cusolver, handles.cusolver_params,
          static_cast<int>(n), static_cast<int>(s)};

    LuWorkspace<T> lu_ws;
    lu_ws.allocate(handles.cusolver, handles.cusolver_params,
                   static_cast<int>(s));

    void *d_scratch = nullptr;

    cusparseDnMatDescr_t temp;
    CUSPARSE_CHECK(cusparseCreateDnMat(&temp, n, s, n, d.temp, cuda_type<T>, CUSPARSE_ORDER_COL));

    T *d_X = nullptr;
    CUSPARSE_CHECK(cusparseDnMatGetValues(X, reinterpret_cast<void **>(&d_X)));

    CUBLAS_CHECK(cublasSetPointerMode(handles.cublas, CUBLAS_POINTER_MODE_DEVICE));
    constexpr int incx = 1;
    T *d_sigma_norm = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_sigma_norm, sizeof(T), stream));

    T *d_R = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_R, sizeof(T) * n * s, stream));
    cusparseDnMatDescr_t R;
    CUSPARSE_CHECK(
        cusparseCreateDnMat(&R, n, s, n, d_R, cuda_type<T>, CUSPARSE_ORDER_COL));

    {
        nvtx3::scoped_range R_range{"R = B - A * X"};
        CudaTimerRange er{g_event_timer, "R = B - A * X", stream};

        // R = B - A * X
        std::size_t buffer_size;
        constexpr T alpha = -1.0;
        constexpr T beta = 1.0;
        constexpr cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
        constexpr cudaDataType_t compute_type = cuda_type<T>;
        constexpr cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT;

        void *d_B_ptr = nullptr;
        CUSPARSE_CHECK(cusparseDnMatGetValues(B, &d_B_ptr));
        CUDA_CHECK(cudaMemcpyAsync(d_R, d_B_ptr, sizeof(T) * n * s,
                                   cudaMemcpyDeviceToDevice, stream));

        CUSPARSE_CHECK(cusparseSpMM_bufferSize(handles.cusparse, op, op, &alpha,
                                               A, X, &beta, B, compute_type,
                                               alg, &buffer_size));

        CUDA_CHECK(cudaMallocAsync(&d_scratch, buffer_size, stream));

        CUSPARSE_CHECK(cusparseSpMM(handles.cusparse, op, op, &alpha, A, X,
                                    &beta, R, compute_type, alg, d_scratch));

        CUDA_CHECK(cudaFreeAsync(d_scratch, stream));
    }

    {
        nvtx3::scoped_range w_sigma_initial_range{"[w sigma] = QR(L^-1 * R)"};
        CudaTimerRange er{g_event_timer, "[w sigma] = QR(L^-1 * R)", stream};

        // [w, sigma] = qr(R, 'econ')
        qr.solve(d.w, d.sigma, d_R, n, s, handles.cublas,
                 handles.cusolver, handles.cusolver_params, stream);
        qr.check(static_cast<int>(s), "initial orthonormalization", stream);
    }

    CUBLAS_CHECK(cublasDnrm2_v2(handles.cublas, s, d.sigma, incx, d_sigma_norm));
    T sigma_norm0 = 0;
    CUDA_CHECK(cudaMemcpyAsync(&sigma_norm0, d_sigma_norm, sizeof(T),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFreeAsync(d_R, stream));
    CUSPARSE_CHECK(cusparseDestroyDnMat(R));

    {
        nvtx3::scoped_range s_initial_range{"s = (L^-1)' * w"};
        CudaTimerRange er{g_event_timer, "s = (L^-1)' * w", stream};

        // s = w
        CUDA_CHECK(cudaMemcpyAsync(d.s, d.w, sizeof(T) * n * s,
                                   cudaMemcpyDeviceToDevice, stream));
    }

    cusparseDnMatDescr_t s_desc;
    CUSPARSE_CHECK(cusparseCreateDnMat(&s_desc, n, s, n, d.s, cuda_type<T>,
                                       CUSPARSE_ORDER_COL));

    cusparseDnMatDescr_t w_desc;
    CUSPARSE_CHECK(cusparseCreateDnMat(&w_desc, n, s, n, d.w, cuda_type<T>,
                                       CUSPARSE_ORDER_COL));

    {
        constexpr T alpha_pos = 1.0;
        constexpr T beta_zero = 0.0;
        constexpr T alpha_neg = -1.0;
        constexpr T beta_pos = 1.0;
        constexpr cusparseOperation_t op_nt = CUSPARSE_OPERATION_NON_TRANSPOSE;
        std::size_t buf_xi = 0;
        std::size_t buf_w_zeta = 0;
        CUSPARSE_CHECK(cusparseSpMM_bufferSize(
            handles.cusparse, op_nt, op_nt, &alpha_pos, A, s_desc, &beta_zero,
            temp, cuda_type<T>, CUSPARSE_SPMM_ALG_DEFAULT, &buf_xi));
        CUSPARSE_CHECK(cusparseSpMM_bufferSize(
            handles.cusparse, op_nt, op_nt, &alpha_neg, A, temp, &beta_pos,
            w_desc, cuda_type<T>, CUSPARSE_SPMM_ALG_DEFAULT, &buf_w_zeta));
        std::size_t scratch_size = std::max(buf_xi, buf_w_zeta);
        if (scratch_size > 0) {
            CUDA_CHECK(cudaMallocAsync(&d_scratch, scratch_size, stream));
        }
    }

    bool converged = false;
    int iterations = 0;
    while (!converged && iterations < max_iterations) {
        nvtx3::scoped_range iteration_range{"iteration"};
        CudaTimerRange er{g_event_timer, "iteration", stream};

        ++iterations;

        compute_xi(handles, A, s_desc, temp, d, lu_ws, n, s, d_scratch, stream);

        update_X(handles, d, d_X, n, s, stream);

        update_w_zeta<T, Qr>(handles, qr, A, temp, w_desc, d, n, s, d_scratch, stream);

        update_s(handles, d, n, s, stream);

        update_sigma(handles, d, s, stream);

        converged = check_convergence(handles, d, tolerance,
                                      sigma_norm0, d_sigma_norm, s, stream);
    }

    CUDA_CHECK(cudaFreeAsync(d_sigma_norm, stream));
    CUDA_CHECK(cudaFreeAsync(d_scratch, stream));
    CUSPARSE_CHECK(cusparseDestroyDnMat(s_desc));
    CUSPARSE_CHECK(cusparseDestroyDnMat(w_desc));

    return iterations;
}

template <SupportedType T, QrPolicy<T> Qr = HouseholderQr<T>>
int solve(Handles &handles, cusparseSpMatDescr_t A, cusparseDnMatDescr_t X,
          cusparseDnMatDescr_t B, cusparseSpMatDescr_t L,
          T tolerance, int max_iterations, cudaStream_t stream) {
    static_assert(std::is_same_v<T, double>, "currently only double supported");
    NVTX3_FUNC_RANGE();

    auto [n, s] = get_size(B);
    DeviceBuffer<T> d(n, s);

    CudaTimerRange solve_range{g_event_timer, "solve", stream};

    handles.set_stream(stream);

    Qr qr{handles.cusolver, handles.cusolver_params,
          static_cast<int>(n), static_cast<int>(s)};

    LuWorkspace<T> lu_ws;
    lu_ws.allocate(handles.cusolver, handles.cusolver_params,
                   static_cast<int>(s));

    void *d_scratch = nullptr;

    cusparseDnMatDescr_t temp;
    CUSPARSE_CHECK(cusparseCreateDnMat(&temp, n, s, n, d.temp, cuda_type<T>, CUSPARSE_ORDER_COL));
    cusparseDnMatDescr_t s_desc;
    CUSPARSE_CHECK(cusparseCreateDnMat(&s_desc, n, s, n, d.s, cuda_type<T>, CUSPARSE_ORDER_COL));
    cusparseDnMatDescr_t w_desc;
    CUSPARSE_CHECK(cusparseCreateDnMat(&w_desc, n, s, n, d.w, cuda_type<T>, CUSPARSE_ORDER_COL));

    SpsmCache<T> spsm_nt;
    spsm_nt.analyze(handles.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, L,
                    w_desc, temp);

    SpsmCache<T> spsm_t;
    spsm_t.analyze(handles.cusparse, CUSPARSE_OPERATION_TRANSPOSE, L, w_desc,
                   temp);

    T *d_X = nullptr;
    CUSPARSE_CHECK(cusparseDnMatGetValues(X, reinterpret_cast<void **>(&d_X)));

    CUBLAS_CHECK(cublasSetPointerMode(handles.cublas, CUBLAS_POINTER_MODE_DEVICE));
    constexpr int incx = 1;
    T *d_sigma_norm = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_sigma_norm, sizeof(T), stream));

    T *d_R = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_R, sizeof(T) * n * s, stream));
    cusparseDnMatDescr_t R;
    CUSPARSE_CHECK(
        cusparseCreateDnMat(&R, n, s, n, d_R, cuda_type<T>, CUSPARSE_ORDER_COL));

    {
        nvtx3::scoped_range R_range{"R = B - A * X"};
        CudaTimerRange er{g_event_timer, "R = B - A * X", stream};

        // R = B - A * X
        std::size_t buffer_size;
        constexpr T alpha = -1.0;
        constexpr T beta = 1.0;
        constexpr cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
        constexpr cudaDataType_t compute_type = cuda_type<T>;
        constexpr cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT;

        void *d_B_ptr = nullptr;
        CUSPARSE_CHECK(cusparseDnMatGetValues(B, &d_B_ptr));
        CUDA_CHECK(cudaMemcpyAsync(d_R, d_B_ptr, sizeof(T) * n * s,
                                   cudaMemcpyDeviceToDevice, stream));

        CUSPARSE_CHECK(cusparseSpMM_bufferSize(handles.cusparse, op, op, &alpha,
                                               A, X, &beta, B, compute_type,
                                               alg, &buffer_size));

        CUDA_CHECK(cudaMallocAsync(&d_scratch, buffer_size, stream));

        CUSPARSE_CHECK(cusparseSpMM(handles.cusparse, op, op, &alpha, A, X,
                                    &beta, R, compute_type, alg, d_scratch));

        CUDA_CHECK(cudaFreeAsync(d_scratch, stream));
    }

    // We break [w sigma] = QR(L^-1 * R) into two steps for timing purposes:
    // 1. temp = L^-1 * R
    // 2. [w sigma] = QR(temp)
    {
        nvtx3::scoped_range w_sigma_initial_range{"temp = L^-1 * R"};
        CudaTimerRange er{g_event_timer, "temp = L^-1 * R", stream};

        sptri_solve<T>(handles.cusparse, temp,
                       CUSPARSE_OPERATION_NON_TRANSPOSE, L, R, spsm_nt);
    }

    {
        nvtx3::scoped_range w_sigma_initial_range{"[w sigma] = QR(temp)"};
        CudaTimerRange er{g_event_timer, "[w sigma] = QR(temp)", stream};

        qr.solve(d.w, d.sigma, d.temp, n, s, handles.cublas,
                 handles.cusolver, handles.cusolver_params, stream);
        qr.check(static_cast<int>(s), "initial orthonormalization", stream);
    }

    CUBLAS_CHECK(cublasDnrm2_v2(handles.cublas, s, d.sigma, incx, d_sigma_norm));
    T sigma_norm0 = 0;
    CUDA_CHECK(cudaMemcpyAsync(&sigma_norm0, d_sigma_norm, sizeof(T),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFreeAsync(d_R, stream));
    CUSPARSE_CHECK(cusparseDestroyDnMat(R));

    {
        nvtx3::scoped_range s_initial_range{"s = (L^-1)' * w"};
        CudaTimerRange er{g_event_timer, "s = (L^-1)' * w", stream};

        // s = (L^-1)' * w
        CUDA_CHECK(cudaMemcpyAsync(d.s, d.w, sizeof(T) * n * s,
                                   cudaMemcpyDeviceToDevice, stream));

        sptri_solve<T>(handles.cusparse, s_desc,
                       CUSPARSE_OPERATION_TRANSPOSE, L, w_desc, spsm_t);
    }

    {
        constexpr T alpha_pos = 1.0;
        constexpr T beta_zero = 0.0;
        constexpr cusparseOperation_t op_nt = CUSPARSE_OPERATION_NON_TRANSPOSE;
        std::size_t buf_xi = 0;
        CUSPARSE_CHECK(cusparseSpMM_bufferSize(
            handles.cusparse, op_nt, op_nt, &alpha_pos, A, s_desc, &beta_zero,
            temp, cuda_type<T>, CUSPARSE_SPMM_ALG_DEFAULT, &buf_xi));
        if (buf_xi > 0) {
            CUDA_CHECK(cudaMallocAsync(&d_scratch, buf_xi, stream));
        }
    }

    bool converged = false;
    int iterations = 0;
    while (!converged && iterations < max_iterations) {
        nvtx3::scoped_range iteration_range{"iteration"};
        CudaTimerRange er{g_event_timer, "iteration", stream};

        ++iterations;

        compute_xi(handles, A, s_desc, temp, d, lu_ws, n, s, d_scratch, stream);

        update_X(handles, d, d_X, n, s, stream);

        // We break [w zeta] = QR(w - L^-1 * A * s * xi) into two steps for timing purposes:
        // 1. w = w - L^-1 * A * s * xi
        // 2. [w zeta] = QR(w)
        update_w(handles, A, s_desc, temp, L, d, spsm_nt, n, s, d_scratch, stream);
        orthonormalize_w<T, Qr>(qr, handles, d, n, s, stream);

        update_s_preconditioned(handles, temp, w_desc, L, d, spsm_t, n, s,
                                stream);

        update_sigma(handles, d, s, stream);

        converged = check_convergence(handles, d, tolerance,
                                      sigma_norm0, d_sigma_norm, s, stream);
    }

    CUDA_CHECK(cudaFreeAsync(d_sigma_norm, stream));
    CUDA_CHECK(cudaFreeAsync(d_scratch, stream));

    return iterations;
}

} // namespace dr_bcg::cuda
