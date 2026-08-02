#pragma once

#include "cuda/detail/device_buffer.cuh"
#include "cuda/detail/initialization.cuh"
#include "cuda/detail/iteration.cuh"
#include "cuda/detail/math.cuh"
#include "cuda/handles.cuh"
#include "cuda/qr.cuh"

#include "common/cuda_checks.h"
#include "common/cuda_event_timer.h"
#include "common/cuda_type.cuh"

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusparse_v2.h>
#include <nvtx3/nvtx3.hpp>

#include <algorithm>

// TODO: Figure out why LU workspace check was in the sigma convergence block

namespace cils::cuda {

template <cils::detail::SupportedType T, QrPolicy<T> Qr = HouseholderQr<T>>
int solve(Handles &handles, cusparseSpMatDescr_t A, cusparseDnMatDescr_t X,
          cusparseDnMatDescr_t B, T tolerance, int max_iterations, cudaStream_t stream) {
    NVTX3_FUNC_RANGE();
    cils::detail::CudaTimerRange solve_range{cils::detail::g_event_timer, "solve", stream};

    CUBLAS_CHECK(cublasSetPointerMode(handles.cublas, CUBLAS_POINTER_MODE_DEVICE));
    handles.set_stream(stream);

    auto [n, s] = detail::get_size(B);
    detail::DeviceBuffer<T> d(n, s);

    const QrDimensions qr_dims{.m = static_cast<int>(n), .n = static_cast<int>(s)};
    Qr qr{handles.cusolver, handles.cusolver_params, qr_dims};

    detail::LuWorkspace<T> lu_ws;
    lu_ws.allocate(handles.cusolver, handles.cusolver_params,
                   static_cast<int>(s));

    void *d_scratch = nullptr;

    cusparseDnMatDescr_t temp = nullptr;
    CUSPARSE_CHECK(cusparseCreateDnMat(&temp, n, s, n, d.temp, cils::detail::cuda_type<T>, CUSPARSE_ORDER_COL));

    T *d_X = nullptr;
    CUSPARSE_CHECK(cusparseDnMatGetValues(X, reinterpret_cast<void **>(&d_X)));

    detail::RCalculator<T> R_calculator{handles.cusparse, n, s, stream};
    R_calculator.calculate(B, A, X);
    {
        nvtx3::scoped_range w_sigma_initial_range{"[w sigma] = QR(R)"};
        cils::detail::CudaTimerRange er{cils::detail::g_event_timer, "[w sigma] = QR(R)", stream};

        // [w, sigma] = qr(R, 'econ')
        qr.solve(d.w, d.sigma, R_calculator.R_memory(), n, s, handles.cublas,
                 handles.cusolver, handles.cusolver_params, stream);
        qr.check(static_cast<int>(s), "initial orthonormalization", stream);
    }
    R_calculator.release();

    {
        nvtx3::scoped_range s_initial_range{"s = w"};
        cils::detail::CudaTimerRange er{cils::detail::g_event_timer, "s = w", stream};

        // s = w
        CUDA_CHECK(cudaMemcpyAsync(d.s, d.w, sizeof(T) * n * s,
                                   cudaMemcpyDeviceToDevice, stream));
    }

    cusparseDnMatDescr_t s_desc = nullptr;
    CUSPARSE_CHECK(cusparseCreateDnMat(&s_desc, n, s, n, d.s, cils::detail::cuda_type<T>,
                                       CUSPARSE_ORDER_COL));

    cusparseDnMatDescr_t w_desc = nullptr;
    CUSPARSE_CHECK(cusparseCreateDnMat(&w_desc, n, s, n, d.w, cils::detail::cuda_type<T>,
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
            temp, cils::detail::cuda_type<T>, CUSPARSE_SPMM_ALG_DEFAULT, &buf_xi));
        CUSPARSE_CHECK(cusparseSpMM_bufferSize(
            handles.cusparse, op_nt, op_nt, &alpha_neg, A, temp, &beta_pos,
            w_desc, cils::detail::cuda_type<T>, CUSPARSE_SPMM_ALG_DEFAULT, &buf_w_zeta));
        std::size_t scratch_size = std::max(buf_xi, buf_w_zeta);
        if (scratch_size > 0) {
            CUDA_CHECK(cudaMallocAsync(&d_scratch, scratch_size, stream));
        }
    }

    detail::RelativeResidualNormConvergence<T> convergence{handles, A, X, B, tolerance, n, stream};

    int iterations = 0;
    while (iterations < max_iterations) {
        nvtx3::scoped_range iteration_range{"iteration"};
        cils::detail::CudaTimerRange er{cils::detail::g_event_timer, "iteration", stream};

        ++iterations;

        compute_xi(handles, A, s_desc, temp, d, lu_ws, n, s, d_scratch, stream);

        update_X(handles, d, d_X, n, s, stream);

        if (convergence.check()) {
            break;
        }

        update_w_zeta<T, Qr>(handles, qr, A, temp, w_desc, d, n, s, d_scratch, stream);

        update_s(handles, d, n, s, stream);

        update_sigma(handles, d, s, stream);
    }

    CUDA_CHECK(cudaFreeAsync(d_scratch, stream));
    CUSPARSE_CHECK(cusparseDestroyDnMat(s_desc));
    CUSPARSE_CHECK(cusparseDestroyDnMat(w_desc));

    return iterations;
}

template <cils::detail::SupportedType T, QrPolicy<T> Qr = HouseholderQr<T>>
int solve(Handles &handles, cusparseSpMatDescr_t A, cusparseDnMatDescr_t X,
          cusparseDnMatDescr_t B, cusparseSpMatDescr_t L,
          T tolerance, int max_iterations, cudaStream_t stream) {
    NVTX3_FUNC_RANGE();
    cils::detail::CudaTimerRange solve_range{cils::detail::g_event_timer, "solve", stream};

    CUBLAS_CHECK(cublasSetPointerMode(handles.cublas, CUBLAS_POINTER_MODE_DEVICE));
    handles.set_stream(stream);

    auto [n, s] = detail::get_size(B);
    detail::DeviceBuffer<T> d(n, s);

    const QrDimensions qr_dims{.m = static_cast<int>(n), .n = static_cast<int>(s)};
    Qr qr{handles.cusolver, handles.cusolver_params, qr_dims};

    detail::LuWorkspace<T> lu_ws;
    lu_ws.allocate(handles.cusolver, handles.cusolver_params,
                   static_cast<int>(s));

    void *d_scratch = nullptr;

    cusparseDnMatDescr_t temp = nullptr;
    CUSPARSE_CHECK(cusparseCreateDnMat(&temp, n, s, n, d.temp, cils::detail::cuda_type<T>, CUSPARSE_ORDER_COL));
    cusparseDnMatDescr_t s_desc = nullptr;
    CUSPARSE_CHECK(cusparseCreateDnMat(&s_desc, n, s, n, d.s, cils::detail::cuda_type<T>, CUSPARSE_ORDER_COL));
    cusparseDnMatDescr_t w_desc = nullptr;
    CUSPARSE_CHECK(cusparseCreateDnMat(&w_desc, n, s, n, d.w, cils::detail::cuda_type<T>, CUSPARSE_ORDER_COL));

    detail::SpsmCache<T> spsm_nt;
    spsm_nt.analyze(handles.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, L,
                    w_desc, temp);

    detail::SpsmCache<T> spsm_t;
    spsm_t.analyze(handles.cusparse, CUSPARSE_OPERATION_TRANSPOSE, L, w_desc,
                   temp);

    T *d_X = nullptr;
    CUSPARSE_CHECK(cusparseDnMatGetValues(X, reinterpret_cast<void **>(&d_X)));

    detail::RCalculator<T> R_calculator{handles.cusparse, n, s, stream};
    R_calculator.calculate(B, A, X);

    // We break [w sigma] = QR(L^-1 * R) into two steps for timing purposes:
    // 1. temp = L^-1 * R
    // 2. [w sigma] = QR(temp)
    {
        nvtx3::scoped_range w_sigma_initial_range{"temp = L^-1 * R"};
        cils::detail::CudaTimerRange er{cils::detail::g_event_timer, "temp = L^-1 * R", stream};

        sptri_solve<T>(handles.cusparse, temp,
                       CUSPARSE_OPERATION_NON_TRANSPOSE, L, R_calculator.R_descriptor(), spsm_nt);
    }

    {
        nvtx3::scoped_range w_sigma_initial_range{"[w sigma] = QR(temp)"};
        cils::detail::CudaTimerRange er{cils::detail::g_event_timer, "[w sigma] = QR(temp)", stream};

        qr.solve(d.w, d.sigma, d.temp, n, s, handles.cublas,
                 handles.cusolver, handles.cusolver_params, stream);
        qr.check(static_cast<int>(s), "initial orthonormalization", stream);
    }

    R_calculator.release();

    initialize_preconditioned_s(handles.cusparse, n, s, s_desc, w_desc, L, spsm_t, stream);

    {
        constexpr T alpha_pos = 1.0;
        constexpr T beta_zero = 0.0;
        constexpr cusparseOperation_t op_nt = CUSPARSE_OPERATION_NON_TRANSPOSE;
        std::size_t buf_xi = 0;
        CUSPARSE_CHECK(cusparseSpMM_bufferSize(
            handles.cusparse, op_nt, op_nt, &alpha_pos, A, s_desc, &beta_zero,
            temp, cils::detail::cuda_type<T>, CUSPARSE_SPMM_ALG_DEFAULT, &buf_xi));
        if (buf_xi > 0) {
            CUDA_CHECK(cudaMallocAsync(&d_scratch, buf_xi, stream));
        }
    }

    detail::RelativeResidualNormConvergence<T> convergence{handles, A, X, B, tolerance, n, stream};

    int iterations = 0;
    while (iterations < max_iterations) {
        nvtx3::scoped_range iteration_range{"iteration"};
        cils::detail::CudaTimerRange er{cils::detail::g_event_timer, "iteration", stream};

        ++iterations;

        compute_xi(handles, A, s_desc, temp, d, lu_ws, n, s, d_scratch, stream);

        update_X(handles, d, d_X, n, s, stream);

        if (convergence.check()) {
            break;
        }

        // We break [w zeta] = QR(w - L^-1 * A * s * xi) into two steps for timing purposes:
        // 1. w = w - L^-1 * A * s * xi
        // 2. [w zeta] = QR(w)
        update_w(handles, A, s_desc, temp, L, d, spsm_nt, n, s, d_scratch, stream);
        orthonormalize_w<T, Qr>(qr, handles, d, n, s, stream);

        update_s_preconditioned(handles, temp, w_desc, L, d, spsm_t, n, s, stream);

        update_sigma(handles, d, s, stream);
    }

    CUDA_CHECK(cudaFreeAsync(d_scratch, stream));

    return iterations;
}

} // namespace cils::cuda
