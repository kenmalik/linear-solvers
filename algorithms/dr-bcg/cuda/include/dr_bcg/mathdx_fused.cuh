#pragma once

// DR-BCG with MathDx-fused xi chain.
//
// TODO: Consider ways to reduce code duplication from original solver implementation.

#ifdef SOLVERS_BUILD_MATHDX

#include "dr_bcg/device_buffer.cuh"
#include "dr_bcg/initialization.cuh"
#include "dr_bcg/iteration.cuh"
#include "dr_bcg/mathdx_qr.cuh"
#include "dr_bcg/mathdx_xi.cuh"

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace dr_bcg::cuda {

template <SupportedType T>
void apply_xi_chain(MathDxXiChain<T> &xi, DeviceBuffer<T> &d, T *d_AS, T *d_X,
                    std::int64_t n, std::int64_t s, cudaStream_t stream) {
    nvtx3::scoped_range xi_range{"xi chain: X += s*xi*sigma; U = AS*xi"};
    CudaTimerRange er{g_event_timer, "xi chain", stream};

    xi.apply(d.s, d_AS, d.sigma, d_X, d.temp, static_cast<int>(n),
             static_cast<int>(s), stream);
    xi.check(static_cast<int>(s), "xi chain", stream);
}

// Unpreconditioned (L = I) fully fused DR-BCG.
template <SupportedType T, QrPolicy<T> Qr = MathDxCholeskyQr2<T>>
int solve_fused(Handles &handles, cusparseSpMatDescr_t A,
                cusparseDnMatDescr_t X, cusparseDnMatDescr_t B, T tolerance,
                int max_iterations, cudaStream_t stream) {
    static_assert(std::is_same_v<T, double>, "currently only double supported");
    NVTX3_FUNC_RANGE();
    CudaTimerRange solve_range{g_event_timer, "solve", stream};

    CUBLAS_CHECK(cublasSetPointerMode(handles.cublas, CUBLAS_POINTER_MODE_DEVICE));
    handles.set_stream(stream);

    auto [n, s] = get_size(B);
    DeviceBuffer<T> d(n, s);

    Qr qr{handles.cusolver, handles.cusolver_params,
          static_cast<int>(n), static_cast<int>(s)};
    MathDxXiChain<T> xi{static_cast<int>(s)};

    cusparseDnMatDescr_t temp;
    CUSPARSE_CHECK(cusparseCreateDnMat(&temp, n, s, n, d.temp, cuda_type<T>,
                                       CUSPARSE_ORDER_COL));

    T *d_X = nullptr;
    CUSPARSE_CHECK(cusparseDnMatGetValues(X, reinterpret_cast<void **>(&d_X)));

    RCalculator<T> R_calculator{handles.cusparse, n, s, stream};
    R_calculator.calculate(B, A, X);
    {
        nvtx3::scoped_range w_sigma_initial_range{"[w sigma] = QR(R)"};
        CudaTimerRange er{g_event_timer, "[w sigma] = QR(R)", stream};

        // [w, sigma] = qr(R, 'econ')
        qr.solve(d.w, d.sigma, R_calculator.R_memory(), n, s, handles.cublas,
                 handles.cusolver, handles.cusolver_params, stream);
        qr.check(static_cast<int>(s), "initial orthonormalization", stream);
    }
    R_calculator.release();

    {
        nvtx3::scoped_range s_initial_range{"s = w"};
        CudaTimerRange er{g_event_timer, "s = w", stream};

        // s = w
        CUDA_CHECK(cudaMemcpyAsync(d.s, d.w, sizeof(T) * n * s,
                                   cudaMemcpyDeviceToDevice, stream));
    }

    cusparseDnMatDescr_t s_desc;
    CUSPARSE_CHECK(cusparseCreateDnMat(&s_desc, n, s, n, d.s, cuda_type<T>,
                                       CUSPARSE_ORDER_COL));

    AsCalculator<T> As_calculator{handles.cusparse, n, s, A, s_desc, stream};

    RelativeResidualNormConvergence<T> convergence{handles, A, X, B, tolerance, n, stream};

    int iterations = 0;
    while (iterations < max_iterations) {
        nvtx3::scoped_range iteration_range{"iteration"};
        CudaTimerRange er{g_event_timer, "iteration", stream};

        ++iterations;

        As_calculator.update();

        apply_xi_chain<T>(xi, d, As_calculator.As_memory(), d_X, n, s, stream);

        if (convergence.check()) {
            break;
        }

        {
            nvtx3::scoped_range w_zeta_range{"[w zeta] = QR(w - A * s * xi)"};
            CudaTimerRange er{g_event_timer, "[w zeta] = QR(w - A * s * xi}", stream};

            // w = w - U
            constexpr cublasOperation_t op = CUBLAS_OP_N;
            CUBLAS_CHECK(cublasDgeam(handles.cublas, op, op, n, s, d.one, d.w, n,
                                     d.neg_one, d.temp, n, d.w, n));

            // [w, zeta] = qr(w, 'econ')
            qr.solve(d.w, d.zeta, d.w, n, s, handles.cublas, handles.cusolver,
                     handles.cusolver_params, stream);
            qr.check(static_cast<int>(s), "iteration orthonormalization",
                     stream);
        }

        update_s(handles, d, n, s, stream);

        update_sigma(handles, d, s, stream);
    }

    CUSPARSE_CHECK(cusparseDestroyDnMat(s_desc));
    CUSPARSE_CHECK(cusparseDestroyDnMat(temp));

    return iterations;
}

// Preconditioned (M = L L^T) fully fused DR-BCG.
template <SupportedType T, QrPolicy<T> Qr = MathDxCholeskyQr2<T>>
int solve_fused(Handles &handles, cusparseSpMatDescr_t A,
                cusparseDnMatDescr_t X, cusparseDnMatDescr_t B,
                cusparseSpMatDescr_t L, T tolerance, int max_iterations,
                cudaStream_t stream) {
    static_assert(std::is_same_v<T, double>, "currently only double supported");
    NVTX3_FUNC_RANGE();
    CudaTimerRange solve_range{g_event_timer, "solve", stream};

    CUBLAS_CHECK(cublasSetPointerMode(handles.cublas, CUBLAS_POINTER_MODE_DEVICE));
    handles.set_stream(stream);

    auto [n, s] = get_size(B);
    DeviceBuffer<T> d(n, s);

    Qr qr{handles.cusolver, handles.cusolver_params,
          static_cast<int>(n), static_cast<int>(s)};
    MathDxXiChain<T> xi{static_cast<int>(s)};

    cusparseDnMatDescr_t temp;
    CUSPARSE_CHECK(cusparseCreateDnMat(&temp, n, s, n, d.temp, cuda_type<T>,
                                       CUSPARSE_ORDER_COL));
    cusparseDnMatDescr_t s_desc;
    CUSPARSE_CHECK(cusparseCreateDnMat(&s_desc, n, s, n, d.s, cuda_type<T>,
                                       CUSPARSE_ORDER_COL));
    cusparseDnMatDescr_t w_desc;
    CUSPARSE_CHECK(cusparseCreateDnMat(&w_desc, n, s, n, d.w, cuda_type<T>,
                                       CUSPARSE_ORDER_COL));

    SpsmCache<T> spsm_nt;
    spsm_nt.analyze(handles.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, L,
                    w_desc, temp);

    SpsmCache<T> spsm_t;
    spsm_t.analyze(handles.cusparse, CUSPARSE_OPERATION_TRANSPOSE, L, w_desc,
                   temp);

    T *d_X = nullptr;
    CUSPARSE_CHECK(cusparseDnMatGetValues(X, reinterpret_cast<void **>(&d_X)));

    RCalculator<T> R_calculator{handles.cusparse, n, s, stream};
    R_calculator.calculate(B, A, X);

    // [w sigma] = QR(L^-1 * R) split for timing: temp = L^-1 R; QR(temp).
    {
        nvtx3::scoped_range w_sigma_initial_range{"temp = L^-1 * R"};
        CudaTimerRange er{g_event_timer, "temp = L^-1 * R", stream};

        sptri_solve<T>(handles.cusparse, temp, CUSPARSE_OPERATION_NON_TRANSPOSE,
                       L, R_calculator.R_descriptor(), spsm_nt);
    }

    {
        nvtx3::scoped_range w_sigma_initial_range{"[w sigma] = QR(temp)"};
        CudaTimerRange er{g_event_timer, "[w sigma] = QR(temp)", stream};

        qr.solve(d.w, d.sigma, d.temp, n, s, handles.cublas, handles.cusolver,
                 handles.cusolver_params, stream);
        qr.check(static_cast<int>(s), "initial orthonormalization", stream);
    }

    R_calculator.release();

    initialize_preconditioned_s(handles.cusparse, n, s, s_desc, w_desc, L, spsm_t, stream);

    AsCalculator<T> As_calculator{handles.cusparse, n, s, A, s_desc, stream};

    RelativeResidualNormConvergence<T> convergence{handles, A, X, B, tolerance, n, stream};

    int iterations = 0;
    while (iterations < max_iterations) {
        nvtx3::scoped_range iteration_range{"iteration"};
        CudaTimerRange er{g_event_timer, "iteration", stream};

        ++iterations;

        As_calculator.update();

        apply_xi_chain<T>(xi, d, As_calculator.As_memory(), d_X, n, s, stream);

        if (convergence.check()) {
            break;
        }

        {
            nvtx3::scoped_range w_zeta_range{
                "[w zeta] = QR(w - L^{-1} * A * s * xi)"};
            CudaTimerRange er{g_event_timer, "[w zeta] = QR(w - L^{-1} * A * s * xi)", stream};

            // V = L^-1 * U
            sptri_solve<T>(handles.cusparse, temp,
                           CUSPARSE_OPERATION_NON_TRANSPOSE, L, temp, spsm_nt);

            // w = w - V
            constexpr cublasOperation_t op = CUBLAS_OP_N;
            CUBLAS_CHECK(cublasDgeam(handles.cublas, op, op, n, s, d.one, d.w, n,
                                     d.neg_one, d.temp, n, d.w, n));

            // [w, zeta] = qr(w, 'econ')
            qr.solve(d.w, d.zeta, d.w, n, s, handles.cublas, handles.cusolver,
                     handles.cusolver_params, stream);
            qr.check(static_cast<int>(s), "iteration orthonormalization",
                     stream);
        }

        update_s_preconditioned(handles, temp, w_desc, L, d, spsm_t, n, s, stream);

        update_sigma(handles, d, s, stream);
    }

    CUSPARSE_CHECK(cusparseDestroyDnMat(s_desc));
    CUSPARSE_CHECK(cusparseDestroyDnMat(w_desc));
    CUSPARSE_CHECK(cusparseDestroyDnMat(temp));

    return iterations;
}

} // namespace dr_bcg::cuda

#endif // SOLVERS_BUILD_MATHDX
