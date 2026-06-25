#pragma once

// Fully fused MathDx DR-BCG solve loop (PLAN.md Stage 3): fused CholeskyQR2
// (Stage 2, MathDxCholeskyQr2) AND the fused reduced-system xi chain (Stage 3,
// MathDxXiChain). Cloned from the two dr_bcg::cuda::solve() overloads in
// cuda.cuh; the only structural changes are in the inner loop:
//
//   * AS = A*s is computed once per iteration (K1 SpMM) and held in d_AS.
//   * The Gram / explicit-LU-inverse / two-GEMM xi block + X update + the
//     second A*s SpMM are replaced by xi.apply (K2-K4): X += s*G^-1*sigma and
//     U = AS*G^-1 (= A*s*xi). The unpreconditioned w update is w -= U; the
//     preconditioned one is w -= L^-1*U (one SpSM boundary, K5).
//   * Breakdown is detected from the POTRF info inside the xi chain
//     (xi.check) and the QR (qr.check); there is no LU workspace.
//
// Everything else (DeviceBuffer, Handles, SpsmCache, sptri_solve, the s/sigma
// updates, the ||sigma(:,1)|| convergence test, and all instrumentation) is
// reused verbatim from the host-API path. Compiles only when
// SOLVERS_BUILD_MATHDX is defined.

#ifdef SOLVERS_BUILD_MATHDX

#include "dr_bcg/cuda.cuh"
#include "dr_bcg/mathdx_qr.cuh"
#include "dr_bcg/mathdx_xi.cuh"

#include <algorithm>
#include <type_traits>

namespace dr_bcg::cuda {

// Unpreconditioned (L = I) fully fused DR-BCG.
template <SupportedType T, QrPolicy<T> Qr = MathDxCholeskyQr2<T>>
int solve_fused(Handles &handles, cusparseSpMatDescr_t A,
                cusparseDnMatDescr_t X, cusparseDnMatDescr_t B, T tolerance,
                int max_iterations, cudaStream_t stream) {
    static_assert(std::is_same_v<T, double>, "currently only double supported");
    NVTX3_FUNC_RANGE();

    auto [n, s] = get_size(B);
    DeviceBuffer<T> d(n, s);

    CudaTimerRange solve_range{g_event_timer, "solve", stream};

    handles.set_stream(stream);

    Qr qr{handles.cusolver, handles.cusolver_params,
          static_cast<int>(n), static_cast<int>(s)};
    MathDxXiChain<T> xi{static_cast<int>(s)};

    void *d_scratch = nullptr;

    cusparseDnMatDescr_t temp;
    CUSPARSE_CHECK(cusparseCreateDnMat(&temp, n, s, n, d.temp, cuda_type<T>,
                                       CUSPARSE_ORDER_COL));

    T *d_AS = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_AS, sizeof(T) * n * s, stream));
    cusparseDnMatDescr_t as_desc;
    CUSPARSE_CHECK(cusparseCreateDnMat(&as_desc, n, s, n, d_AS, cuda_type<T>,
                                       CUSPARSE_ORDER_COL));

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
        CudaTimerRange er(g_event_timer, "R = B - A * X", stream);

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
        CudaTimerRange er(g_event_timer, "[w sigma] = QR(L^-1 * R)", stream);

        // [w, sigma] = qr(R, 'econ')
        qr.solve(d.w, d.sigma, d_R, n, s, handles.cublas, handles.cusolver,
                 handles.cusolver_params, stream);
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
        CudaTimerRange er(g_event_timer, "s = (L^-1)' * w", stream);

        // s = w
        CUDA_CHECK(cudaMemcpyAsync(d.s, d.w, sizeof(T) * n * s,
                                   cudaMemcpyDeviceToDevice, stream));
    }

    cusparseDnMatDescr_t s_desc;
    CUSPARSE_CHECK(cusparseCreateDnMat(&s_desc, n, s, n, d.s, cuda_type<T>,
                                       CUSPARSE_ORDER_COL));

    {
        // Scratch for the single AS = A * s SpMM (the second SpMM is gone).
        constexpr T alpha_pos = 1.0;
        constexpr T beta_zero = 0.0;
        constexpr cusparseOperation_t op_nt = CUSPARSE_OPERATION_NON_TRANSPOSE;
        std::size_t buf_as = 0;
        CUSPARSE_CHECK(cusparseSpMM_bufferSize(
            handles.cusparse, op_nt, op_nt, &alpha_pos, A, s_desc, &beta_zero,
            as_desc, cuda_type<T>, CUSPARSE_SPMM_ALG_DEFAULT, &buf_as));
        if (buf_as > 0) {
            CUDA_CHECK(cudaMallocAsync(&d_scratch, buf_as, stream));
        }
    }

    int iterations = 0;
    while (iterations < max_iterations) {
        nvtx3::scoped_range iteration_range{"iteration"};
        CudaTimerRange iteration_event_range(g_event_timer, "iteration", stream);

        ++iterations;

        {
            nvtx3::scoped_range as_range{"AS = A * s"};
            CudaTimerRange er(g_event_timer, "AS = A * s", stream);

            // AS = A * s (K1)
            constexpr T alpha = 1.0;
            constexpr T beta = 0.0;
            constexpr cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
            constexpr cudaDataType_t compute_type = cuda_type<T>;
            constexpr cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT;

            CUSPARSE_CHECK(cusparseSpMM(handles.cusparse, op, op, &alpha, A,
                                        s_desc, &beta, as_desc, compute_type,
                                        alg, d_scratch));
        }

        {
            nvtx3::scoped_range xi_range{"xi chain: X += s*xi*sigma; U = AS*xi"};
            CudaTimerRange er(g_event_timer, "xi chain", stream);

            // K2-K4: X += s * G^-1 * sigma; U(=d.temp) = AS * G^-1
            xi.apply(d.s, d_AS, d.sigma, d_X, d.temp, static_cast<int>(n),
                     static_cast<int>(s), stream);
            xi.check(static_cast<int>(s), "xi chain", stream);
        }

        {
            nvtx3::scoped_range w_zeta_range{"[w zeta] = QR(w - A * s * xi)"};
            CudaTimerRange er(g_event_timer, "[w zeta] = QR(w - A * s * xi)",
                              stream);

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

        {
            nvtx3::scoped_range s_range{"s = (L^-1)' * w + s * zeta'"};
            CudaTimerRange er(g_event_timer, "s = (L^-1)' * w + s * zeta'",
                              stream);

            // s = w + s * zeta'
            constexpr cublasSideMode_t side = CUBLAS_SIDE_RIGHT;
            constexpr cublasFillMode_t fill_mode = CUBLAS_FILL_MODE_UPPER;
            constexpr cublasDiagType_t diag_type = CUBLAS_DIAG_NON_UNIT;
            constexpr cublasOperation_t op_zeta = CUBLAS_OP_T;

            CUBLAS_CHECK(cublasDtrmm_v2(handles.cublas, side, fill_mode, op_zeta,
                                        diag_type, n, s, d.one, d.zeta, s, d.s,
                                        n, d.s, n));

            constexpr cublasOperation_t sgeam_op = CUBLAS_OP_N;
            CUBLAS_CHECK(cublasDgeam(handles.cublas, sgeam_op, sgeam_op, n, s,
                                     d.one, d.s, n, d.one, d.w, n, d.s, n));
        }

        {
            nvtx3::scoped_range sigma_range{"sigma = zeta * sigma"};
            CudaTimerRange er(g_event_timer, "sigma = zeta * sigma", stream);

            // sigma = zeta * sigma
            constexpr cublasSideMode_t side = CUBLAS_SIDE_LEFT;
            constexpr cublasFillMode_t fill_mode = CUBLAS_FILL_MODE_UPPER;
            constexpr cublasDiagType_t diag_type = CUBLAS_DIAG_NON_UNIT;
            constexpr cublasOperation_t op_zeta = CUBLAS_OP_N;

            CUBLAS_CHECK(cublasDtrmm_v2(handles.cublas, side, fill_mode, op_zeta,
                                        diag_type, s, s, d.one, d.zeta, s,
                                        d.sigma, s, d.sigma, s));
        }

        {
            nvtx3::scoped_range sigma_norm_range{"||sigma(:,1)||"};
            CudaTimerRange er(g_event_timer, "||sigma(:,1)||", stream);

            CUBLAS_CHECK(
                cublasDnrm2_v2(handles.cublas, s, d.sigma, incx, d_sigma_norm));
            T sigma_norm = 0;
            CUDA_CHECK(cudaMemcpyAsync(&sigma_norm, d_sigma_norm, sizeof(T),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            LOG_TRACE(sigma_norm / sigma_norm0);
            if (sigma_norm / sigma_norm0 < tolerance)
                break;
        }
    }

    CUDA_CHECK(cudaFreeAsync(d_sigma_norm, stream));
    CUDA_CHECK(cudaFreeAsync(d_scratch, stream));
    CUDA_CHECK(cudaFreeAsync(d_AS, stream));
    CUSPARSE_CHECK(cusparseDestroyDnMat(as_desc));
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

    auto [n, s] = get_size(B);
    DeviceBuffer<T> d(n, s);

    CudaTimerRange solve_range{g_event_timer, "solve", stream};

    handles.set_stream(stream);

    Qr qr{handles.cusolver, handles.cusolver_params,
          static_cast<int>(n), static_cast<int>(s)};
    MathDxXiChain<T> xi{static_cast<int>(s)};

    void *d_scratch = nullptr;

    cusparseDnMatDescr_t temp;
    CUSPARSE_CHECK(cusparseCreateDnMat(&temp, n, s, n, d.temp, cuda_type<T>,
                                       CUSPARSE_ORDER_COL));
    cusparseDnMatDescr_t s_desc;
    CUSPARSE_CHECK(cusparseCreateDnMat(&s_desc, n, s, n, d.s, cuda_type<T>,
                                       CUSPARSE_ORDER_COL));
    cusparseDnMatDescr_t w_desc;
    CUSPARSE_CHECK(cusparseCreateDnMat(&w_desc, n, s, n, d.w, cuda_type<T>,
                                       CUSPARSE_ORDER_COL));

    T *d_AS = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_AS, sizeof(T) * n * s, stream));
    cusparseDnMatDescr_t as_desc;
    CUSPARSE_CHECK(cusparseCreateDnMat(&as_desc, n, s, n, d_AS, cuda_type<T>,
                                       CUSPARSE_ORDER_COL));

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
        CudaTimerRange er(g_event_timer, "R = B - A * X", stream);

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

    // [w sigma] = QR(L^-1 * R) split for timing: temp = L^-1 R; QR(temp).
    {
        nvtx3::scoped_range w_sigma_initial_range{"temp = L^-1 * R"};
        CudaTimerRange er(g_event_timer, "temp = L^-1 * R", stream);

        sptri_solve<T>(handles.cusparse, temp, CUSPARSE_OPERATION_NON_TRANSPOSE,
                       L, R, spsm_nt);
    }

    {
        nvtx3::scoped_range w_sigma_initial_range{"[w sigma] = QR(temp)"};
        CudaTimerRange er(g_event_timer, "[w sigma] = QR(temp)", stream);

        qr.solve(d.w, d.sigma, d.temp, n, s, handles.cublas, handles.cusolver,
                 handles.cusolver_params, stream);
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
        CudaTimerRange er(g_event_timer, "s = (L^-1)' * w", stream);

        // s = (L^-1)' * w
        CUDA_CHECK(cudaMemcpyAsync(d.s, d.w, sizeof(T) * n * s,
                                   cudaMemcpyDeviceToDevice, stream));

        sptri_solve<T>(handles.cusparse, s_desc, CUSPARSE_OPERATION_TRANSPOSE, L,
                       w_desc, spsm_t);
    }

    {
        // Scratch for the single AS = A * s SpMM.
        constexpr T alpha_pos = 1.0;
        constexpr T beta_zero = 0.0;
        constexpr cusparseOperation_t op_nt = CUSPARSE_OPERATION_NON_TRANSPOSE;
        std::size_t buf_as = 0;
        CUSPARSE_CHECK(cusparseSpMM_bufferSize(
            handles.cusparse, op_nt, op_nt, &alpha_pos, A, s_desc, &beta_zero,
            as_desc, cuda_type<T>, CUSPARSE_SPMM_ALG_DEFAULT, &buf_as));
        if (buf_as > 0) {
            CUDA_CHECK(cudaMallocAsync(&d_scratch, buf_as, stream));
        }
    }

    int iterations = 0;
    while (iterations < max_iterations) {
        nvtx3::scoped_range iteration_range{"iteration"};
        CudaTimerRange iteration_event_range(g_event_timer, "iteration", stream);

        ++iterations;

        {
            nvtx3::scoped_range as_range{"AS = A * s"};
            CudaTimerRange er(g_event_timer, "AS = A * s", stream);

            // AS = A * s (K1)
            constexpr T alpha = 1.0;
            constexpr T beta = 0.0;
            constexpr cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
            constexpr cudaDataType_t compute_type = cuda_type<T>;
            constexpr cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT;

            CUSPARSE_CHECK(cusparseSpMM(handles.cusparse, op, op, &alpha, A,
                                        s_desc, &beta, as_desc, compute_type,
                                        alg, d_scratch));
        }

        {
            nvtx3::scoped_range xi_range{"xi chain: X += s*xi*sigma; U = AS*xi"};
            CudaTimerRange er(g_event_timer, "xi chain", stream);

            // K2-K4: X += s * G^-1 * sigma; U(=d.temp) = AS * G^-1
            xi.apply(d.s, d_AS, d.sigma, d_X, d.temp, static_cast<int>(n),
                     static_cast<int>(s), stream);
            xi.check(static_cast<int>(s), "xi chain", stream);
        }

        {
            nvtx3::scoped_range w_zeta_range{
                "[w zeta] = QR(w - L^{-1} * A * s * xi)"};
            CudaTimerRange er(g_event_timer,
                              "[w zeta] = QR(w - L^{-1} * A * s * xi)", stream);

            // V = L^-1 * U (in place in temp); K5 sparse boundary
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

        {
            nvtx3::scoped_range s_range{"s = (L^-1)' * w + s * zeta'"};
            CudaTimerRange er(g_event_timer, "s = (L^-1)' * w + s * zeta'",
                              stream);

            // s = (L^-1)' * w + s * zeta'
            constexpr cublasSideMode_t side = CUBLAS_SIDE_RIGHT;
            constexpr cublasFillMode_t fill_mode = CUBLAS_FILL_MODE_UPPER;
            constexpr cublasDiagType_t diag_type = CUBLAS_DIAG_NON_UNIT;
            constexpr cublasOperation_t op_zeta = CUBLAS_OP_T;

            CUBLAS_CHECK(cublasDtrmm_v2(handles.cublas, side, fill_mode, op_zeta,
                                        diag_type, n, s, d.one, d.zeta, s, d.s,
                                        n, d.s, n));

            sptri_solve<T>(handles.cusparse, temp, CUSPARSE_OPERATION_TRANSPOSE,
                           L, w_desc, spsm_t);

            constexpr cublasOperation_t sgeam_op = CUBLAS_OP_N;
            CUBLAS_CHECK(cublasDgeam(handles.cublas, sgeam_op, sgeam_op, n, s,
                                     d.one, d.s, n, d.one, d.temp, n, d.s, n));
        }

        {
            nvtx3::scoped_range sigma_range{"sigma = zeta * sigma"};
            CudaTimerRange er(g_event_timer, "sigma = zeta * sigma", stream);

            // sigma = zeta * sigma
            constexpr cublasSideMode_t side = CUBLAS_SIDE_LEFT;
            constexpr cublasFillMode_t fill_mode = CUBLAS_FILL_MODE_UPPER;
            constexpr cublasDiagType_t diag_type = CUBLAS_DIAG_NON_UNIT;
            constexpr cublasOperation_t op_zeta = CUBLAS_OP_N;

            CUBLAS_CHECK(cublasDtrmm_v2(handles.cublas, side, fill_mode, op_zeta,
                                        diag_type, s, s, d.one, d.zeta, s,
                                        d.sigma, s, d.sigma, s));
        }

        {
            nvtx3::scoped_range sigma_norm_range{"||sigma(:,1)||"};
            CudaTimerRange er(g_event_timer, "||sigma(:,1)||", stream);

            CUBLAS_CHECK(
                cublasDnrm2_v2(handles.cublas, s, d.sigma, incx, d_sigma_norm));
            T sigma_norm = 0;
            CUDA_CHECK(cudaMemcpyAsync(&sigma_norm, d_sigma_norm, sizeof(T),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            LOG_TRACE(sigma_norm / sigma_norm0);
            if (sigma_norm / sigma_norm0 < tolerance)
                break;
        }
    }

    CUDA_CHECK(cudaFreeAsync(d_sigma_norm, stream));
    CUDA_CHECK(cudaFreeAsync(d_scratch, stream));
    CUDA_CHECK(cudaFreeAsync(d_AS, stream));
    CUSPARSE_CHECK(cusparseDestroyDnMat(as_desc));
    CUSPARSE_CHECK(cusparseDestroyDnMat(s_desc));
    CUSPARSE_CHECK(cusparseDestroyDnMat(w_desc));
    CUSPARSE_CHECK(cusparseDestroyDnMat(temp));

    return iterations;
}

} // namespace dr_bcg::cuda

#endif // SOLVERS_BUILD_MATHDX
