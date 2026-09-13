#pragma once

// DR-BCG with MathDx-fused xi chain.

#include "config.h"

#ifdef SOLVERS_BUILD_MATHDX

#include "cuda/detail/device_buffer.cuh"
#include "cuda/detail/initialization.cuh"
#include "cuda/detail/iteration.cuh"
#include "cuda/detail/mathdx_qr.cuh"
#include "cuda/detail/mathdx_xi.cuh"

#include "common/cuda_checks.h"
#include "common/cuda_event_timer.h"
#include "common/supported_type.h"

#include <nvtx3/nvtx3.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace cils::cuda::detail {

template <cils::detail::SupportedType T>
void apply_xi_chain(MathDxXiChain<T> &xi, DeviceBuffer<T> &d, T *d_AS, T *d_X,
                    std::int64_t n, std::int64_t s, cudaStream_t stream) {
    nvtx3::scoped_range xi_range{"xi chain: X += s*xi*sigma; U = AS*xi"};
    cils::detail::CudaTimerRange er{cils::detail::g_event_timer, "xi chain", stream};

    xi.apply(d.s, d_AS, d.sigma, d_X, d.temp, static_cast<int>(n),
             static_cast<int>(s), stream);
    xi.check(static_cast<int>(s), "xi chain", stream);
}

} // namespace cils::cuda::detail

namespace cils::cuda {

// Fused Cholesky QR policy for use in DR-BCG configuration
//
template <cils::detail::SupportedType T>
class MathDxCholeskyQr2 {
  public:
    // m: rows (problem size n), n: cols (block size s).
    MathDxCholeskyQr2(cusolverDnHandle_t & /*cusolverH*/,
                      cusolverDnParams_t & /*params*/, QrDimensions dims)
        : block_size(dims.n) {
        CUDA_CHECK(cudaMalloc(&d_G, sizeof(T) * dims.n * dims.n));
        CUDA_CHECK(cudaMalloc(&d_R1, sizeof(T) * dims.n * dims.n));
        CUDA_CHECK(cudaMalloc(&d_R2, sizeof(T) * dims.n * dims.n));
        CUDA_CHECK(cudaMalloc(&d_Rinv, sizeof(T) * dims.n * dims.n));
        CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&h_info),
                                  sizeof(int)));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&h_diag),
                                  sizeof(T) * dims.n));
    }

    MathDxCholeskyQr2(const MathDxCholeskyQr2 &) = delete;
    MathDxCholeskyQr2 &operator=(const MathDxCholeskyQr2 &) = delete;
    MathDxCholeskyQr2(MathDxCholeskyQr2 &&) = delete;
    MathDxCholeskyQr2 &operator=(MathDxCholeskyQr2 &&) = delete;

    ~MathDxCholeskyQr2() {
        if (d_G != nullptr) {
            CUDA_CHECK(cudaFree(d_G));
        }
        if (d_R1 != nullptr) {
            CUDA_CHECK(cudaFree(d_R1));
        }
        if (d_R2 != nullptr) {
            CUDA_CHECK(cudaFree(d_R2));
        }
        if (d_Rinv != nullptr) {
            CUDA_CHECK(cudaFree(d_Rinv));
        }
        if (d_info != nullptr) {
            CUDA_CHECK(cudaFree(d_info));
        }
        if (h_info != nullptr) {
            CUDA_CHECK(cudaFreeHost(h_info));
        }
        if (h_diag != nullptr) {
            CUDA_CHECK(cudaFreeHost(h_diag));
        }
    }

    void solve(T *&d_Q, T *&d_R, const T *d_A, int m, int n,
               cublasHandle_t & /*cublasH*/, cusolverDnHandle_t & /*cusolverH*/,
               cusolverDnParams_t & /*params*/, cudaStream_t &stream) {
        assert(n < m && "Expect cols to be less than rows for DR-BCG");
        cils::detail::CudaTimerRange rng{cils::detail::g_event_timer, "QR:func", stream};

        dispatch(d_Q, d_R, d_A, {.m = m, .n = n}, stream);

        // Stage R2's diagonal (final-pass factor) for the breakdown check.
        CUDA_CHECK(cudaMemcpy2DAsync(h_diag, sizeof(T), d_R2,
                                     sizeof(T) * (n + 1), sizeof(T), n,
                                     cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_info, d_info, sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
    }

    void check(int n, const char *stage, cudaStream_t stream) {
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (*h_info > 0) {
            throw std::runtime_error(
                std::string(stage) +
                ": MathDx CholeskyQR2 failed, Gram lost positive definiteness "
                "at leading minor " +
                std::to_string(*h_info) +
                " (input may be too ill-conditioned for CholQR2)");
        }
        if (*h_info < 0) {
            throw std::runtime_error(std::string(stage) + ": " +
                                     std::to_string(-*h_info) +
                                     "-th parameter is wrong in MathDx POTRF");
        }
        for (int i = 0; i < n; ++i) {
            if (h_diag[i] == T{0}) {
                throw std::runtime_error(
                    std::string(stage) +
                    ": MathDx CholeskyQR2 produced a zero diagonal in R");
            }
        }
    }

  private:
    void dispatch(T *d_Q, T *d_R, const T *d_A, QrDimensions dims,
                  cudaStream_t stream) {

        switch (dims.n) {
            using detail::launch_cholqr2;
            // NOLINTBEGIN
        case 1:
            launch_cholqr2<T, 1>(d_Q, d_R, d_A, dims.m, dims.m, d_G, d_R1, d_R2, d_Rinv, d_info, stream);
            break;
        case 2:
            launch_cholqr2<T, 2>(d_Q, d_R, d_A, dims.m, dims.m, d_G, d_R1, d_R2, d_Rinv, d_info, stream);
            break;
        case 4:
            launch_cholqr2<T, 4>(d_Q, d_R, d_A, dims.m, dims.m, d_G, d_R1, d_R2, d_Rinv, d_info, stream);
            break;
        case 8:
            launch_cholqr2<T, 8>(d_Q, d_R, d_A, dims.m, dims.m, d_G, d_R1, d_R2, d_Rinv, d_info, stream);
            break;
        case 16:
            launch_cholqr2<T, 16>(d_Q, d_R, d_A, dims.m, dims.m, d_G, d_R1, d_R2, d_Rinv, d_info, stream);
            break;
        case 32:
            launch_cholqr2<T, 32>(d_Q, d_R, d_A, dims.m, dims.m, d_G, d_R1, d_R2, d_Rinv, d_info, stream);
            break;
        case 64:
            launch_cholqr2<T, 64>(d_Q, d_R, d_A, dims.m, dims.m, d_G, d_R1, d_R2, d_Rinv, d_info, stream);
            break;
            // NOLINTEND
        default:
            throw std::runtime_error(
                "unsupported block size " + std::to_string(dims.n) +
                " for MathDx CholQR2 (supported: 1, 2, 4, 8, 16, 32, 64)");
        }
    }

    int block_size = 0;
    T *d_G = nullptr;
    T *d_R1 = nullptr;
    T *d_R2 = nullptr;
    T *d_Rinv = nullptr;
    int *d_info = nullptr;
    int *h_info = nullptr;
    T *h_diag = nullptr;
};

// Unpreconditioned (L = I) fully fused DR-BCG.
template <cils::detail::SupportedType T, QrPolicy<T> Qr = MathDxCholeskyQr2<T>>
int solve_fused(Handles &handles, cusparseSpMatDescr_t A,
                cusparseDnMatDescr_t X, cusparseDnMatDescr_t B, T tolerance,
                int max_iterations, cudaStream_t stream) {
    static_assert(std::is_same_v<T, double>, "currently only double supported");
    NVTX3_FUNC_RANGE();
    cils::detail::CudaTimerRange solve_range{cils::detail::g_event_timer, "solve", stream};

    CUBLAS_CHECK(cublasSetPointerMode(handles.cublas, CUBLAS_POINTER_MODE_DEVICE));
    handles.set_stream(stream);

    auto [n, s] = detail::get_size(B);
    detail::DeviceBuffer<T> d(n, s);

    const QrDimensions qr_dims{.m = static_cast<int>(n), .n = static_cast<int>(s)};
    Qr qr{handles.cusolver, handles.cusolver_params, qr_dims};
    detail::MathDxXiChain<T> xi{static_cast<int>(s)};

    cusparseDnMatDescr_t temp = nullptr;
    CUSPARSE_CHECK(cusparseCreateDnMat(&temp, n, s, n, d.temp, cils::detail::cuda_type<T>,
                                       CUSPARSE_ORDER_COL));

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

    detail::AsCalculator<T> As_calculator{handles.cusparse, n, s, A, s_desc, stream};

    detail::RelativeResidualNormConvergence<T> convergence{handles, A, X, B, tolerance, n, stream};

    int iterations = 0;
    while (iterations < max_iterations) {
        nvtx3::scoped_range iteration_range{"iteration"};
        cils::detail::CudaTimerRange er{cils::detail::g_event_timer, "iteration", stream};

        ++iterations;

        As_calculator.update();

        apply_xi_chain<T>(xi, d, As_calculator.As_memory(), d_X, n, s, stream);

        if (convergence.check()) {
            break;
        }

        {
            nvtx3::scoped_range w_zeta_range{"[w zeta] = QR(w - A * s * xi)"};
            cils::detail::CudaTimerRange er{cils::detail::g_event_timer, "[w zeta] = QR(w - A * s * xi}", stream};

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
template <cils::detail::SupportedType T, QrPolicy<T> Qr = MathDxCholeskyQr2<T>>
int solve_fused(Handles &handles, cusparseSpMatDescr_t A,
                cusparseDnMatDescr_t X, cusparseDnMatDescr_t B,
                cusparseSpMatDescr_t L, T tolerance, int max_iterations,
                cudaStream_t stream) {
    static_assert(std::is_same_v<T, double>, "currently only double supported");
    NVTX3_FUNC_RANGE();
    cils::detail::CudaTimerRange solve_range{cils::detail::g_event_timer, "solve", stream};

    CUBLAS_CHECK(cublasSetPointerMode(handles.cublas, CUBLAS_POINTER_MODE_DEVICE));
    handles.set_stream(stream);

    auto [n, s] = detail::get_size(B);
    detail::DeviceBuffer<T> d(n, s);

    const QrDimensions qr_dims{.m = static_cast<int>(n), .n = static_cast<int>(s)};
    Qr qr{handles.cusolver, handles.cusolver_params, qr_dims};
    detail::MathDxXiChain<T> xi{static_cast<int>(s)};

    cusparseDnMatDescr_t temp = nullptr;
    CUSPARSE_CHECK(cusparseCreateDnMat(&temp, n, s, n, d.temp, cils::detail::cuda_type<T>,
                                       CUSPARSE_ORDER_COL));
    cusparseDnMatDescr_t s_desc = nullptr;
    CUSPARSE_CHECK(cusparseCreateDnMat(&s_desc, n, s, n, d.s, cils::detail::cuda_type<T>,
                                       CUSPARSE_ORDER_COL));
    cusparseDnMatDescr_t w_desc = nullptr;
    CUSPARSE_CHECK(cusparseCreateDnMat(&w_desc, n, s, n, d.w, cils::detail::cuda_type<T>,
                                       CUSPARSE_ORDER_COL));

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

    // [w sigma] = QR(L^-1 * R) split for timing: temp = L^-1 R; QR(temp).
    {
        nvtx3::scoped_range w_sigma_initial_range{"temp = L^-1 * R"};
        cils::detail::CudaTimerRange er{cils::detail::g_event_timer, "temp = L^-1 * R", stream};

        sptri_solve<T>(handles.cusparse, temp, CUSPARSE_OPERATION_NON_TRANSPOSE,
                       L, R_calculator.R_descriptor(), spsm_nt);
    }

    {
        nvtx3::scoped_range w_sigma_initial_range{"[w sigma] = QR(temp)"};
        cils::detail::CudaTimerRange er{cils::detail::g_event_timer, "[w sigma] = QR(temp)", stream};

        qr.solve(d.w, d.sigma, d.temp, n, s, handles.cublas, handles.cusolver,
                 handles.cusolver_params, stream);
        qr.check(static_cast<int>(s), "initial orthonormalization", stream);
    }

    R_calculator.release();

    initialize_preconditioned_s(handles.cusparse, n, s, s_desc, w_desc, L, spsm_t, stream);

    detail::AsCalculator<T> As_calculator{handles.cusparse, n, s, A, s_desc, stream};

    detail::RelativeResidualNormConvergence<T> convergence{handles, A, X, B, tolerance, n, stream};

    int iterations = 0;
    while (iterations < max_iterations) {
        nvtx3::scoped_range iteration_range{"iteration"};
        cils::detail::CudaTimerRange er{cils::detail::g_event_timer, "iteration", stream};

        ++iterations;

        As_calculator.update();

        apply_xi_chain<T>(xi, d, As_calculator.As_memory(), d_X, n, s, stream);

        if (convergence.check()) {
            break;
        }

        {
            nvtx3::scoped_range w_zeta_range{
                "[w zeta] = QR(w - L^{-1} * A * s * xi)"};
            cils::detail::CudaTimerRange er{cils::detail::g_event_timer, "[w zeta] = QR(w - L^{-1} * A * s * xi)", stream};

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

} // namespace cils::cuda

#endif // SOLVERS_BUILD_MATHDX
