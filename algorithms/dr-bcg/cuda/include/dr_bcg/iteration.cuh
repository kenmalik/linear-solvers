#pragma once

#include "dr_bcg/device_buffer.cuh"
#include "dr_bcg/handles.cuh"
#include "dr_bcg/math.h"
#include "dr_bcg/qr.cuh"

#include "common/cuda_checks.h"
#include "common/cuda_event_timer.h"
#include "common/cuda_type.cuh"
#include "common/log.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <nvtx3/nvtx3.hpp>

#include <cstdint>
#include <type_traits>

namespace cils::dr_bcg::cuda {

template <cils::SupportedType T>
class RelativeResidualNormConvergence {
  public:
    [[nodiscard]] RelativeResidualNormConvergence(Handles &handles,
                                                  cusparseSpMatDescr_t A, cusparseDnMatDescr_t X, cusparseDnMatDescr_t B,
                                                  T tolerance, std::int64_t n, cudaStream_t stream) noexcept
        : handles{handles}, A{A}, tolerance{tolerance}, n{n}, stream{stream} {
        CUDA_CHECK(cudaMallocAsync(&d_norm, sizeof(T), stream));

        CUDA_CHECK(cudaMallocAsync(&d_r, sizeof(T) * n, stream));
        CUSPARSE_CHECK(cusparseCreateDnVec(&temp, n, d_r, compute_type));

        CUSPARSE_CHECK(cusparseDnMatGetValues(B, reinterpret_cast<void **>(&d_B)));
        CUSPARSE_CHECK(cusparseCreateDnVec(&B1, n, d_B, compute_type));

        T *d_X = nullptr;
        CUSPARSE_CHECK(cusparseDnMatGetValues(X, reinterpret_cast<void **>(&d_X)));
        CUSPARSE_CHECK(cusparseCreateDnVec(&X1, n, d_X, compute_type));

        // Precalculate B1 norm for conversion checks
        if constexpr (std::is_same_v<T, float>) {
            CUBLAS_CHECK(cublasSnrm2_v2(handles.cublas, n, d_B, incx, d_norm));
        } else {
            CUBLAS_CHECK(cublasDnrm2_v2(handles.cublas, n, d_B, incx, d_norm));
        }
        CUDA_CHECK(cudaMemcpyAsync(&B1_norm, d_norm, sizeof(T), cudaMemcpyDeviceToHost, stream));

        std::size_t bufsize = 0;
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(handles.cusparse, op, &alpha, A, X1, &beta, B1,
                                               compute_type, alg, &bufsize));
        CUDA_CHECK(cudaMallocAsync(&buffer, bufsize, stream));

        CUSPARSE_CHECK(cusparseSpMV_preprocess(handles.cusparse, op, &alpha, A, X1, &beta, B1,
                                               compute_type, alg, buffer));
    }

    RelativeResidualNormConvergence(const RelativeResidualNormConvergence &) = delete;
    RelativeResidualNormConvergence &operator=(const RelativeResidualNormConvergence &) = delete;
    RelativeResidualNormConvergence(RelativeResidualNormConvergence &&) = delete;
    RelativeResidualNormConvergence &operator=(RelativeResidualNormConvergence &&) = delete;

    ~RelativeResidualNormConvergence() noexcept {
        CUSPARSE_CHECK(cusparseDestroyDnVec(X1));
        CUSPARSE_CHECK(cusparseDestroyDnVec(B1));
        CUSPARSE_CHECK(cusparseDestroyDnVec(temp));

        CUDA_CHECK(cudaFreeAsync(buffer, stream));
        CUDA_CHECK(cudaFreeAsync(d_r, stream));
        CUDA_CHECK(cudaFreeAsync(d_norm, stream));
    }

    [[nodiscard]] bool check() noexcept {
        nvtx3::scoped_range rrn_range{"norm(B1 - A * X1) / norm(B1)"};
        CudaTimerRange er(g_event_timer, "norm(B1 - A * X1) / norm(B1)", stream);

        T relative_residual_norm = 0;

        // temp = norm(B(:,1) - A * X(:,1))
        CUDA_CHECK(cudaMemcpyAsync(d_r, d_B, sizeof(T) * n,
                                   cudaMemcpyDeviceToDevice, stream));

        CUSPARSE_CHECK(cusparseSpMV(handles.cusparse, op, &alpha, A, X1,
                                    &beta, temp, compute_type, alg, buffer));

        // r_norm = norm(temp)
        if constexpr (std::is_same_v<T, float>) {
            CUBLAS_CHECK(cublasSnrm2_v2(handles.cublas, n, d_r, incx, d_norm));
        } else {
            CUBLAS_CHECK(cublasDnrm2_v2(handles.cublas, n, d_r, incx, d_norm));
        }

        T residual_norm = 0;
        CUDA_CHECK(cudaMemcpyAsync(&residual_norm, d_norm, sizeof(T),
                                   cudaMemcpyDeviceToHost, stream));

        // rrn = r_norm / B1_norm
        CUDA_CHECK(cudaStreamSynchronize(stream));
        relative_residual_norm = residual_norm / B1_norm;

        cils::log(relative_residual_norm);
        return relative_residual_norm < tolerance;
    }

  private:
    static constexpr cudaDataType_t compute_type = cuda_type<T>;
    static constexpr cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    static constexpr T alpha = -1.0;
    static constexpr T beta = 1.0;
    static constexpr cusparseSpMVAlg_t alg = CUSPARSE_SPMV_ALG_DEFAULT;

    static constexpr int incx = 1;

    Handles &handles;
    const T tolerance;
    const std::int64_t n;
    const cudaStream_t stream;

    cusparseSpMatDescr_t A;
    cusparseDnVecDescr_t X1{};
    cusparseDnVecDescr_t B1{};
    cusparseDnVecDescr_t temp{};

    void *buffer = nullptr;
    T *d_B = nullptr;
    T *d_norm = nullptr;
    T *d_r = nullptr;
    T B1_norm{};
};

// xi = (s' * A * s)^-1
template <cils::SupportedType T>
void compute_xi(Handles &handles, cusparseSpMatDescr_t A,
                cusparseDnMatDescr_t s_desc, cusparseDnMatDescr_t temp,
                DeviceBuffer<T> &d, LuWorkspace<T> &lu_ws, std::int64_t n,
                std::int64_t s, void *d_scratch, cudaStream_t stream) {
    nvtx3::scoped_range xi_range{"xi = (s' * As)^-1"};
    CudaTimerRange er{g_event_timer, "xi = (s' * As)^-1", stream};

    constexpr T alpha = 1.0;
    constexpr T beta = 0.0;
    constexpr cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    constexpr cudaDataType_t compute_type = cuda_type<T>;
    constexpr cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT;

    CUSPARSE_CHECK(cusparseSpMM(handles.cusparse, op, op, &alpha, A, s_desc,
                                &beta, temp, compute_type, alg, d_scratch));

    constexpr cublasOperation_t op_t = CUBLAS_OP_T;
    constexpr cublasOperation_t op_n = CUBLAS_OP_N;

    if constexpr (std::is_same_v<T, float>) {
        CUBLAS_CHECK(cublasSgemm_v2(handles.cublas, op_t, op_n, s, s, n, d.one,
                                    d.s, n, d.temp, n, d.zero, d.xi, s));
    } else {
        CUBLAS_CHECK(cublasDgemm_v2(handles.cublas, op_t, op_n, s, s, n, d.one,
                                    d.s, n, d.temp, n, d.zero, d.xi, s));
    }

    invert_square_matrix(handles.cusolver, handles.cusolver_params, d.xi, s,
                         lu_ws, stream);
}

// X = X + s * xi * sigma
template <cils::SupportedType T>
void update_X(Handles &handles, DeviceBuffer<T> &d, T *d_X, std::int64_t n,
              std::int64_t s, cudaStream_t stream) {
    nvtx3::scoped_range X_range{"X = X + s * xi * sigma"};
    CudaTimerRange er{g_event_timer, "X = X + s * xi * sigma", stream};

    if constexpr (std::is_same_v<T, float>) {
        CUBLAS_CHECK(cublasSgemm_v2(handles.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                    s, s, s, d.one, d.xi, s, d.sigma, s, d.zero,
                                    d.temp, n));

        CUBLAS_CHECK(cublasSgemm_v2(handles.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                    n, s, s, d.one, d.s, n, d.temp, n, d.one,
                                    d_X, n));
    } else {
        CUBLAS_CHECK(cublasDgemm_v2(handles.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                    s, s, s, d.one, d.xi, s, d.sigma, s, d.zero,
                                    d.temp, n));

        CUBLAS_CHECK(cublasDgemm_v2(handles.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                    n, s, s, d.one, d.s, n, d.temp, n, d.one,
                                    d_X, n));
    }
}

// sigma = zeta * sigma
template <cils::SupportedType T>
void update_sigma(Handles &handles, DeviceBuffer<T> &d, std::int64_t s,
                  cudaStream_t stream) {
    nvtx3::scoped_range sigma_range{"sigma = zeta * sigma"};
    CudaTimerRange er{g_event_timer, "sigma = zeta * sigma", stream};

    constexpr cublasSideMode_t side = CUBLAS_SIDE_LEFT;
    constexpr cublasFillMode_t fill_mode = CUBLAS_FILL_MODE_UPPER;
    constexpr cublasDiagType_t diag_type = CUBLAS_DIAG_NON_UNIT;
    constexpr cublasOperation_t op_zeta = CUBLAS_OP_N;

    if constexpr (std::is_same_v<T, float>) {
        CUBLAS_CHECK(cublasStrmm_v2(handles.cublas, side, fill_mode, op_zeta,
                                    diag_type, s, s, d.one, d.zeta, s, d.sigma,
                                    s, d.sigma, s));
    } else {
        CUBLAS_CHECK(cublasDtrmm_v2(handles.cublas, side, fill_mode, op_zeta,
                                    diag_type, s, s, d.one, d.zeta, s, d.sigma,
                                    s, d.sigma, s));
    }
}

// [w, zeta] = qr(w - A * s * xi, 'econ')
template <cils::SupportedType T, QrPolicy<T> Qr>
void update_w_zeta(Handles &handles, Qr &qr, cusparseSpMatDescr_t A,
                   cusparseDnMatDescr_t temp, cusparseDnMatDescr_t w_desc,
                   DeviceBuffer<T> &d, std::int64_t n, std::int64_t s,
                   void *d_scratch, cudaStream_t stream) {
    nvtx3::scoped_range w_zeta_range{"[w zeta] = QR(w - A * s * xi)"};
    CudaTimerRange er{g_event_timer, "[w zeta] = QR(w - A * s * xi}", stream};

    constexpr cublasOperation_t op = CUBLAS_OP_N;

    if constexpr (std::is_same_v<T, float>) {
        CUBLAS_CHECK(cublasSgemm_v2(handles.cublas, op, op, n, s, s, d.one,
                                    d.s, n, d.xi, s, d.zero, d.temp, n));
    } else {
        CUBLAS_CHECK(cublasDgemm_v2(handles.cublas, op, op, n, s, s, d.one,
                                    d.s, n, d.xi, s, d.zero, d.temp, n));
    }

    constexpr cusparseOperation_t spmm_op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    constexpr T spmm_alpha = -1.0;
    constexpr T spmm_beta = 1.0;
    constexpr cudaDataType_t compute_type = cuda_type<T>;
    constexpr cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT;

    CUSPARSE_CHECK(cusparseSpMM(handles.cusparse, spmm_op, spmm_op, &spmm_alpha,
                                A, temp, &spmm_beta, w_desc, compute_type, alg,
                                d_scratch));

    qr.solve(d.w, d.zeta, d.w, n, s, handles.cublas, handles.cusolver,
             handles.cusolver_params, stream);
    qr.check(static_cast<int>(s), "iteration orthonormalization", stream);
}

// s = w + s * zeta'
template <cils::SupportedType T>
void update_s(Handles &handles, DeviceBuffer<T> &d, std::int64_t n,
              std::int64_t s, cudaStream_t stream) {
    nvtx3::scoped_range s_range{"s = w + s * zeta'"};
    CudaTimerRange er{g_event_timer, "s = w + s * zeta'", stream};

    constexpr cublasSideMode_t side = CUBLAS_SIDE_RIGHT;
    constexpr cublasFillMode_t fill_mode = CUBLAS_FILL_MODE_UPPER;
    constexpr cublasDiagType_t diag_type = CUBLAS_DIAG_NON_UNIT;
    constexpr cublasOperation_t op_zeta = CUBLAS_OP_T;

    if constexpr (std::is_same_v<T, float>) {
        CUBLAS_CHECK(cublasStrmm_v2(handles.cublas, side, fill_mode, op_zeta,
                                    diag_type, n, s, d.one, d.zeta, s, d.s, n,
                                    d.s, n));
    } else {
        CUBLAS_CHECK(cublasDtrmm_v2(handles.cublas, side, fill_mode, op_zeta,
                                    diag_type, n, s, d.one, d.zeta, s, d.s, n,
                                    d.s, n));
    }

    constexpr cublasOperation_t sgeam_op = CUBLAS_OP_N;

    if constexpr (std::is_same_v<T, float>) {
        CUBLAS_CHECK(cublasSgeam(handles.cublas, sgeam_op, sgeam_op, n, s,
                                 d.one, d.s, n, d.one, d.w, n, d.s, n));
    } else {
        CUBLAS_CHECK(cublasDgeam(handles.cublas, sgeam_op, sgeam_op, n, s,
                                 d.one, d.s, n, d.one, d.w, n, d.s, n));
    }
}

// w = w - L^-1 * A * s * xi
template <cils::SupportedType T>
void update_w(Handles &handles, cusparseSpMatDescr_t A,
              cusparseDnMatDescr_t s_desc, cusparseDnMatDescr_t temp,
              cusparseSpMatDescr_t L, DeviceBuffer<T> &d,
              const SpsmCache<T> &spsm_nt, std::int64_t n, std::int64_t s,
              void *d_scratch, cudaStream_t stream) {
    nvtx3::scoped_range w_zeta_range{"w = w - L^-1 * A * s * xi"};
    CudaTimerRange er{g_event_timer, "w = w - L^-1 * A * s * xi", stream};

    // temp = A * s
    constexpr cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    constexpr T alpha = 1.0;
    constexpr T beta = 0.0;
    constexpr cudaDataType compute_type = cuda_type<T>;
    constexpr cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT;

    CUSPARSE_CHECK(cusparseSpMM(handles.cusparse, op, op, &alpha, A, s_desc,
                                &beta, temp, compute_type, alg, d_scratch));

    // temp = L^-1 * temp
    sptri_solve<T>(handles.cusparse, temp, op, L, temp, spsm_nt);

    // w = w - temp * xi
    constexpr cublasOperation_t sgemm_op = CUBLAS_OP_N;

    if constexpr (std::is_same_v<T, float>) {
        CUBLAS_CHECK(cublasSgemm_v2(handles.cublas, sgemm_op, sgemm_op, n, s,
                                    s, d.neg_one, d.temp, n, d.xi, s, d.one,
                                    d.w, n));
    } else {
        CUBLAS_CHECK(cublasDgemm_v2(handles.cublas, sgemm_op, sgemm_op, n, s,
                                    s, d.neg_one, d.temp, n, d.xi, s, d.one,
                                    d.w, n));
    }
}

// [w, zeta] = qr(w)
template <cils::SupportedType T, QrPolicy<T> Qr>
void orthonormalize_w(Qr &qr, Handles &handles, DeviceBuffer<T> &d,
                      std::int64_t n, std::int64_t s, cudaStream_t stream) {
    nvtx3::scoped_range w_zeta_range{"[w zeta] = QR(w)"};
    CudaTimerRange er{g_event_timer, "[w zeta] = QR(w)", stream};

    qr.solve(d.w, d.zeta, d.w, n, s, handles.cublas, handles.cusolver,
             handles.cusolver_params, stream);
    qr.check(static_cast<int>(s), "iteration orthonormalization", stream);
}

// s = (L^-1)' * w + s * zeta'
template <cils::SupportedType T>
void update_s_preconditioned(Handles &handles, cusparseDnMatDescr_t temp,
                             cusparseDnMatDescr_t w_desc, cusparseSpMatDescr_t L,
                             DeviceBuffer<T> &d, const SpsmCache<T> &spsm_t,
                             std::int64_t n, std::int64_t s,
                             cudaStream_t stream) {
    nvtx3::scoped_range s_range{"s = (L^-1)' * w + s * zeta'"};
    CudaTimerRange er{g_event_timer, "s = (L^-1)' * w + s * zeta'", stream};

    constexpr cublasSideMode_t side = CUBLAS_SIDE_RIGHT;
    constexpr cublasFillMode_t fill_mode = CUBLAS_FILL_MODE_UPPER;
    constexpr cublasDiagType_t diag_type = CUBLAS_DIAG_NON_UNIT;
    constexpr cublasOperation_t op_zeta = CUBLAS_OP_T;

    if constexpr (std::is_same_v<T, float>) {
        CUBLAS_CHECK(cublasStrmm_v2(handles.cublas, side, fill_mode, op_zeta,
                                    diag_type, n, s, d.one, d.zeta, s, d.s, n,
                                    d.s, n));
    } else {
        CUBLAS_CHECK(cublasDtrmm_v2(handles.cublas, side, fill_mode, op_zeta,
                                    diag_type, n, s, d.one, d.zeta, s, d.s, n,
                                    d.s, n));
    }

    sptri_solve<T>(handles.cusparse, temp, CUSPARSE_OPERATION_TRANSPOSE, L,
                   w_desc, spsm_t);

    constexpr cublasOperation_t sgeam_op = CUBLAS_OP_N;

    if constexpr (std::is_same_v<T, float>) {
        CUBLAS_CHECK(cublasSgeam(handles.cublas, sgeam_op, sgeam_op, n, s,
                                 d.one, d.s, n, d.one, d.temp, n, d.s, n));
    } else {
        CUBLAS_CHECK(cublasDgeam(handles.cublas, sgeam_op, sgeam_op, n, s,
                                 d.one, d.s, n, d.one, d.temp, n, d.s, n));
    }
}

#ifdef SOLVERS_BUILD_MATHDX

// As = A * s
template <cils::SupportedType T>
class [[nodiscard]] AsCalculator {
  public:
    AsCalculator(cusparseHandle_t cusparse, std::int64_t n, std::int64_t s,
                 cusparseSpMatDescr_t A_desc, cusparseDnMatDescr_t s_desc, cudaStream_t stream) noexcept
        : cusparse{cusparse}, n{n}, s{s}, A_desc{A_desc}, s_desc{s_desc}, stream{stream} {
        CUDA_CHECK(cudaMallocAsync(&d_As, sizeof(T) * n * s, stream));
        CUSPARSE_CHECK(cusparseCreateDnMat(&As_desc, n, s, n, d_As, compute_type,
                                           CUSPARSE_ORDER_COL));

        std::size_t buffer_size = 0;
        CUSPARSE_CHECK(cusparseSpMM_bufferSize(
            cusparse, op, op, &alpha, A_desc, s_desc, &beta,
            As_desc, compute_type, alg, &buffer_size));
        if (buffer_size > 0) {
            CUDA_CHECK(cudaMallocAsync(&d_buffer, buffer_size, stream));
        }
    }

    AsCalculator(const AsCalculator &) = delete;
    AsCalculator &operator=(const AsCalculator &) = delete;
    AsCalculator(AsCalculator &&) = delete;
    AsCalculator &operator=(AsCalculator &&) = delete;

    ~AsCalculator() noexcept {
        release();
    }

    void update() noexcept {
        nvtx3::scoped_range as_range{"AS = A * s"};
        CudaTimerRange er{g_event_timer, "AS = A * s", stream};

        CUSPARSE_CHECK(cusparseSpMM(cusparse, op, op, &alpha, A_desc, s_desc,
                                    &beta, As_desc, compute_type, alg, d_buffer));
    }

    void release() noexcept {
        if (As_desc != nullptr) {
            CUSPARSE_CHECK(cusparseDestroyDnMat(As_desc));
        }
        As_desc = nullptr;

        if (d_As != nullptr) {
            CUDA_CHECK(cudaFreeAsync(d_As, stream));
        }
        d_As = nullptr;

        if (d_buffer != nullptr) {
            CUDA_CHECK(cudaFreeAsync(d_buffer, stream));
        }
        d_buffer = nullptr;
    }

    [[nodiscard]] cusparseDnMatDescr_t As_descriptor() const noexcept {
        return As_desc;
    }

    [[nodiscard]] T *As_memory() const noexcept {
        return d_As;
    }

  private:
    static constexpr cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    static constexpr T alpha = 1.0;
    static constexpr T beta = 0.0;
    static constexpr cudaDataType_t compute_type = cuda_type<T>;
    static constexpr cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT;

    const cusparseHandle_t cusparse;
    const std::int64_t n;
    const std::int64_t s;
    const cusparseSpMatDescr_t A_desc;
    const cusparseDnMatDescr_t s_desc;
    const cudaStream_t stream;

    T *d_As = nullptr;
    cusparseDnMatDescr_t As_desc = nullptr;
    void *d_buffer = nullptr;
};

#endif // SOLVERS_BUILD_MATHDX

} // namespace cils::dr_bcg::cuda
