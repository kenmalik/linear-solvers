#pragma once

#include <cassert>
#include <vector>

#include <nvtx3/nvtx3.hpp>

#include <cusolverDn.h>

#include "common/cuda_checks.h"
#include "dr_bcg/cuda.h"
#include "dr_bcg/helper.h"
#include "dr_bcg/internal/type_info.h"

template <typename T> struct HouseholderQrWorkspace {
    T *d_tau = nullptr;
    void *d_work = nullptr;
    int *d_info = nullptr;
    int *h_info = nullptr;
    void *h_work = nullptr;

    std::size_t lwork_geqrf_d = 0;
    std::size_t lwork_geqrf_h = 0;
    int numfloats_orgqr_d = 0;

    HouseholderQrWorkspace() = default;
    HouseholderQrWorkspace(const HouseholderQrWorkspace &) = delete;
    HouseholderQrWorkspace &operator=(const HouseholderQrWorkspace &) = delete;

    // m: rows of Q (problem size n), n: cols of Q (block size s)
    void allocate(cusolverDnHandle_t &cusolverH, cusolverDnParams_t &params,
                  int m, int n) {
        constexpr cudaDataType_t data_type = Type_info<T>::cuda;

        CUDA_CHECK(cudaMalloc(&d_tau, sizeof(T) * n));
        CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
        CUDA_CHECK(
            cudaMallocHost(reinterpret_cast<void **>(&h_info), sizeof(int)));

        // Dummy m×n device buffer needed to query workspace sizes.
        // Buffer size queries are dimension/type-driven; values are not read.
        T *d_dummy = nullptr;
        CUDA_CHECK(cudaMalloc(&d_dummy, sizeof(T) * m * n));

        CUSOLVER_CHECK(cusolverDnXgeqrf_bufferSize(
            cusolverH, params, m, n, data_type, d_dummy, m, data_type, d_tau,
            data_type, &lwork_geqrf_d, &lwork_geqrf_h));

        if constexpr (std::is_same_v<T, float>) {
            CUSOLVER_CHECK(cusolverDnSorgqr_bufferSize(
                cusolverH, m, n, n, d_dummy, m, d_tau, &numfloats_orgqr_d));
        } else {
            CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(
                cusolverH, m, n, n, d_dummy, m, d_tau, &numfloats_orgqr_d));
        }

        CUDA_CHECK(cudaFree(d_dummy));

        const std::size_t lwork_orgqr_d = numfloats_orgqr_d * sizeof(T);
        CUDA_CHECK(cudaMalloc(&d_work, std::max(lwork_geqrf_d, lwork_orgqr_d)));

        if (lwork_geqrf_h > 0) {
            h_work = malloc(lwork_geqrf_h);
            if (!h_work)
                throw std::runtime_error(
                    "Error: QrWorkspace h_work not allocated.");
        }
    }

    ~HouseholderQrWorkspace() {
        if (d_tau)
            CUDA_CHECK(cudaFree(d_tau));
        if (d_work)
            CUDA_CHECK(cudaFree(d_work));
        if (d_info)
            CUDA_CHECK(cudaFree(d_info));
        if (h_info)
            CUDA_CHECK(cudaFreeHost(h_info));
        if (h_work)
            free(h_work);
    }
};

template <typename T> struct CholQrWorkspace {
    T *d_gram = nullptr;
    int *d_info = nullptr;
    int *h_info = nullptr;
    void *d_work = nullptr;
    std::size_t d_work_size = 0;
    void *h_work = nullptr;
    std::size_t h_work_size = 0;
    T *h_factor = nullptr;

    CholQrWorkspace() = default;
    CholQrWorkspace(const CholQrWorkspace &) = delete;
    CholQrWorkspace &operator=(const CholQrWorkspace &) = delete;

    void allocate(cusolverDnHandle_t &cusolverH, cusolverDnParams_t &params,
                  int n) {
        constexpr cudaDataType_t data_type = Type_info<T>::cuda;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_gram),
                              sizeof(T) * n * n));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));
        CUDA_CHECK(
            cudaMallocHost(reinterpret_cast<void **>(&h_info), sizeof(int)));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&h_factor),
                                  sizeof(T) * n * n));

        T *d_dummy = nullptr;
        CUDA_CHECK(cudaMalloc(&d_dummy, sizeof(T) * n * n));
        CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(
            cusolverH, params, CUBLAS_FILL_MODE_UPPER, n, data_type, d_dummy,
            n, data_type, &d_work_size, &h_work_size));
        CUDA_CHECK(cudaFree(d_dummy));

        if (d_work_size > 0) {
            CUDA_CHECK(
                cudaMalloc(reinterpret_cast<void **>(&d_work), d_work_size));
        }
        if (h_work_size > 0) {
            h_work = malloc(h_work_size);
            if (!h_work)
                throw std::runtime_error(
                    "Error: CholQrWorkspace h_work not allocated.");
        }
    }

    ~CholQrWorkspace() {
        if (d_gram)
            CUDA_CHECK(cudaFree(d_gram));
        if (d_info)
            CUDA_CHECK(cudaFree(d_info));
        if (h_info)
            CUDA_CHECK(cudaFreeHost(h_info));
        if (d_work)
            CUDA_CHECK(cudaFree(d_work));
        if (h_factor)
            CUDA_CHECK(cudaFreeHost(h_factor));
        if (h_work)
            free(h_work);
    }
};

template <typename T> struct LuWorkspace {
    int64_t *d_Ipiv = nullptr;
    void *d_work = nullptr;
    std::size_t d_work_size = 0;
    void *h_work = nullptr;
    std::size_t h_work_size = 0;
    int *d_info = nullptr;
    int *h_info = nullptr;
    T *d_I = nullptr;
    T *h_I = nullptr;

    LuWorkspace() = default;
    LuWorkspace(const LuWorkspace &) = delete;
    LuWorkspace &operator=(const LuWorkspace &) = delete;

    // n: matrix side length (block size s)
    void allocate(cusolverDnHandle_t &cusolverH, cusolverDnParams_t &params,
                  int n) {
        constexpr cudaDataType_t data_type = Type_info<T>::cuda;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Ipiv),
                              sizeof(int64_t) * n));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));
        CUDA_CHECK(
            cudaMallocHost(reinterpret_cast<void **>(&h_info), sizeof(int)));

        // Pre-build the identity matrix on pinned host memory.
        // h_I is constant; d_I is restored from h_I on each call (async).
        CUDA_CHECK(
            cudaMallocHost(reinterpret_cast<void **>(&h_I), sizeof(T) * n * n));
        std::fill(h_I, h_I + n * n, T{0});
        for (int i = 0; i < n; ++i)
            h_I[i * n + i] = T{1};
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void **>(&d_I), sizeof(T) * n * n));

        // Query LU workspace size with a dummy n×n buffer.
        T *d_dummy = nullptr;
        CUDA_CHECK(cudaMalloc(&d_dummy, sizeof(T) * n * n));
        CUSOLVER_CHECK(cusolverDnXgetrf_bufferSize(
            cusolverH, params, n, n, data_type, d_dummy, n, data_type,
            &d_work_size, &h_work_size));
        CUDA_CHECK(cudaFree(d_dummy));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), d_work_size));
        if (h_work_size > 0) {
            h_work = malloc(h_work_size);
            if (!h_work)
                throw std::runtime_error(
                    "Error: LuWorkspace h_work not allocated.");
        }
    }

    ~LuWorkspace() {
        if (d_Ipiv)
            CUDA_CHECK(cudaFree(d_Ipiv));
        if (d_work)
            CUDA_CHECK(cudaFree(d_work));
        if (d_info)
            CUDA_CHECK(cudaFree(d_info));
        if (h_info)
            CUDA_CHECK(cudaFreeHost(h_info));
        if (d_I)
            CUDA_CHECK(cudaFree(d_I));
        if (h_I)
            CUDA_CHECK(cudaFreeHost(h_I));
        if (h_work)
            free(h_work);
    }
};

template <typename T>
void orthonormalize_block(
    cublasHandle_t &cublasH, cusolverDnHandle_t &cusolverH,
    cusolverDnParams_t &params, T *d_Q, T *d_R, const int m, const int n,
    const T *d_A, dr_bcg::cuda::QrBackend backend,
    HouseholderQrWorkspace<T> &householder_ws, CholQrWorkspace<T> &cholqr_ws,
    cudaStream_t stream) {
    NVTX3_FUNC_RANGE();

    constexpr cudaDataType_t data_type = Type_info<T>::cuda;

    assert(n < m && "Expect cols to be less than rows for DR-BCG");

    switch (backend) {
    case dr_bcg::cuda::QrBackend::Householder: {
        CUDA_CHECK(cudaMemcpyAsync(d_Q, d_A, sizeof(T) * m * n,
                                   cudaMemcpyDeviceToDevice, stream));

        CUSOLVER_CHECK(cusolverDnXgeqrf(
            cusolverH, params, m, n, data_type, d_Q, m, data_type,
            householder_ws.d_tau, data_type, householder_ws.d_work,
            householder_ws.lwork_geqrf_d, householder_ws.h_work,
            householder_ws.lwork_geqrf_h, householder_ws.d_info));

        copy_upper_triangular(d_R, d_Q, m, n, stream);

        if constexpr (std::is_same_v<T, float>) {
            CUSOLVER_CHECK(cusolverDnSorgqr(
                cusolverH, m, n, n, d_Q, m, householder_ws.d_tau,
                reinterpret_cast<T *>(householder_ws.d_work),
                householder_ws.numfloats_orgqr_d, householder_ws.d_info));
        } else {
            CUSOLVER_CHECK(cusolverDnDorgqr(
                cusolverH, m, n, n, d_Q, m, householder_ws.d_tau,
                reinterpret_cast<T *>(householder_ws.d_work),
                householder_ws.numfloats_orgqr_d, householder_ws.d_info));
        }

        CUDA_CHECK(cudaMemcpyAsync(householder_ws.h_info,
                                   householder_ws.d_info, sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
        break;
    }
    case dr_bcg::cuda::QrBackend::CholQR: {
        constexpr T alpha = 1;
        constexpr T beta = 0;
        CUDA_CHECK(cudaMemcpyAsync(d_Q, d_A, sizeof(T) * m * n,
                                   cudaMemcpyDeviceToDevice, stream));
        CUBLAS_CHECK(cublasSetPointerMode(cublasH, CUBLAS_POINTER_MODE_HOST));

        if constexpr (std::is_same_v<T, float>) {
            CUBLAS_CHECK(cublasSsyrk(cublasH, CUBLAS_FILL_MODE_UPPER,
                                     CUBLAS_OP_T, n, m, &alpha, d_A, m, &beta,
                                     cholqr_ws.d_gram, n));
        } else {
            CUBLAS_CHECK(cublasDsyrk(cublasH, CUBLAS_FILL_MODE_UPPER,
                                     CUBLAS_OP_T, n, m, &alpha, d_A, m, &beta,
                                     cholqr_ws.d_gram, n));
        }

        CUSOLVER_CHECK(cusolverDnXpotrf(
            cusolverH, params, CUBLAS_FILL_MODE_UPPER, n, data_type,
            cholqr_ws.d_gram, n, data_type, cholqr_ws.d_work,
            cholqr_ws.d_work_size, cholqr_ws.h_work, cholqr_ws.h_work_size,
            cholqr_ws.d_info));

        copy_upper_triangular(d_R, cholqr_ws.d_gram, n, n, stream);

        if constexpr (std::is_same_v<T, float>) {
            CUBLAS_CHECK(cublasStrsm_v2(
                cublasH, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha,
                cholqr_ws.d_gram, n, d_Q, m));
        } else {
            CUBLAS_CHECK(cublasDtrsm_v2(
                cublasH, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha,
                cholqr_ws.d_gram, n, d_Q, m));
        }

        CUDA_CHECK(cudaMemcpyAsync(cholqr_ws.h_info, cholqr_ws.d_info,
                                   sizeof(int), cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaMemcpyAsync(cholqr_ws.h_factor, cholqr_ws.d_gram,
                                   sizeof(T) * n * n, cudaMemcpyDeviceToHost,
                                   stream));
        CUBLAS_CHECK(cublasSetPointerMode(cublasH, CUBLAS_POINTER_MODE_DEVICE));
        break;
    }
    default:
        throw std::runtime_error("Unknown QR backend");
    }
}

template <typename T> struct SpsmCache {
    cusparseSpSMDescr_t spsm = nullptr;
    void *buffer = nullptr;

    SpsmCache() = default;

    SpsmCache(const SpsmCache &) = delete;
    SpsmCache &operator=(const SpsmCache &) = delete;

    void analyze(const cusparseHandle_t &cusparseH, cusparseOperation_t opA,
                 const cusparseSpMatDescr_t &A, const cusparseDnMatDescr_t &B,
                 cusparseDnMatDescr_t &C) {
        constexpr cusparseOperation_t OP_B = CUSPARSE_OPERATION_NON_TRANSPOSE;
        constexpr cudaDataType_t compute_type = Type_info<T>::cuda;
        constexpr T alpha = 1;
        constexpr cusparseSpSMAlg_t ALG_TYPE = CUSPARSE_SPSM_ALG_DEFAULT;

        CUSPARSE_CHECK(cusparseSpSM_createDescr(&spsm));

        size_t buffer_size = 0;
        CUSPARSE_CHECK(cusparseSpSM_bufferSize(
            cusparseH, opA, OP_B, reinterpret_cast<const void *>(&alpha), A, B,
            C, compute_type, ALG_TYPE, spsm, &buffer_size));

        if (buffer_size > 0) {
            CUDA_CHECK(cudaMalloc(&buffer, buffer_size));
        } else {
            throw std::runtime_error("spsm cache: buffer not allocated");
        }

        CUSPARSE_CHECK(cusparseSpSM_analysis(
            cusparseH, opA, OP_B, reinterpret_cast<const void *>(&alpha), A, B,
            C, compute_type, ALG_TYPE, spsm, buffer));
    }

    ~SpsmCache() {
        if (buffer) {
            CUDA_CHECK(cudaFree(buffer));
            buffer = nullptr;
        }
        if (spsm) {
            CUSPARSE_CHECK(cusparseSpSM_destroyDescr(spsm));
            spsm = nullptr;
        }
    }
};

template <typename T>
void sptri_solve(const cusparseHandle_t &cusparseH, cusparseDnMatDescr_t &C,
                 cusparseOperation_t opA, const cusparseSpMatDescr_t &A,
                 const cusparseDnMatDescr_t &B, const SpsmCache<T> &cache) {
    NVTX3_FUNC_RANGE();

    constexpr cusparseOperation_t OP_B = CUSPARSE_OPERATION_NON_TRANSPOSE;
    constexpr cudaDataType_t compute_type = Type_info<T>::cuda;
    constexpr T alpha = 1;
    constexpr cusparseSpSMAlg_t ALG_TYPE = CUSPARSE_SPSM_ALG_DEFAULT;

    CUSPARSE_CHECK(cusparseSpSM_solve(
        cusparseH, opA, OP_B, reinterpret_cast<const void *>(&alpha), A, B, C,
        compute_type, ALG_TYPE, cache.spsm));
}

template <typename T>
void invert_square_matrix(cusolverDnHandle_t &cusolverH,
                          cusolverDnParams_t &params, T *d_A, const int n,
                          LuWorkspace<T> &ws, cudaStream_t stream) {
    NVTX3_FUNC_RANGE();

    constexpr cudaDataType_t data_type = Type_info<T>::cuda;

    // Restore identity into d_I from the pinned h_I template (async).
    CUDA_CHECK(cudaMemcpyAsync(ws.d_I, ws.h_I, sizeof(T) * n * n,
                               cudaMemcpyHostToDevice, stream));

    CUSOLVER_CHECK(cusolverDnXgetrf(
        cusolverH, params, n, n, data_type, d_A, n, ws.d_Ipiv, data_type,
        ws.d_work, ws.d_work_size, ws.h_work, ws.h_work_size, ws.d_info));

    // Solve A * X = I; result (A^{-1}) lands in ws.d_I.
    CUSOLVER_CHECK(cusolverDnXgetrs(cusolverH, params, CUBLAS_OP_N, n, n,
                                    data_type, d_A, n, ws.d_Ipiv, data_type,
                                    ws.d_I, n, ws.d_info));

    CUDA_CHECK(cudaMemcpyAsync(d_A, ws.d_I, sizeof(T) * n * n,
                               cudaMemcpyDeviceToDevice, stream));

    // Async readback of final d_info; caller checks *ws.h_info after
    // cudaStreamSynchronize.
    CUDA_CHECK(cudaMemcpyAsync(ws.h_info, ws.d_info, sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
}
