#pragma once

#include "common/cuda_checks.h"
#include "common/cuda_type.cuh"

#include <cassert>

#include <cstddef>
#include <cusolverDn.h>
#include <cusparse_v2.h>
#include <nvtx3/nvtx3.hpp>
#include <vector>

template <cils::SupportedType T>
struct LuWorkspace {
    int64_t *d_Ipiv = nullptr;
    void *d_work = nullptr;
    std::size_t d_work_size = 0;
    std::vector<std::byte> h_work;
    std::size_t h_work_size = 0;
    int *d_info = nullptr;
    int *h_info = nullptr;
    T *d_I = nullptr;
    T *h_I = nullptr;

    LuWorkspace() = default;
    LuWorkspace(const LuWorkspace &) = delete;
    LuWorkspace &operator=(const LuWorkspace &) = delete;
    LuWorkspace(LuWorkspace &&) = delete;
    LuWorkspace &operator=(LuWorkspace &&) = delete;

    // n: matrix side length (block size s)
    void allocate(cusolverDnHandle_t &cusolverH, cusolverDnParams_t &params,
                  int n) {
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
        for (int i = 0; i < n; ++i) {
            h_I[(i * n) + i] = T{1};
        }
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void **>(&d_I), sizeof(T) * n * n));

        // Query LU workspace size with a dummy n×n buffer.
        T *d_dummy = nullptr;
        CUDA_CHECK(cudaMalloc(&d_dummy, sizeof(T) * n * n));
        CUSOLVER_CHECK(cusolverDnXgetrf_bufferSize(
            cusolverH, params, n, n, cils::detail::cuda_type<T>, d_dummy, n, cils::detail::cuda_type<T>,
            &d_work_size, &h_work_size));
        CUDA_CHECK(cudaFree(d_dummy));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), d_work_size));
        if (h_work_size > 0) {
            h_work.resize(h_work_size);
        }
    }

    ~LuWorkspace() {
        if (d_Ipiv != nullptr) {
            CUDA_CHECK(cudaFree(d_Ipiv));
        }
        if (d_work != nullptr) {
            CUDA_CHECK(cudaFree(d_work));
        }
        if (d_info != nullptr) {
            CUDA_CHECK(cudaFree(d_info));
        }
        if (h_info != nullptr) {
            CUDA_CHECK(cudaFreeHost(h_info));
        }
        if (d_I) {
            CUDA_CHECK(cudaFree(d_I));
        }
        if (h_I) {
            CUDA_CHECK(cudaFreeHost(h_I));
        }
    }
};

template <cils::SupportedType T>
struct SpsmCache {
    cusparseSpSMDescr_t spsm = nullptr;
    void *buffer = nullptr;

    SpsmCache() = default;

    SpsmCache(const SpsmCache &) = delete;
    SpsmCache &operator=(const SpsmCache &) = delete;
    SpsmCache(SpsmCache &&) = delete;
    SpsmCache &operator=(SpsmCache &&) = delete;

    void analyze(const cusparseHandle_t &cusparseH, cusparseOperation_t opA,
                 const cusparseSpMatDescr_t &A, const cusparseDnMatDescr_t &B,
                 cusparseDnMatDescr_t &C) {
        constexpr cusparseOperation_t OP_B = CUSPARSE_OPERATION_NON_TRANSPOSE;
        constexpr T alpha = 1;
        constexpr cusparseSpSMAlg_t ALG_TYPE = CUSPARSE_SPSM_ALG_DEFAULT;

        CUSPARSE_CHECK(cusparseSpSM_createDescr(&spsm));

        size_t buffer_size = 0;
        CUSPARSE_CHECK(cusparseSpSM_bufferSize(
            cusparseH, opA, OP_B, reinterpret_cast<const void *>(&alpha), A, B,
            C, cils::detail::cuda_type<T>, ALG_TYPE, spsm, &buffer_size));

        if (buffer_size > 0) {
            CUDA_CHECK(cudaMalloc(&buffer, buffer_size));
        } else {
            throw std::runtime_error("spsm cache: buffer not allocated");
        }

        CUSPARSE_CHECK(cusparseSpSM_analysis(
            cusparseH, opA, OP_B, reinterpret_cast<const void *>(&alpha), A, B,
            C, cils::detail::cuda_type<T>, ALG_TYPE, spsm, buffer));
    }

    ~SpsmCache() {
        if (buffer != nullptr) {
            CUDA_CHECK(cudaFree(buffer));
            buffer = nullptr;
        }
        if (spsm != nullptr) {
            CUSPARSE_CHECK(cusparseSpSM_destroyDescr(spsm));
            spsm = nullptr;
        }
    }
};

template <cils::SupportedType T>
void sptri_solve(const cusparseHandle_t &cusparseH, cusparseDnMatDescr_t &C,
                 cusparseOperation_t opA, const cusparseSpMatDescr_t &A,
                 const cusparseDnMatDescr_t &B, const SpsmCache<T> &cache) {
    NVTX3_FUNC_RANGE();

    constexpr cusparseOperation_t OP_B = CUSPARSE_OPERATION_NON_TRANSPOSE;
    constexpr T alpha = 1;
    constexpr cusparseSpSMAlg_t ALG_TYPE = CUSPARSE_SPSM_ALG_DEFAULT;

    CUSPARSE_CHECK(cusparseSpSM_solve(
        cusparseH, opA, OP_B, reinterpret_cast<const void *>(&alpha), A, B, C,
        cils::detail::cuda_type<T>, ALG_TYPE, cache.spsm));
}

template <cils::SupportedType T>
void invert_square_matrix(cusolverDnHandle_t &cusolverH,
                          cusolverDnParams_t &params, T *d_A, const int n,
                          LuWorkspace<T> &ws, cudaStream_t stream) {
    NVTX3_FUNC_RANGE();

    // Restore identity into d_I from the pinned h_I template (async).
    CUDA_CHECK(cudaMemcpyAsync(ws.d_I, ws.h_I, sizeof(T) * n * n,
                               cudaMemcpyHostToDevice, stream));

    CUSOLVER_CHECK(cusolverDnXgetrf(
        cusolverH, params, n, n, cils::detail::cuda_type<T>, d_A, n, ws.d_Ipiv, cils::detail::cuda_type<T>,
        ws.d_work, ws.d_work_size, ws.h_work.data(), ws.h_work_size, ws.d_info));

    // Solve A * X = I; result (A^{-1}) lands in ws.d_I.
    CUSOLVER_CHECK(cusolverDnXgetrs(cusolverH, params, CUBLAS_OP_N, n, n,
                                    cils::detail::cuda_type<T>, d_A, n, ws.d_Ipiv, cils::detail::cuda_type<T>,
                                    ws.d_I, n, ws.d_info));

    CUDA_CHECK(cudaMemcpyAsync(d_A, ws.d_I, sizeof(T) * n * n,
                               cudaMemcpyDeviceToDevice, stream));

    // Async readback of final d_info; caller checks *ws.h_info after
    // cudaStreamSynchronize.
    CUDA_CHECK(cudaMemcpyAsync(ws.h_info, ws.d_info, sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
}
