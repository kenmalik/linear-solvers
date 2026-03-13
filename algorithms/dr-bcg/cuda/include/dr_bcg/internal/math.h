#pragma once

#include <cassert>
#include <vector>

#include <nvtx3/nvtx3.hpp>

#include <cusolverDn.h>

#include "common/checks.h"
#include "dr_bcg/internal/type_info.h"
#include "dr_bcg/helper.h"

template <typename T>
void qr_factorization(cusolverDnHandle_t &cusolverH, cusolverDnParams_t &params,
                      T *d_Q, T *d_R, const int m, const int n, const T *d_A) {
    NVTX3_FUNC_RANGE();

    constexpr cudaDataType_t data_type = Type_info<T>::cuda;

    assert(n < m && "Expect cols to be less than rows for DR-BCG");

    int info = 0;

    T *d_tau = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tau, sizeof(T) * n));

    int *d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    void *d_work = nullptr;
    std::size_t lwork_geqrf_d = 0;

    void *h_work = nullptr;
    std::size_t lwork_geqrf_h = 0;

    CUDA_CHECK(
        cudaMemcpy(d_Q, d_A, sizeof(T) * m * n, cudaMemcpyDeviceToDevice));

    // Create device buffer
    CUSOLVER_CHECK(cusolverDnXgeqrf_bufferSize(
        cusolverH, params, m, n, data_type, d_Q, m, data_type, d_tau, data_type,
        &lwork_geqrf_d, &lwork_geqrf_h));

    int numfloats_orgqr_d = 0;
    if constexpr (std::is_same_v<T, float>) {
        CUSOLVER_CHECK(cusolverDnSorgqr_bufferSize(cusolverH, m, n, n, d_Q, m,
                                                   d_tau, &numfloats_orgqr_d));
    } else {
        CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(cusolverH, m, n, n, d_Q, m,
                                                   d_tau, &numfloats_orgqr_d));
    }
    const std::size_t lwork_orgqr_d = numfloats_orgqr_d * sizeof(T);

    // Note: The legacy cuSOLVER API returns lwork number of array values
    // while the generic API returns lwork in bytes.
    // This is why we multiply lwork_orgqr by sizeof(T) to get a
    // proper comparison in workspace sizes.
    const std::size_t lwork_bytes_d = std::max(lwork_geqrf_d, lwork_orgqr_d);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), lwork_bytes_d));

    if (lwork_geqrf_h > 0) {
        h_work = reinterpret_cast<void *>(malloc(lwork_geqrf_h));
        if (h_work == nullptr) {
            throw std::runtime_error("Error: h_work not allocated.");
        }
    }

    CUSOLVER_CHECK(cusolverDnXgeqrf(
        cusolverH, params, m, n, data_type, d_Q, m, data_type, d_tau, data_type,
        d_work, lwork_geqrf_d, h_work, lwork_geqrf_h, d_info));

    if (h_work) {
        free(h_work); // No longer needed
    }

    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (0 > info) {
        throw std::runtime_error(std::to_string(-info) +
                                 "-th parameter is wrong \n");
    }

    copy_upper_triangular(d_R, d_Q, m, n);

    // Explicitly compute Q
    if constexpr (std::is_same_v<T, float>) {
        CUSOLVER_CHECK(cusolverDnSorgqr(cusolverH, m, n, n, d_Q, m, d_tau,
                                        reinterpret_cast<T *>(d_work),
                                        numfloats_orgqr_d, d_info));
    } else {
        CUSOLVER_CHECK(cusolverDnDorgqr(cusolverH, m, n, n, d_Q, m, d_tau,
                                        reinterpret_cast<T *>(d_work),
                                        numfloats_orgqr_d, d_info));
    }

    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (0 > info) {
        throw std::runtime_error(std::to_string(-info) +
                                 "-th parameter is wrong \n");
    }

    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_tau));
    CUDA_CHECK(cudaFree(d_work));
}

template <typename T>
void sptri_left_multiply(const cusparseHandle_t &cusparseH,
                         cusparseDnMatDescr_t &C, cusparseOperation_t opA,
                         const cusparseSpMatDescr_t &A,
                         const cusparseDnMatDescr_t &B) {
    NVTX3_FUNC_RANGE();

    constexpr cusparseOperation_t OP_B = CUSPARSE_OPERATION_NON_TRANSPOSE;
    constexpr cudaDataType_t compute_type = Type_info<T>::cuda;

    constexpr T alpha = 1;
    constexpr cusparseSpSMAlg_t ALG_TYPE = CUSPARSE_SPSM_ALG_DEFAULT;

    cusparseSpSMDescr_t spsm{};
    CUSPARSE_CHECK(cusparseSpSM_createDescr(&spsm));

    void *buffer = nullptr;
    size_t buffer_size = 0;

    CUSPARSE_CHECK(cusparseSpSM_bufferSize(
        cusparseH, opA, OP_B, reinterpret_cast<const void *>(&alpha), A, B, C,
        compute_type, ALG_TYPE, spsm, &buffer_size));

    if (buffer_size > 0) {
        CUDA_CHECK(cudaMalloc(&buffer, buffer_size));
    } else {
        throw std::runtime_error("s solve: buffer not allocated");
    }

    CUSPARSE_CHECK(cusparseSpSM_analysis(
        cusparseH, opA, OP_B, reinterpret_cast<const void *>(&alpha), A, B, C,
        compute_type, ALG_TYPE, spsm, buffer));

    CUSPARSE_CHECK(cusparseSpSM_solve(cusparseH, opA, OP_B,
                                      reinterpret_cast<const void *>(&alpha), A,
                                      B, C, compute_type, ALG_TYPE, spsm));

    CUDA_CHECK(cudaFree(buffer));
    CUSPARSE_CHECK(cusparseSpSM_destroyDescr(spsm));
}

template <typename T>
void invert_square_matrix(cusolverDnHandle_t &cusolverH,
                          cusolverDnParams_t &params, T *d_A, const int n) {
    NVTX3_FUNC_RANGE();

    constexpr cudaDataType_t data_type = Type_info<T>::cuda;

    // LU Decomposition
    size_t d_work_size = 0;
    void *d_work = nullptr;
    size_t h_work_size = 0;
    void *h_work = nullptr;

    int info = 0;
    int *d_info = nullptr;

    int64_t *d_Ipiv = nullptr;

    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_Ipiv), sizeof(int64_t) * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    CUSOLVER_CHECK(cusolverDnXgetrf_bufferSize(cusolverH, params, n, n,
                                               data_type, d_A, n, data_type,
                                               &d_work_size, &h_work_size));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), d_work_size));
    if (h_work_size > 0) {
        h_work = reinterpret_cast<void *>(malloc(h_work_size));
        if (h_work == nullptr) {
            throw std::runtime_error("Error: h_work not allocated.");
        }
    }

    CUSOLVER_CHECK(cusolverDnXgetrf(cusolverH, params, n, n, data_type, d_A, n,
                                    d_Ipiv, data_type, d_work, d_work_size,
                                    h_work, h_work_size, d_info));

    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (0 > info) {
        throw std::runtime_error(std::to_string(-info) +
                                 "-th parameter is wrong \n");
    }

    CUDA_CHECK(cudaFree(d_work));
    free(h_work);

    // Solve A * X = I for inverse
    std::vector<T> h_I(n * n, 0);
    T *d_I = nullptr;

    for (int i = 0; i < n; i++) {
        h_I.at(i * n + i) = 1;
    }
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_I), sizeof(T) * h_I.size()));
    CUDA_CHECK(cudaMemcpy(d_I, h_I.data(), sizeof(T) * h_I.size(),
                          cudaMemcpyHostToDevice));

    CUSOLVER_CHECK(cusolverDnXgetrs(cusolverH, params, CUBLAS_OP_N, n, n,
                                    data_type, d_A, n, d_Ipiv, data_type, d_I,
                                    n, d_info));

    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (0 > info) {
        throw std::runtime_error(std::to_string(-info) +
                                 "-th parameter is wrong \n");
    }

    CUDA_CHECK(
        cudaMemcpy(d_A, d_I, sizeof(T) * h_I.size(), cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaFree(d_I));
    CUDA_CHECK(cudaFree(d_Ipiv));
    CUDA_CHECK(cudaFree(d_info));
}
