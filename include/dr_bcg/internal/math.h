#pragma once

#include "dr_bcg/helper.h"
#include "dr_bcg/internal/type_info.h"

template <typename T>
void qr_factorization(cusolverDnHandle_t &cusolverH, cusolverDnParams_t &params,
                      T *d_Q, T *d_R, const int m, const int n, const T *d_A) {
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