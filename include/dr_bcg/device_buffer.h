#pragma once

#include "dr_bcg/helper.h"
#include <type_traits>

/**
 * @brief Templated device pointers for reused device buffers.
 *
 * This template manages device memory for all buffers used in the DR-BCG
 * algorithm. It accepts any arithmetic type (float, double, integer types,
 * etc.) but note that many CUDA/cuBLAS/cuSPARSE routines expect floating
 * point types; callers are responsible for choosing a sensible `T`.
 */
template <typename T> struct DeviceBuffer {
    static_assert(std::is_arithmetic<T>::value,
                  "DeviceBuffer<T> requires an arithmetic type");

    T *w = nullptr;        ///< Device pointer for matrix w (n x s)
    T *sigma = nullptr;    ///< Device pointer for matrix sigma (s x s)
    T *s = nullptr;        ///< Device pointer for matrix s (n x s)
    T *xi = nullptr;       ///< Device pointer for matrix xi (s x s)
    T *zeta = nullptr;     ///< Device pointer for matrix zeta (s x s)
    T *temp = nullptr;     ///< Device pointer for temporary matrix (n x s)
    T *residual = nullptr; ///< Device pointer for residual vector (n)

    DeviceBuffer(int n, int s) { allocate(n, s); }
    ~DeviceBuffer() { deallocate(); }

    void allocate(int n, int s) {
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void **>(&w), sizeof(T) * n * s));
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void **>(&sigma), sizeof(T) * s * s));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&(this->s)),
                              sizeof(T) * n * s));
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void **>(&xi), sizeof(T) * s * s));
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void **>(&zeta), sizeof(T) * s * s));
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void **>(&temp), sizeof(T) * n * s));
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void **>(&residual), sizeof(T) * n));
    }

    void deallocate() {
        if (w)
            CUDA_CHECK(cudaFree(w));
        if (sigma)
            CUDA_CHECK(cudaFree(sigma));
        if (s)
            CUDA_CHECK(cudaFree(s));
        if (xi)
            CUDA_CHECK(cudaFree(xi));
        if (zeta)
            CUDA_CHECK(cudaFree(zeta));
        if (temp)
            CUDA_CHECK(cudaFree(temp));
        if (residual)
            CUDA_CHECK(cudaFree(residual));

        w = sigma = s = xi = zeta = temp = residual = nullptr;
    }
};

// Common aliases
using DeviceBufferFloat = DeviceBuffer<float>;
using DeviceBufferDouble = DeviceBuffer<double>;