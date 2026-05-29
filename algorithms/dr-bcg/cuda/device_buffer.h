#pragma once

#include "common/cuda_checks.h"
#include "common/type_info.h"

/// Templated device pointers for reused device buffers.
///
/// This template manages device memory for all buffers used in the DR-BCG
/// algorithm. It only accepts `float` or `double` as the template parameter.
template <SupportedType T>
struct DeviceBuffer {
    T *w = nullptr;        ///< Device pointer for matrix w (n x s)
    T *sigma = nullptr;    ///< Device pointer for matrix sigma (s x s)
    T *s = nullptr;        ///< Device pointer for matrix s (n x s)
    T *xi = nullptr;       ///< Device pointer for matrix xi (s x s)
    T *zeta = nullptr;     ///< Device pointer for matrix zeta (s x s)
    T *temp = nullptr;     ///< Device pointer for temporary matrix (n x s)
    T *residual = nullptr; ///< Device pointer for residual vector (n)
    T *one = nullptr;      ///< Device scalar: 1.0
    T *zero = nullptr;     ///< Device scalar: 0.0
    T *neg_one = nullptr;  ///< Device scalar: -1.0

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

        const T h_one = 1;
        const T h_zero = 0;
        const T h_neg_one = -1;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&one), sizeof(T)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&zero), sizeof(T)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&neg_one), sizeof(T)));
        CUDA_CHECK(cudaMemcpy(one, &h_one, sizeof(T), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(zero, &h_zero, sizeof(T), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(neg_one, &h_neg_one, sizeof(T), cudaMemcpyHostToDevice));
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

        if (one)
            CUDA_CHECK(cudaFree(one));
        if (zero)
            CUDA_CHECK(cudaFree(zero));
        if (neg_one)
            CUDA_CHECK(cudaFree(neg_one));
        one = zero = neg_one = nullptr;
    }
};

// Common aliases
using DeviceBufferFloat = DeviceBuffer<float>;
using DeviceBufferDouble = DeviceBuffer<double>;
