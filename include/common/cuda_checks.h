#pragma once

#include <cstdlib>
#include <iostream>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cusolver_common.h>

// Helper macros for checking CUDA / cuBLAS / cuSPARSE calls.
// On error these print a concise message (file:line) and abort().

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << ": " << cudaGetErrorString(_err) << " (" << _err      \
                      << ")\n";                                                \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t _status = (call);                                       \
        if (_status != CUBLAS_STATUS_SUCCESS) {                                \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__     \
                      << ": status=" << static_cast<int>(_status) << '\n';     \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

#define CUSPARSE_CHECK(call)                                                   \
    do {                                                                       \
        cusparseStatus_t _status = (call);                                     \
        if (_status != CUSPARSE_STATUS_SUCCESS) {                              \
            std::cerr << "cuSPARSE error at " << __FILE__ << ":" << __LINE__   \
                      << ": status=" << static_cast<int>(_status) << '\n';     \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

#define CUSOLVER_CHECK(call)                                                   \
    do {                                                                       \
        cusolverStatus_t _status = (call);                                     \
        if (_status != CUSOLVER_STATUS_SUCCESS) {                              \
            std::cerr << "cuSOLVER error at " << __FILE__ << ":" << __LINE__   \
                      << ": status=" << static_cast<int>(_status) << '\n';     \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

// Convenience: a small macro to check the last CUDA runtime error
#define CUDA_CHECK_LAST()                                                      \
    do {                                                                       \
        cudaError_t _err = cudaGetLastError();                                 \
        if (_err != cudaSuccess) {                                             \
            std::cerr << "CUDA runtime error at " << __FILE__ << ":"           \
                      << __LINE__ << ": " << cudaGetErrorString(_err) << " ("  \
                      << _err << ")\n";                                        \
            std::abort();                                                      \
        }                                                                      \
    } while (0)
