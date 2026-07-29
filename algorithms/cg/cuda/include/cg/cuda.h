#pragma once

#include "common/cuda_checks.h"
#include "common/cuda_type.cuh"

#include <cstdint>

#include <cublas_v2.h>
#include <cusparse_v2.h>

namespace {

template <cils::SupportedType T>
struct DeviceBuffers {
    DeviceBuffers(std::int64_t n) noexcept {
        CUDA_CHECK(cudaMalloc(&d_r, sizeof(T) * n));
        CUDA_CHECK(cudaMalloc(&d_s, sizeof(T) * n));
        CUDA_CHECK(cudaMalloc(&d_d, sizeof(T) * n));
        CUDA_CHECK(cudaMalloc(&d_q, sizeof(T) * n));

        CUSPARSE_CHECK(cusparseCreateDnVec(&r, n, d_r, cils::detail::cuda_type<T>));
        CUSPARSE_CHECK(cusparseCreateDnVec(&s, n, d_s, cils::detail::cuda_type<T>));
        CUSPARSE_CHECK(cusparseCreateDnVec(&d, n, d_d, cils::detail::cuda_type<T>));
        CUSPARSE_CHECK(cusparseCreateDnVec(&q, n, d_q, cils::detail::cuda_type<T>));
    }

    DeviceBuffers(const DeviceBuffers &) = delete;
    DeviceBuffers(DeviceBuffers &&) = delete;
    DeviceBuffers &operator=(const DeviceBuffers &) = delete;
    DeviceBuffers &operator=(DeviceBuffers &&) = delete;

    ~DeviceBuffers() noexcept {
        CUDA_CHECK(cudaFree(d_r));
        CUDA_CHECK(cudaFree(d_s));
        CUDA_CHECK(cudaFree(d_d));
        CUDA_CHECK(cudaFree(d_q));

        CUSPARSE_CHECK(cusparseDestroyDnVec(r));
        CUSPARSE_CHECK(cusparseDestroyDnVec(s));
        CUSPARSE_CHECK(cusparseDestroyDnVec(d));
        CUSPARSE_CHECK(cusparseDestroyDnVec(q));
    }

    cusparseDnVecDescr_t r{};
    cusparseDnVecDescr_t s{};
    cusparseDnVecDescr_t d{};
    cusparseDnVecDescr_t q{};

    T *d_r;
    T *d_s;
    T *d_d;
    T *d_q;
};

} // namespace

namespace cg::cuda {

struct Config {
    double tolerance = 1e-6;  // NOLINT
    int max_iterations = 100; // NOLINT
    bool real_residual = false;
    cudaStream_t stream = nullptr;
};

int solve(cusparseHandle_t cusparse, cublasHandle_t cublas,
          cusparseSpMatDescr_t A, cusparseDnVecDescr_t b,
          cusparseDnVecDescr_t x, cusparseSpMatDescr_t L,
          Config config);

} // namespace cg::cuda
