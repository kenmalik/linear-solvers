#pragma once

#include "common/cuda_checks.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusparse.h>

#include <utility>

namespace cils::cuda {

class Handles {
  public:
    cusparseHandle_t cusparse{};
    cusolverDnHandle_t cusolver{};
    cusolverDnParams_t cusolver_params{};
    cublasHandle_t cublas{};

    [[nodiscard]] Handles() noexcept {
        CUSPARSE_CHECK(cusparseCreate(&cusparse));
        CUSOLVER_CHECK(cusolverDnCreate(&cusolver));
        CUSOLVER_CHECK(cusolverDnCreateParams(&cusolver_params));
        CUBLAS_CHECK(cublasCreate_v2(&cublas));
    }

    ~Handles() noexcept {
        free();
    }

    Handles(const Handles &) = delete;
    Handles &operator=(const Handles &) = delete;

    [[nodiscard]] Handles(Handles &&other) noexcept : cusparse{std::exchange(other.cusparse, nullptr)},
                                                      cusolver{std::exchange(other.cusolver, nullptr)},
                                                      cusolver_params{std::exchange(other.cusolver_params, nullptr)},
                                                      cublas{std::exchange(other.cublas, nullptr)} {}

    Handles &operator=(Handles &&other) noexcept {
        if (this != &other) {
            free();

            cusparse = std::exchange(other.cusparse, nullptr);
            cusolver = std::exchange(other.cusolver, nullptr);
            cusolver_params = std::exchange(other.cusolver_params, nullptr);
            cublas = std::exchange(other.cublas, nullptr);
        }
        return *this;
    }

    void set_stream(cudaStream_t stream) const noexcept {
        CUSPARSE_CHECK(cusparseSetStream(cusparse, stream));
        CUSOLVER_CHECK(cusolverDnSetStream(cusolver, stream));
        CUBLAS_CHECK(cublasSetStream_v2(cublas, stream));
    }

  private:
    void free() const noexcept {
        CUSPARSE_CHECK(cusparseDestroy(cusparse));
        CUSOLVER_CHECK(cusolverDnDestroy(cusolver));
        CUSOLVER_CHECK(cusolverDnDestroyParams(cusolver_params));
        CUBLAS_CHECK(cublasDestroy_v2(cublas));
    }
};

} // namespace cils::cuda
