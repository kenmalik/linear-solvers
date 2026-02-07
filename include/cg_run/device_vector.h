#pragma once

#include <vector>

#include "cg_run/checks.h"

namespace cg_run {
class DeviceVector {
  public:
    DeviceVector() noexcept {}

    DeviceVector(const std::vector<double> &h_vals) noexcept {
        CUDA_CHECK(cudaMalloc(&d_vals, sizeof(double) * h_vals.size()));
        CUDA_CHECK(cudaMemcpy(d_vals, h_vals.data(),
                              sizeof(double) * h_vals.size(),
                              cudaMemcpyHostToDevice));
        CUSPARSE_CHECK(
            cusparseCreateDnVec(&x, h_vals.size(), d_vals, CUDA_R_64F));
    }

    DeviceVector(const DeviceVector &other) = delete;
    DeviceVector &operator=(const DeviceVector &other) = delete;

    DeviceVector(DeviceVector &&other) noexcept
        : d_vals(other.d_vals), x(other.x) {
        other.d_vals = nullptr;
        other.x = nullptr;
    }

    DeviceVector &operator=(DeviceVector &&other) noexcept {
        if (this != &other) {
            reset();

            d_vals = other.d_vals;
            x = other.x;

            other.d_vals = nullptr;
            other.x = nullptr;
        }
        return *this;
    }

    void reset() noexcept {
        if (d_vals) {
            CUDA_CHECK(cudaFree(d_vals));
            d_vals = nullptr;
        }
        if (x) {
            CUSPARSE_CHECK(cusparseDestroyDnVec(x));
            x = nullptr;
        }
    }

    ~DeviceVector() noexcept { reset(); }

    cusparseDnVecDescr_t &get() noexcept { return x; }

  private:
    double *d_vals = nullptr;
    cusparseDnVecDescr_t x = nullptr;
};
} // namespace cg_run