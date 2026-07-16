#pragma once

#include "common/cuda_checks.h"
#include "common/type_info.h"

#include <cuda_runtime.h>

#include <utility>

namespace dr_bcg::cuda {

template <SupportedType T>
class DeviceBuffer {
  public:
    T *w = nullptr;
    T *sigma = nullptr;
    T *s = nullptr;
    T *xi = nullptr;
    T *zeta = nullptr;
    T *temp = nullptr;
    T *one = nullptr;
    T *zero = nullptr;
    T *neg_one = nullptr;

    [[nodiscard]] DeviceBuffer(int n, int s) noexcept {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&w), sizeof(T) * n * s));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&sigma), sizeof(T) * s * s));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&(this->s)), sizeof(T) * n * s));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&xi), sizeof(T) * s * s));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&zeta), sizeof(T) * s * s));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&temp), sizeof(T) * n * s));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&one), sizeof(T)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&zero), sizeof(T)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&neg_one), sizeof(T)));
        CUDA_CHECK(cudaMemcpy(one, &h_one, sizeof(T), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(zero, &h_zero, sizeof(T), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(neg_one, &h_neg_one, sizeof(T), cudaMemcpyHostToDevice));
    }

    ~DeviceBuffer() noexcept { deallocate(); }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    [[nodiscard]] DeviceBuffer(DeviceBuffer &&other) noexcept : w{std::exchange(other.w, nullptr)},
                                                                sigma{std::exchange(other.sigma, nullptr)},
                                                                s{std::exchange(other.s, nullptr)},
                                                                xi{std::exchange(other.xi, nullptr)},
                                                                zeta{std::exchange(other.zeta, nullptr)},
                                                                temp{std::exchange(other.temp, nullptr)},
                                                                one{std::exchange(other.one, nullptr)},
                                                                zero{std::exchange(other.zero, nullptr)},
                                                                neg_one{std::exchange(other.neg_one, nullptr)} {}

    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
        if (this != &other) {
            deallocate();

            w = std::exchange(other.w, nullptr);
            sigma = std::exchange(other.sigma, nullptr);
            s = std::exchange(other.s, nullptr);
            xi = std::exchange(other.xi, nullptr);
            zeta = std::exchange(other.zeta, nullptr);
            temp = std::exchange(other.temp, nullptr);
            one = std::exchange(other.one, nullptr);
            zero = std::exchange(other.zero, nullptr);
            neg_one = std::exchange(other.neg_one, nullptr);
        }
        return *this;
    }

  private:
    static constexpr T h_one{1};
    static constexpr T h_zero{0};
    static constexpr T h_neg_one{-1};

    void deallocate() noexcept {
        if (w) {
            CUDA_CHECK(cudaFree(w));
        }
        if (sigma) {
            CUDA_CHECK(cudaFree(sigma));
        }
        if (s) {
            CUDA_CHECK(cudaFree(s));
        }
        if (xi) {
            CUDA_CHECK(cudaFree(xi));
        }
        if (zeta) {
            CUDA_CHECK(cudaFree(zeta));
        }
        if (temp) {
            CUDA_CHECK(cudaFree(temp));
        }
        w = sigma = s = xi = zeta = temp = nullptr;

        if (one) {
            CUDA_CHECK(cudaFree(one));
        }
        if (zero) {
            CUDA_CHECK(cudaFree(zero));
        }
        if (neg_one) {
            CUDA_CHECK(cudaFree(neg_one));
        }
        one = zero = neg_one = nullptr;
    }
};

} // namespace dr_bcg::cuda
