#pragma once

#include "cuda/detail/math.cuh"

#include "common/cuda_event_timer.h"
#include "common/cuda_type.cuh"

#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace cils::cuda::detail {

template <cils::detail::SupportedType T>
class [[nodiscard]] RCalculator {
  public:
    RCalculator(cusparseHandle_t cusparse, std::int64_t n, std::int64_t s, cudaStream_t stream) noexcept
        : cusparse{cusparse}, n{n}, s{s}, stream{stream} {
        CUDA_CHECK(cudaMallocAsync(&d_R, sizeof(T) * n * s, stream));
        CUSPARSE_CHECK(
            cusparseCreateDnMat(&R, n, s, n, d_R, cils::detail::cuda_type<T>, CUSPARSE_ORDER_COL));
    }

    RCalculator(const RCalculator &) = delete;
    RCalculator &operator=(const RCalculator &) = delete;
    RCalculator(RCalculator &&) = delete;
    RCalculator &operator=(RCalculator &&) = delete;

    ~RCalculator() noexcept {
        release();
    }

    void calculate(cusparseDnMatDescr_t B, cusparseSpMatDescr_t A, cusparseDnMatDescr_t X) noexcept {
        nvtx3::scoped_range R_range{"R = B - A * X"};
        cils::detail::CudaTimerRange er{cils::detail::g_event_timer, "R = B - A * X", stream};

        constexpr T alpha = -1.0;
        constexpr T beta = 1.0;
        constexpr cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
        constexpr cudaDataType_t compute_type = cils::detail::cuda_type<T>;
        constexpr cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT;

        void *d_B_ptr = nullptr;
        CUSPARSE_CHECK(cusparseDnMatGetValues(B, &d_B_ptr));
        CUDA_CHECK(cudaMemcpyAsync(d_R, d_B_ptr, sizeof(T) * n * s,
                                   cudaMemcpyDeviceToDevice, stream));

        std::size_t buffer_size = 0;
        CUSPARSE_CHECK(cusparseSpMM_bufferSize(cusparse, op, op, &alpha,
                                               A, X, &beta, B, compute_type,
                                               alg, &buffer_size));

        void *buffer = nullptr;
        CUDA_CHECK(cudaMallocAsync(&buffer, buffer_size, stream));

        CUSPARSE_CHECK(cusparseSpMM(cusparse, op, op, &alpha, A, X,
                                    &beta, R, compute_type, alg, buffer));

        CUDA_CHECK(cudaFreeAsync(buffer, stream));
    }

    void release() noexcept {
        if (d_R) {
            CUDA_CHECK(cudaFreeAsync(d_R, stream));
        }
        d_R = nullptr;

        if (R != nullptr) {
            CUSPARSE_CHECK(cusparseDestroyDnMat(R));
        }
        R = nullptr;
    }

    [[nodiscard]] cusparseDnMatDescr_t R_descriptor() const noexcept {
        return R;
    }

    [[nodiscard]] T *R_memory() const noexcept {
        return d_R;
    }

  private:
    const cusparseHandle_t cusparse;
    const std::int64_t n;
    const std::int64_t s;
    const cudaStream_t stream;

    T *d_R = nullptr;
    cusparseDnMatDescr_t R = nullptr;
};

template <cils::detail::SupportedType T>
void initialize_preconditioned_s(
    const cusparseHandle_t cusparse, std::int64_t n, std::int64_t s,
    cusparseDnMatDescr_t s_desc, cusparseDnMatDescr_t w_desc, cusparseSpMatDescr_t L_desc,
    const SpsmCache<T> &spsm_transpose, const cudaStream_t stream) {
    nvtx3::scoped_range s_initial_range{"s = (L^-1)' * w"};
    cils::detail::CudaTimerRange er{cils::detail::g_event_timer, "s = (L^-1)' * w", stream};

    T *d_s = nullptr;
    T *d_w = nullptr;

    CUSPARSE_CHECK(cusparseDnMatGetValues(s_desc, reinterpret_cast<void **>(&d_s)));
    CUSPARSE_CHECK(cusparseDnMatGetValues(w_desc, reinterpret_cast<void **>(&d_w)));

    CUDA_CHECK(cudaMemcpyAsync(d_s, d_w, sizeof(T) * n * s,
                               cudaMemcpyDeviceToDevice, stream));

    sptri_solve<T>(cusparse, s_desc, CUSPARSE_OPERATION_TRANSPOSE,
                   L_desc, w_desc, spsm_transpose);
}

inline std::pair<std::int64_t, std::int64_t> get_size(cusparseDnMatDescr_t mat) {
    std::int64_t n = 0;
    std::int64_t s = 0;
    std::int64_t ld = 0;
    void *vals = nullptr;
    cudaDataType_t data_type = CUDA_R_32F;
    cusparseOrder_t order = CUSPARSE_ORDER_COL;

    CUSPARSE_CHECK(cusparseDnMatGet(mat, &n, &s, &ld, &vals, &data_type, &order));

    return {n, s};
}

} // namespace cils::cuda::detail
