#pragma once

#include "common/type_info.h"

#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include <cstddef>

namespace dr_bcg::cuda {

template <SupportedType T>
class [[nodiscard]] RCalculator {
  public:
    RCalculator(cusparseHandle_t cusparse, std::int64_t n, std::int64_t s, cudaStream_t stream) noexcept
        : cusparse{cusparse}, n{n}, s{s}, stream{stream} {
        CUDA_CHECK(cudaMallocAsync(&d_R, sizeof(T) * n * s, stream));
        CUSPARSE_CHECK(
            cusparseCreateDnMat(&R, n, s, n, d_R, cuda_type<T>, CUSPARSE_ORDER_COL));
    }

    RCalculator(const RCalculator &) = delete;
    RCalculator &operator=(const RCalculator &) = delete;

    ~RCalculator() noexcept {
        release();
    }

    void calculate(cusparseDnMatDescr_t B, cusparseSpMatDescr_t A, cusparseDnMatDescr_t X) noexcept {
        nvtx3::scoped_range R_range{"R = B - A * X"};
        CudaTimerRange er{g_event_timer, "R = B - A * X", stream};

        constexpr T alpha = -1.0;
        constexpr T beta = 1.0;
        constexpr cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
        constexpr cudaDataType_t compute_type = cuda_type<T>;
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

        if (R) {
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

} // namespace dr_bcg::cuda