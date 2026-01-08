#include <cg_run/cg.h>
#include <cstdint>
#include <iostream>
#include <type_traits>

namespace {
template <typename T> struct Device_buffers {
    cudaDataType_t cuda_type =
        std::is_same_v<T, float> ? CUDA_R_32F : CUDA_R_64F;

    Device_buffers(std::int64_t n) {
        cudaMalloc(&r_d, sizeof(T) * n);
        cusparseCreateDnVec(&r, n, r_d, cuda_type);
    }

    ~Device_buffers() {
        cudaFree(r_d);
        cusparseDestroyDnVec(r);
    }

    cusparseDnVecDescr_t r;
    T *r_d;
};
} // namespace

namespace cg_run {
// A is
// R is lower triangular incomplete Cholesky where A ~= M = R^TR
using data_type = double;
int cg(cusparseHandle_t cusparse, cublasHandle_t cublas, cusparseSpMatDescr_t A,
       cusparseDnVecDescr_t x, cusparseSpMatDescr_t R, data_type tolerance,
       int max_iterations) {
    std::int64_t n;
    double *_ = nullptr;
    cudaDataType_t compute_type;
    cusparseDnVecGet(x, &n, reinterpret_cast<void **>(&_), &compute_type);

    // cusparseSpSVDescr_t SV_R_descr;
    // cusparseSpSVDescr_t SV_RT_descr;
    // cusparseSpSV_createDescr(&SV_R_descr);
    // cusparseSpSV_createDescr(&SV_RT_descr);

    Device_buffers<data_type> d{n};

    {
        constexpr data_type alpha = 1.0;
        constexpr data_type beta = 0.0;

        std::size_t buffer_size = 0;
        cusparseSpMV_bufferSize(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, A, x, &beta, d.r, compute_type,
                                CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size);

        void *buffer = nullptr;
        cudaMalloc(&buffer, buffer_size);

        cusparseSpMV(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A, x,
                     &beta, d.r, compute_type, CUSPARSE_SPMV_ALG_DEFAULT,
                     buffer);

        cudaFree(buffer);
    }

    data_type norm_r_0 = 0;
    cublasDnrm2_v2(cublas, n, d.r_d, 1, &norm_r_0);
    std::cout << norm_r_0 << std::endl;

    return 0;
}
} // namespace cg_run