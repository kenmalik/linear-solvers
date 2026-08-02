#pragma once

#include <cublas_v2.h>
#include <cusparse_v2.h>

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
