#pragma once

#include <cublas_v2.h>
#include <cusparse_v2.h>

namespace cg::cuda {
int solve(cusparseHandle_t cusparse, cublasHandle_t cublas, cusparseSpMatDescr_t A,
       cusparseDnVecDescr_t b, cusparseDnVecDescr_t x, cusparseSpMatDescr_t L,
       double tolerance = 1e-6, int max_iterations = 1,
       bool real_residual = true);
}
