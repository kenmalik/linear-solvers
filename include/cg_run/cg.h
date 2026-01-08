#pragma once

#include <cublas_v2.h>
#include <cusparse_v2.h>

namespace cg_run {
int cg(cusparseHandle_t cusparse, cublasHandle_t cublas, cusparseSpMatDescr_t A,
       cusparseDnVecDescr_t x, cusparseSpMatDescr_t R, double tolerance = 1e-6,
       int max_iterations = 1);
}