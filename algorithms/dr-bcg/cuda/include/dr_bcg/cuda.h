#pragma once

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusparse_v2.h>

namespace dr_bcg::cuda {

int solve(cusparseSpMatDescr_t A, cusparseDnMatDescr_t X,
          cusparseDnMatDescr_t B, double tolerance = 1e-6,
          int max_iterations = 100, cudaStream_t stream = nullptr);

int solve(cusparseSpMatDescr_t A, cusparseDnMatDescr_t X,
          cusparseDnMatDescr_t B, cusparseSpMatDescr_t L,
          double tolerance = 1e-6, int max_iterations = 100,
          cudaStream_t stream = nullptr);

} // namespace dr_bcg::cuda
