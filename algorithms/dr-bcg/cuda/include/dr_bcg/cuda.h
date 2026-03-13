#pragma once

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusparse_v2.h>

#include <functional>

namespace dr_bcg::cuda {

int dr_bcg(cusparseSpMatDescr_t A, cusparseDnMatDescr_t X,
           cusparseDnMatDescr_t B, float tolerance = 1e-6,
           int max_iterations = 100,
           std::function<void(int, float)> residual_callback = nullptr);

int dr_bcg(cusparseSpMatDescr_t A, cusparseDnMatDescr_t X,
           cusparseDnMatDescr_t B, double tolerance = 1e-6,
           int max_iterations = 100,
           std::function<void(int, double)> residual_callback = nullptr);

int solve(cusparseSpMatDescr_t A, cusparseDnMatDescr_t X,
           cusparseDnMatDescr_t B, cusparseSpMatDescr_t L,
           double tolerance = 1e-6, int max_iterations = 100);

int dr_bcg(cusparseSpMatDescr_t A, cusparseDnMatDescr_t X,
           cusparseDnMatDescr_t B, cusparseSpMatDescr_t L,
           float tolerance = 1e-6, int max_iterations = 100,
           std::function<void(int, float)> residual_callback = nullptr);

} // namespace dr_bcg::cuda
