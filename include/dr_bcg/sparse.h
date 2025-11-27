#pragma once

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusparse_v2.h>

namespace dr_bcg {

int dr_bcg(cusparseSpMatDescr_t A, cusparseDnMatDescr_t X,
           cusparseDnMatDescr_t B, float tolerance = 1e-6,
           int max_iterations = 100);

int dr_bcg(cusparseSpMatDescr_t A, cusparseDnMatDescr_t X,
           cusparseDnMatDescr_t B, double tolerance = 1e-6,
           int max_iterations = 100);

}