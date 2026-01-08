#include <cg_run/cg.h>

namespace cg_run {
// A is
// R is lower triangular incomplete Cholesky where A ~= M = R^TR
using data_type = double;
int cg(cusparseHandle_t cusparse, cublasHandle_t cublas,
       const cusparseSpMatDescr_t &A, const cusparseSpMatDescr_t &R,
       data_type tolerance, int max_iterations) {
   constexpr cudaDataType_t compute_type = CUDA_R_64F;

   cusparseSpSVDescr_t SV_R_descr;
   cusparseSpSVDescr_t SV_RT_descr;
   cusparseSpSV_createDescr(&SV_R_descr);
   cusparseSpSV_createDescr(&SV_RT_descr);

   return 0;
}
} // namespace cg_run