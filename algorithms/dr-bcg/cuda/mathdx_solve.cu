// Fused MathDx CholeskyQR2 DR-BCG entry points.

#include "config.h"

#include "dr_bcg/mathdx_solve.cuh"

#ifdef SOLVERS_BUILD_MATHDX

#include "dr_bcg/cuda.cuh"
#include "dr_bcg/mathdx_fused.cuh"
#include "dr_bcg/mathdx_qr.cuh"

namespace dr_bcg::cuda {

int solve_cholqr_dx(Handles &handles, cusparseSpMatDescr_t A,
                    cusparseDnMatDescr_t X, cusparseDnMatDescr_t B,
                    cusparseSpMatDescr_t L, double tolerance,
                    int max_iterations, cudaStream_t stream) {
    return solve<double, MathDxCholeskyQr2<double>>(
        handles, A, X, B, L, tolerance, max_iterations, stream);
}

int solve_cholqr_dx(Handles &handles, cusparseSpMatDescr_t A,
                    cusparseDnMatDescr_t X, cusparseDnMatDescr_t B,
                    double tolerance, int max_iterations, cudaStream_t stream) {
    return solve<double, MathDxCholeskyQr2<double>>(
        handles, A, X, B, tolerance, max_iterations, stream);
}

int solve_fused_dx(Handles &handles, cusparseSpMatDescr_t A,
                   cusparseDnMatDescr_t X, cusparseDnMatDescr_t B,
                   cusparseSpMatDescr_t L, double tolerance, int max_iterations,
                   cudaStream_t stream) {
    return solve_fused<double>(handles, A, X, B, L, tolerance, max_iterations,
                               stream);
}

int solve_fused_dx(Handles &handles, cusparseSpMatDescr_t A,
                   cusparseDnMatDescr_t X, cusparseDnMatDescr_t B,
                   double tolerance, int max_iterations, cudaStream_t stream) {
    return solve_fused<double>(handles, A, X, B, tolerance, max_iterations,
                               stream);
}

} // namespace dr_bcg::cuda

#endif // SOLVERS_BUILD_MATHDX
