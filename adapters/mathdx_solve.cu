// Fused MathDx CholeskyQR2 DR-BCG entry points.

#include "config.h"

#ifdef SOLVERS_BUILD_MATHDX

#include "mathdx_solve.cuh"

#include "cuda/detail/mathdx_qr.cuh"
#include "cuda/dr_bcg.cuh"
#include "cuda/mathdx_fused.cuh"

namespace cils::cuda::detail {

int solve_cholqr_dx(Handles &handles, cusparseSpMatDescr_t A,
                    cusparseDnMatDescr_t X, cusparseDnMatDescr_t B,
                    cusparseSpMatDescr_t L, double tolerance,
                    int max_iterations, cudaStream_t stream) {
    return dr_bcg<double, MathDxCholeskyQr2<double>>(
        handles, A, X, B, L, tolerance, max_iterations, stream);
}

int solve_cholqr_dx(Handles &handles, cusparseSpMatDescr_t A,
                    cusparseDnMatDescr_t X, cusparseDnMatDescr_t B,
                    double tolerance, int max_iterations, cudaStream_t stream) {
    return dr_bcg<double, MathDxCholeskyQr2<double>>(
        handles, A, X, B, tolerance, max_iterations, stream);
}

int solve_fused_dx(Handles &handles, cusparseSpMatDescr_t A,
                   cusparseDnMatDescr_t X, cusparseDnMatDescr_t B,
                   cusparseSpMatDescr_t L, double tolerance, int max_iterations,
                   FusedXiQr qr, cudaStream_t stream) {
    switch (qr) {
    case FusedXiQr::Householder:
        return solve_fused<double, HouseholderQr<double>>(
            handles, A, X, B, L, tolerance, max_iterations, stream);
    case FusedXiQr::CholQR:
        return solve_fused<double, CholeskyQr<double>>(
            handles, A, X, B, L, tolerance, max_iterations, stream);
    case FusedXiQr::CholQRDx:
    default:
        return solve_fused<double, MathDxCholeskyQr2<double>>(
            handles, A, X, B, L, tolerance, max_iterations, stream);
    }
}

int solve_fused_dx(Handles &handles, cusparseSpMatDescr_t A,
                   cusparseDnMatDescr_t X, cusparseDnMatDescr_t B,
                   double tolerance, int max_iterations, FusedXiQr qr,
                   cudaStream_t stream) {
    switch (qr) {
    case FusedXiQr::Householder:
        return solve_fused<double, HouseholderQr<double>>(
            handles, A, X, B, tolerance, max_iterations, stream);
    case FusedXiQr::CholQR:
        return solve_fused<double, CholeskyQr<double>>(
            handles, A, X, B, tolerance, max_iterations, stream);
    case FusedXiQr::CholQRDx:
    default:
        return solve_fused<double, MathDxCholeskyQr2<double>>(
            handles, A, X, B, tolerance, max_iterations, stream);
    }
}

} // namespace cils::cuda::detail

#endif // SOLVERS_BUILD_MATHDX
