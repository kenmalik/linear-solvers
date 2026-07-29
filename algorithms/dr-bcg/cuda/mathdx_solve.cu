// Fused MathDx CholeskyQR2 DR-BCG entry points.

#include "config.h"

#include "dr_bcg/mathdx_solve.cuh"

#ifdef SOLVERS_BUILD_MATHDX

#include "dr_bcg/cuda.cuh"
#include "dr_bcg/mathdx_fused.cuh"
#include "dr_bcg/mathdx_qr.cuh"

namespace cils::dr_bcg::cuda::detail {

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

} // namespace cils::dr_bcg::cuda::detail

#endif // SOLVERS_BUILD_MATHDX
