#include "cg/cg.h"
#include <cassert>
#include <mkl_cblas.h>
#include <mkl_spblas.h>

#ifdef PRINT_RRN
#include <cmath>
#include <iostream>
#endif // PRINT_RRN

namespace cg::mkl {
int solve(const MKLSparse &A, const std::vector<double> &b,
          std::vector<double> &x, const MKLSparse &L, double tolerance,
          int max_iterations, bool real_residual) {
    assert(A.descr.type == SPARSE_MATRIX_TYPE_GENERAL);
    assert(L.descr.type == SPARSE_MATRIX_TYPE_TRIANGULAR);
    assert(L.descr.mode == SPARSE_FILL_MODE_UPPER);
    assert(L.descr.diag == SPARSE_DIAG_NON_UNIT);

    int n = static_cast<int>(b.size());

    // Relative residual threshold: ||r||^2 <= tol^2 * ||b||^2
    double norm_b_sq = cblas_ddot(n, b.data(), 1, b.data(), 1);
    double tol_sq = tolerance * tolerance * norm_b_sq;

    // Working vectors
    std::vector<double> r(n);   // residual
    std::vector<double> d(n);   // search direction
    std::vector<double> q(n);   // A * d
    std::vector<double> s(n);   // preconditioned residual: M^{-1} r
    std::vector<double> tmp(n); // intermediate for triangular solves

    // Apply preconditioner: z = L^{-T} L^{-1} rhs  (i.e. M^{-1} rhs)
    // mkl_L stores L^T (upper triangular), so:
    //   Step 1: solve L * y   = rhs  → (L^T)^T * y = rhs  → TRANSPOSE on mkl_L
    //   Step 2: solve L^T * z = y    → NON_TRANSPOSE on mkl_L
    auto apply_precond = [&L, &tmp](const std::vector<double> &rhs,
                                    std::vector<double> &result) {
        mkl_sparse_d_trsv(SPARSE_OPERATION_TRANSPOSE, 1.0, L.mat, L.descr,
                          rhs.data(), tmp.data());
        mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, L.mat, L.descr,
                          tmp.data(), result.data());
    };

    // r = b - A * x
    cblas_dcopy(n, b.data(), 1, r.data(), 1);
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A.mat, A.descr,
                    x.data(), 1.0, r.data());

    // d = M^{-1} * r
    apply_precond(r, d);

    double delta_new = cblas_ddot(n, r.data(), 1, d.data(), 1);
    double residual_sq = cblas_ddot(n, r.data(), 1, r.data(), 1);

    int iter;
    for (iter = 0; iter < max_iterations; ++iter) {
#ifdef PRINT_RRN
        std::cerr << std::sqrt(residual_sq / norm_b_sq) << std::endl;
#endif // PRINT_RRN

        if (residual_sq <= tol_sq) {
            break;
        }

        // q = A * d
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A.mat, A.descr,
                        d.data(), 0.0, q.data());

        double alpha = delta_new / cblas_ddot(n, d.data(), 1, q.data(), 1);

        // x = x + alpha * d
        cblas_daxpy(n, alpha, d.data(), 1, x.data(), 1);

        if (real_residual) {
            // r = b - A * x  (exact recomputation avoids accumulated error)
            cblas_dcopy(n, b.data(), 1, r.data(), 1);
            mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A.mat,
                            A.descr, x.data(), 1.0, r.data());
        } else {
            // r = r - alpha * q
            cblas_daxpy(n, -alpha, q.data(), 1, r.data(), 1);
        }

        residual_sq = cblas_ddot(n, r.data(), 1, r.data(), 1);

        // s = M^{-1} * r
        apply_precond(r, s);

        double delta_old = delta_new;
        delta_new = cblas_ddot(n, r.data(), 1, s.data(), 1);
        double beta = delta_new / delta_old;

        // d = s + beta * d
        cblas_dscal(n, beta, d.data(), 1);
        cblas_daxpy(n, 1.0, s.data(), 1, d.data(), 1);
    }

    return iter;
}
} // namespace cg::mkl
