#include "cg/cg.h"
#include <cassert>
#include <mkl_cblas.h>
#include <mkl_spblas.h>
#include <stdexcept>

#ifdef PRINT_RRN
#include <cmath>
#include <iostream>
#endif // PRINT_RRN

namespace cg::mkl {
int solve(const std::vector<double> &A, const std::vector<double> &b,
          std::vector<double> &x, double tolerance, int max_iterations) {

    int n = static_cast<int>(b.size());

    if (A.size() != static_cast<size_t>(n * n)) {
        throw std::invalid_argument(
            "Matrix A must be n x n where n is the size of b");
    }
    if (x.size() != static_cast<size_t>(n)) {
        throw std::invalid_argument("Vector x must have the same size as b");
    }

    // Working vectors
    std::vector<double> r(n);  // residual
    std::vector<double> p(n);  // search direction
    std::vector<double> Ap(n); // A * p

    // r = b - A * x
    cblas_dcopy(n, b.data(), 1, r.data(), 1);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, -1.0, A.data(), n, x.data(),
                1, 1.0, r.data(), 1);

    // p = r
    cblas_dcopy(n, r.data(), 1, p.data(), 1);

    // rs_old = r' * r
    double rs_old = cblas_ddot(n, r.data(), 1, r.data(), 1);

    double tol_squared = tolerance * tolerance;

    int iter;
    for (iter = 0; iter < max_iterations; ++iter) {
        // Check convergence: ||r||^2 < tolerance^2
        if (rs_old < tol_squared) {
            break;
        }

        // Ap = A * p
        cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, 1.0, A.data(), n,
                    p.data(), 1, 0.0, Ap.data(), 1);

        // alpha = rs_old / (p' * Ap)
        double pAp = cblas_ddot(n, p.data(), 1, Ap.data(), 1);
        double alpha = rs_old / pAp;

        // x = x + alpha * p
        cblas_daxpy(n, alpha, p.data(), 1, x.data(), 1);

        // r = r - alpha * Ap
        cblas_daxpy(n, -alpha, Ap.data(), 1, r.data(), 1);

        // rs_new = r' * r
        double rs_new = cblas_ddot(n, r.data(), 1, r.data(), 1);

        // beta = rs_new / rs_old
        double beta = rs_new / rs_old;

        // p = r + beta * p
        cblas_dscal(n, beta, p.data(), 1);
        cblas_daxpy(n, 1.0, r.data(), 1, p.data(), 1);

        rs_old = rs_new;
    }

    return iter;
}

int solve(CsrMatrix &A, const std::vector<double> &b, std::vector<double> &x,
          double tolerance, int max_iterations) {

    int n = static_cast<int>(b.size());

    if (A.n != n) {
        throw std::invalid_argument("Matrix A dimension must match size of b");
    }
    if (x.size() != static_cast<size_t>(n)) {
        throw std::invalid_argument("Vector x must have the same size as b");
    }
    if (static_cast<int>(A.row_ptr.size()) != n + 1) {
        throw std::invalid_argument("row_ptr must have size n+1");
    }

    // Create MKL sparse matrix handle from CSR data.
    // mkl_sparse_d_create_csr expects rows_start = row_ptr[0..n-1]
    // and rows_end = row_ptr[1..n].
    sparse_matrix_t mkl_A;
    mkl_sparse_d_create_csr(&mkl_A, SPARSE_INDEX_BASE_ZERO, n, n,
                            A.row_ptr.data(), A.row_ptr.data() + 1,
                            A.col_idx.data(), A.values.data());

    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    // Working vectors
    std::vector<double> r(n);  // residual
    std::vector<double> p(n);  // search direction
    std::vector<double> Ap(n); // A * p

    // r = b - A * x
    cblas_dcopy(n, b.data(), 1, r.data(), 1);
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, mkl_A, descr,
                    x.data(), 1.0, r.data());

    // p = r
    cblas_dcopy(n, r.data(), 1, p.data(), 1);

    // rs_old = r' * r
    double rs_old = cblas_ddot(n, r.data(), 1, r.data(), 1);

    double tol_squared = tolerance * tolerance;

    int iter;
    for (iter = 0; iter < max_iterations; ++iter) {
        // Check convergence: ||r||^2 < tolerance^2
        if (rs_old < tol_squared) {
            break;
        }

        // Ap = A * p
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, mkl_A, descr,
                        p.data(), 0.0, Ap.data());

        // alpha = rs_old / (p' * Ap)
        double pAp = cblas_ddot(n, p.data(), 1, Ap.data(), 1);
        double alpha = rs_old / pAp;

        // x = x + alpha * p
        cblas_daxpy(n, alpha, p.data(), 1, x.data(), 1);

        // r = r - alpha * Ap
        cblas_daxpy(n, -alpha, Ap.data(), 1, r.data(), 1);

        // rs_new = r' * r
        double rs_new = cblas_ddot(n, r.data(), 1, r.data(), 1);

        // beta = rs_new / rs_old
        double beta = rs_new / rs_old;

        // p = r + beta * p
        cblas_dscal(n, beta, p.data(), 1);
        cblas_daxpy(n, 1.0, r.data(), 1, p.data(), 1);

        rs_old = rs_new;
    }

    mkl_sparse_destroy(mkl_A);
    return iter;
}

// int solve(const mat_utils::SpMatReader &A, const std::vector<double> &b,
//           std::vector<double> &x, double tolerance, int max_iterations) {
//
//     // SpMatReader accessors are not const-qualified, so cast away const.
//     // The accessors are read-only; this is safe.
//     auto &A_mut = const_cast<mat_utils::SpMatReader &>(A);
//
//     int n = static_cast<int>(b.size());
//
//     if (A_mut.rows() != static_cast<size_t>(n) ||
//         A_mut.cols() != static_cast<size_t>(n)) {
//         throw std::invalid_argument("Matrix A dimension must match size of
//         b");
//     }
//     if (x.size() != static_cast<size_t>(n)) {
//         throw std::invalid_argument("Vector x must have the same size as b");
//     }
//
//     // SpMatReader stores the matrix in MATLAB's CSC format:
//     //   jc: column pointers (size cols+1)
//     //   ir: row indices    (size nnz)
//     //
//     // Since A is symmetric (SPD), treating the CSC arrays as CSR gives Aᵀ =
//     A,
//     // so we can use mkl_sparse_d_create_csr directly with jc/ir.
//     //
//     // The index arrays are size_t; copy to MKL_INT64 as required by the API.
//     const size_t *jc = A_mut.jc();
//     const size_t *ir = A_mut.ir();
//     std::vector<MKL_INT64> row_ptr(jc, jc + n + 1);
//     std::vector<MKL_INT64> col_idx(ir, ir + A_mut.nnz());
//
//     sparse_matrix_t mkl_A;
//     mkl_sparse_d_create_csr(&mkl_A, SPARSE_INDEX_BASE_ZERO, n, n,
//                             row_ptr.data(), row_ptr.data() + 1,
//                             col_idx.data(), A_mut.data());
//
//     struct matrix_descr descr;
//     descr.type = SPARSE_MATRIX_TYPE_GENERAL;
//
//     // Working vectors
//     std::vector<double> r(n);  // residual
//     std::vector<double> p(n);  // search direction
//     std::vector<double> Ap(n); // A * p
//
//     // r = b - A * x
//     cblas_dcopy(n, b.data(), 1, r.data(), 1);
//     mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, mkl_A, descr,
//                     x.data(), 1.0, r.data());
//
//     // p = r
//     cblas_dcopy(n, r.data(), 1, p.data(), 1);
//
//     // rs_old = r' * r
//     double rs_old = cblas_ddot(n, r.data(), 1, r.data(), 1);
//
//     double tol_squared = tolerance * tolerance;
//
//     int iter;
//     for (iter = 0; iter < max_iterations; ++iter) {
//         // Check convergence: ||r||^2 < tolerance^2
//         if (rs_old < tol_squared) {
//             break;
//         }
//
//         // Ap = A * p
//         mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, mkl_A, descr,
//                         p.data(), 0.0, Ap.data());
//
//         // alpha = rs_old / (p' * Ap)
//         double pAp = cblas_ddot(n, p.data(), 1, Ap.data(), 1);
//         double alpha = rs_old / pAp;
//
//         // x = x + alpha * p
//         cblas_daxpy(n, alpha, p.data(), 1, x.data(), 1);
//
//         // r = r - alpha * Ap
//         cblas_daxpy(n, -alpha, Ap.data(), 1, r.data(), 1);
//
//         // rs_new = r' * r
//         double rs_new = cblas_ddot(n, r.data(), 1, r.data(), 1);
//
//         // beta = rs_new / rs_old
//         double beta = rs_new / rs_old;
//
//         // p = r + beta * p
//         cblas_dscal(n, beta, p.data(), 1);
//         cblas_daxpy(n, 1.0, r.data(), 1, p.data(), 1);
//
//         rs_old = rs_new;
//     }
//
//     mkl_sparse_destroy(mkl_A);
//     return iter;
// }

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
