#include "cg/cg.h"
#include <mkl_cblas.h>
#include <mkl_spblas.h>
#include <stdexcept>

int cg(const std::vector<double> &A, const std::vector<double> &b,
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

int cg(CsrMatrix &A, const std::vector<double> &b, std::vector<double> &x,
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

#ifdef USE_MAT_UTILS

int cg(const mat_utils::SpMatReader &A, const std::vector<double> &b,
       std::vector<double> &x, double tolerance, int max_iterations) {

    // SpMatReader accessors are not const-qualified, so cast away const.
    // The accessors are read-only; this is safe.
    auto &A_mut = const_cast<mat_utils::SpMatReader &>(A);

    int n = static_cast<int>(b.size());

    if (A_mut.rows() != static_cast<size_t>(n) ||
        A_mut.cols() != static_cast<size_t>(n)) {
        throw std::invalid_argument("Matrix A dimension must match size of b");
    }
    if (x.size() != static_cast<size_t>(n)) {
        throw std::invalid_argument("Vector x must have the same size as b");
    }

    // SpMatReader stores the matrix in MATLAB's CSC format:
    //   jc: column pointers (size cols+1)
    //   ir: row indices    (size nnz)
    //
    // Since A is symmetric (SPD), treating the CSC arrays as CSR gives Aᵀ = A,
    // so we can use mkl_sparse_d_create_csr directly with jc/ir.
    //
    // The index arrays are size_t; copy to MKL_INT64 as required by the API.
    const size_t *jc = A_mut.jc();
    const size_t *ir = A_mut.ir();
    std::vector<MKL_INT64> row_ptr(jc, jc + n + 1);
    std::vector<MKL_INT64> col_idx(ir, ir + A_mut.nnz());

    sparse_matrix_t mkl_A;
    mkl_sparse_d_create_csr(&mkl_A, SPARSE_INDEX_BASE_ZERO, n, n,
                            row_ptr.data(), row_ptr.data() + 1, col_idx.data(),
                            A_mut.data());

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

#endif // USE_MAT_UTILS
