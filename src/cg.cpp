#include "cg/cg.h"
#include <mkl_cblas.h>
#include <stdexcept>

int cg(std::vector<double>& A, std::vector<double>& b, std::vector<double>& x,
       double tolerance, int max_iterations) {

    int n = static_cast<int>(b.size());

    if (A.size() != static_cast<size_t>(n * n)) {
        throw std::invalid_argument("Matrix A must be n x n where n is the size of b");
    }
    if (x.size() != static_cast<size_t>(n)) {
        throw std::invalid_argument("Vector x must have the same size as b");
    }

    // Working vectors
    std::vector<double> r(n);   // residual
    std::vector<double> p(n);   // search direction
    std::vector<double> Ap(n);  // A * p

    // r = b - A * x
    cblas_dcopy(n, b.data(), 1, r.data(), 1);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, -1.0, A.data(), n,
                x.data(), 1, 1.0, r.data(), 1);

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
