#include "mkl/dr_bcg.h"

#include "common/log.h"
#include "common/mkl_checks.h"
#include "common/mkl_matrices.h"
#include "common/timer.h"

#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#include <mkl_spblas.h>
#include <mkl_types.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
namespace {

enum class Transpose : std::uint8_t { True,
                                      False };

// Allocate a column-major dense matrix of given size (uninitialized)
DenseMatrix alloc_dense(MKL_INT rows, MKL_INT cols) noexcept {
    DenseMatrix m;
    m.rows = rows;
    m.cols = cols;
    m.data.resize(static_cast<size_t>(rows) * cols);
    return m;
}

// Compute Y = alpha * op(A_sparse) * X_dense + beta * Y_dense
// Uses MKL sparse BLAS (mkl_dcsrmm)
void sparse_mm(const CSRMatrix &A, Transpose op, double alpha, const DenseMatrix &X,
               double beta, DenseMatrix &Y) noexcept {
    sparse_operation_t op_type = (op == Transpose::True) ? SPARSE_OPERATION_TRANSPOSE
                                                         : SPARSE_OPERATION_NON_TRANSPOSE;

    // Output rows depend on operation
    MKL_INT out_rows = (op == Transpose::True) ? A.cols : A.rows;

    MKL_SPARSE_CHECK(mkl_sparse_d_mm(op_type, alpha, A.mat, A.descr,
                                     SPARSE_LAYOUT_COLUMN_MAJOR, X.data.data(),
                                     X.cols, // number of columns (vectors)
                                     X.rows, // leading dimension of X
                                     beta, Y.data.data(),
                                     out_rows)); // leading dimension of Y
}

// Solve op(L) * Y = X, writing result back into X.
// L is lower triangular CSR.
// MKL requires separate input/output buffers, so we allocate Y internally
// and move it into X on success.
void sparse_trsm(const CSRMatrix &L, Transpose op, DenseMatrix &X) noexcept {
    sparse_operation_t op_type = (op == Transpose::True) ? SPARSE_OPERATION_TRANSPOSE
                                                         : SPARSE_OPERATION_NON_TRANSPOSE;

    // mkl_sparse_d_trsm: y = alpha * inv(op(L)) * x
    // x and y must not overlap — use a fresh output buffer.
    DenseMatrix Y = alloc_dense(X.rows, X.cols);

    constexpr double alpha = 1.0;
    MKL_SPARSE_CHECK(mkl_sparse_d_trsm(
        op_type, alpha, L.mat, L.descr, SPARSE_LAYOUT_COLUMN_MAJOR,
        X.data.data(), X.cols, X.rows, Y.data.data(), Y.rows));

    X = std::move(Y);
}

// Compute C = alpha * A * B + beta * C  (dense matrix multiply, column-major)
// A: m x k,  B: k x n,  C: m x n
void dense_mm(Transpose op_a, Transpose op_b, MKL_INT m, MKL_INT n, MKL_INT k,
              double alpha, const double *A, MKL_INT lda, const double *B,
              MKL_INT ldb, double beta, double *C, MKL_INT ldc) noexcept {
    CBLAS_TRANSPOSE ta = (op_a == Transpose::True) ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE tb = (op_b == Transpose::True) ? CblasTrans : CblasNoTrans;
    cblas_dgemm(CblasColMajor, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C,
                ldc);
}

// Thin (economy) QR factorization of M (m x n, m >= n), column-major.
// On return:
//   Q  - m x n orthonormal matrix (replaces M in output)
//   R_out - n x n upper triangular factor
// We use LAPACK dgeqrf + dorgqr.
void thin_qr(const DenseMatrix &M, DenseMatrix &Q,
             DenseMatrix &R_out) noexcept {
    MKL_INT m = M.rows;
    MKL_INT n = M.cols;

    assert(m >= n && "thin_qr: requires m >= n");

    // Copy M into Q (we work in-place)
    Q.rows = m;
    Q.cols = n;
    Q.data = M.data; // copy

    std::vector<double> tau(n);

    // QR factorization
    MKL_LAPACKE_CHECK(
        LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, Q.data.data(), m, tau.data()));

    // Extract upper triangular R (n x n) from the upper triangle of Q
    R_out = alloc_dense(n, n);
    for (MKL_INT j = 0; j < n; ++j) {
        for (MKL_INT i = 0; i <= j; ++i) {
            R_out.data[(j * n) + i] = Q.data[(m * j) + i];
        }
        for (MKL_INT i = j + 1; i < n; ++i) {
            R_out.data[(j * n) + i] = 0.0;
        }
    }

    // Form Q explicitly
    MKL_LAPACKE_CHECK(LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, n, n, Q.data.data(),
                                     m, tau.data()));
}

// Invert a small square matrix in-place using LAPACK dgetrf + dgetri.
void invert_square(std::vector<double> &A_data, MKL_INT n) {
    std::vector<lapack_int> ipiv(n);

    MKL_LAPACKE_CHECK(
        LAPACKE_dgetrf(LAPACK_COL_MAJOR, n, n, A_data.data(), n, ipiv.data()));
    MKL_LAPACKE_CHECK(
        LAPACKE_dgetri(LAPACK_COL_MAJOR, n, A_data.data(), n, ipiv.data()));
}

} // namespace

namespace cils::mkl {

int solve(const CSRMatrix &A, const CSRMatrix &L, const DenseMatrix &B,
          DenseMatrix &X, Config config) noexcept {
    CpuTimerRange solve_range{g_timer, "solve"};

    const MKL_INT n = A.rows;
    const MKL_INT nrhs = B.cols;

    assert(X.rows == n && X.cols == nrhs &&
           "X dimensions do not match A and B");
    assert(B.rows == n && "B row count does not match A");
    assert(L.rows == n && L.cols == n && "L dimensions do not match A");

    // ------------------------------------------------------------------
    // Initialization
    // ------------------------------------------------------------------

    DenseMatrix R = alloc_dense(n, nrhs);
    {
        CpuTimerRange r_range(g_timer, "R = B - A * X");
        R = B;
        sparse_mm(A, Transpose::False, -1.0, X, 1.0, R); // R = B - A*X
    }

    // We break [w sigma] = QR(L^-1 * R) into two steps for timing purposes:
    // 1. temp = L^-1 * R
    // 2. [w sigma] = QR(temp)
    DenseMatrix w;
    DenseMatrix sigma;
    DenseMatrix temp = R;
    {
        CpuTimerRange w_sigma_range(g_timer, "temp = L^-1 * R");

        sparse_trsm(L, Transpose::False, temp);
    }

    {
        CpuTimerRange w_sigma_range(g_timer, "[w sigma] = QR(temp)");

        thin_qr(temp, w, sigma);
    }

    DenseMatrix s = w;
    {
        CpuTimerRange s_initial_range(g_timer, "s = (L^-1)' * w");
        sparse_trsm(L, Transpose::True, s);
    }

    // ------------------------------------------------------------------
    // Precompute norm of first column of B for convergence check
    // ------------------------------------------------------------------
    double b_norm = cblas_dnrm2(n, B.data.data(), 1);
    if (b_norm == 0.0) {
        b_norm = 1.0; // guard against zero rhs
    }

    int iterations = 0;

    // ------------------------------------------------------------------
    // Main iteration loop
    // ------------------------------------------------------------------
    for (int k = 0; k < config.max_iterations; ++k) {
        CpuTimerRange iteration_range(g_timer, "iteration");
        ++iterations;

        DenseMatrix As = alloc_dense(n, nrhs);
        DenseMatrix xi(alloc_dense(nrhs, nrhs));
        {
            CpuTimerRange xi_range(g_timer, "xi = (s' * As)^-1");
            // Step 1: As = A * s  (n x nrhs)
            As.data.assign(As.data.size(), 0.0);
            sparse_mm(A, Transpose::False, 1.0, s, 0.0, As);

            // Step 2: xi_inv = s' * As  (nrhs x nrhs)
            dense_mm(Transpose::True, Transpose::False, nrhs, nrhs, n, 1.0, s.data.data(), n,
                     As.data.data(), n, 0.0, xi.data.data(), nrhs);

            // Step 3: xi = xi_inv^{-1}
            invert_square(xi.data, nrhs);
        }

        DenseMatrix xi_sigma = alloc_dense(nrhs, nrhs);
        {
            CpuTimerRange x_range(g_timer, "X = X + s * xi * sigma");
            // Step a: tmp2 = xi * sigma  (nrhs x nrhs)
            dense_mm(Transpose::False, Transpose::False, nrhs, nrhs, nrhs, 1.0, xi.data.data(), nrhs,
                     sigma.data.data(), nrhs, 0.0, xi_sigma.data.data(),
                     nrhs);

            // Step b: X += s * xi_sigma  (n x nrhs)
            dense_mm(Transpose::False, Transpose::False, n, nrhs, nrhs, 1.0, s.data.data(), n,
                     xi_sigma.data.data(), nrhs, 1.0, X.data.data(), n);
        }

        // ------------------------------------------------------------------
        // Convergence check: rrn = ||B(:,1) - A*X(:,1)|| / ||B(:,1)||
        // ------------------------------------------------------------------
        double residual_norm = 0.0;
        {
            CpuTimerRange rrn_range(g_timer, "norm(B1 - A * X1) / norm(B1)");
            DenseMatrix X_col1 = alloc_dense(n, 1);
            std::copy(X.data.begin(), X.data.begin() + n, X_col1.data.begin());

            DenseMatrix r1 = alloc_dense(n, 1);
            std::copy(B.data.begin(), B.data.begin() + n, r1.data.begin());
            sparse_mm(A, Transpose::False, -1.0, X_col1, 1.0, r1);

            residual_norm = cblas_dnrm2(n, r1.data.data(), 1);
            cils::detail::log(residual_norm / b_norm);
        }

        if (residual_norm / b_norm < config.tolerance) {
            break;
        }

        // We break [w zeta] = QR(w - L^-1 * A * s * xi) into two steps for timing purposes:
        // 1. w = w - L^-1 * A * s * xi
        // 2. [w zeta] = QR(w)
        DenseMatrix zeta;
        {
            CpuTimerRange w_zeta_range(g_timer, "w = w - L^-1 * A * s * xi");

            // temp = As * xi
            dense_mm(Transpose::False, Transpose::False, n, nrhs, nrhs, 1.0, As.data.data(), n,
                     xi.data.data(), nrhs, 0.0, temp.data.data(), n);

            // temp = L^-1 * As_xi
            sparse_trsm(L, Transpose::False, temp);

            // w = w - L^{-1} * A * s * xi
            for (size_t i = 0; i < w.data.size(); ++i) {
                w.data[i] = w.data[i] - temp.data[i];
            }
        }

        {
            CpuTimerRange w_zeta_range(g_timer, "[w zeta] = QR(w)");
            thin_qr(w, w, zeta);
        }

        {
            CpuTimerRange s_range(g_timer, "s = (L^-1)' * w + s * zeta'");
            DenseMatrix Linv_T_w = w;
            sparse_trsm(L, Transpose::True, Linv_T_w); // Linv_T_w = L^{-T} * w

            // s = Linv_T_w + s * zeta'
            DenseMatrix s_new = alloc_dense(n, nrhs);
            // s_new = s * zeta^T
            dense_mm(Transpose::False, Transpose::True, n, nrhs, nrhs, 1.0, s.data.data(), n,
                     zeta.data.data(), nrhs, 0.0, s_new.data.data(), n);
            // s_new += Linv_T_w
            for (size_t i = 0; i < s_new.data.size(); ++i) {
                s_new.data[i] += Linv_T_w.data[i];
            }
            s = std::move(s_new);
        }

        {
            CpuTimerRange sigma_range(g_timer, "sigma = zeta * sigma");
            DenseMatrix sigma_new = alloc_dense(nrhs, nrhs);
            dense_mm(Transpose::False, Transpose::False, nrhs, nrhs, nrhs, 1.0, zeta.data.data(), nrhs,
                     sigma.data.data(), nrhs, 0.0, sigma_new.data.data(),
                     nrhs);
            sigma = std::move(sigma_new);
        }
    }

    return iterations;
}

} // namespace cils::mkl
