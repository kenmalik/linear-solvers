#include "dr_bcg/mkl.h"

#include "common/mkl_checks.h"

#include <cassert>
#include <cstring>

#include <mkl.h>
#include <mkl_lapacke.h>
#include <mkl_spblas.h>

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
namespace {
// Allocate a column-major dense matrix of given size (uninitialized)
DenseMatrix alloc_dense(MKL_INT rows, MKL_INT cols) noexcept {
    DenseMatrix m;
    m.rows = rows;
    m.cols = cols;
    m.data.resize(static_cast<size_t>(rows) * cols);
    return m;
}

// Copy src into dst (must have identical dimensions)
void copy_dense(DenseMatrix &dst, const DenseMatrix &src) noexcept {
    dst.rows = src.rows;
    dst.cols = src.cols;
    dst.data = src.data;
}

// Compute Y = alpha * op(A_sparse) * X_dense + beta * Y_dense
// op: 'N' = no transpose, 'T' = transpose
// Uses MKL sparse BLAS (mkl_dcsrmm)
void sparse_mm(const CSRMatrix &A, char op, double alpha, const DenseMatrix &X,
               double beta, DenseMatrix &Y) noexcept {
    sparse_operation_t op_type = (op == 'T') ? SPARSE_OPERATION_TRANSPOSE
                                             : SPARSE_OPERATION_NON_TRANSPOSE;

    // Output rows depend on operation
    MKL_INT out_rows = (op == 'T') ? A.cols : A.rows;

    MKL_SPARSE_CHECK(mkl_sparse_d_mm(op_type, alpha, A.mat, A.descr,
                                     SPARSE_LAYOUT_COLUMN_MAJOR, X.data.data(),
                                     X.cols, // number of columns (vectors)
                                     X.rows, // leading dimension of X
                                     beta, Y.data.data(),
                                     out_rows)); // leading dimension of Y
}

// Solve op(L) * Y = X, writing result back into X.
// op: 'N' => L*Y=X (forward solve), 'T' => L^T*Y=X (backward solve)
// L is lower triangular CSR.
// MKL requires separate input/output buffers, so we allocate Y internally
// and move it into X on success.
void sparse_trsm(const CSRMatrix &L, char op, DenseMatrix &X) noexcept {
    sparse_operation_t op_type = (op == 'T') ? SPARSE_OPERATION_TRANSPOSE
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
// op_a / op_b: 'N' or 'T'
void dense_mm(char op_a, char op_b, MKL_INT m, MKL_INT n, MKL_INT k,
              double alpha, const double *A, MKL_INT lda, const double *B,
              MKL_INT ldb, double beta, double *C, MKL_INT ldc) noexcept {
    CBLAS_TRANSPOSE ta = (op_a == 'T') ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE tb = (op_b == 'T') ? CblasTrans : CblasNoTrans;
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
        for (MKL_INT i = 0; i <= j; ++i)
            R_out.data[j * n + i] = Q.data[j * m + i];
        for (MKL_INT i = j + 1; i < n; ++i)
            R_out.data[j * n + i] = 0.0;
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

// ---------------------------------------------------------------------------
// PDR-BCG implementation
// ---------------------------------------------------------------------------
namespace dr_bcg::mkl {
int solve(const CSRMatrix &A, const CSRMatrix &L, const DenseMatrix &B,
          DenseMatrix &X, double tolerance, int max_iterations) noexcept {
    const MKL_INT n = A.rows;
    const MKL_INT nrhs = B.cols;

    assert(X.rows == n && X.cols == nrhs &&
           "X dimensions do not match A and B");
    assert(B.rows == n && "B row count does not match A");
    assert(L.rows == n && L.cols == n && "L dimensions do not match A");

    // ------------------------------------------------------------------
    // Initialization
    // ------------------------------------------------------------------

    // R = B - A * X
    DenseMatrix R = alloc_dense(n, nrhs);
    copy_dense(R, B);                   // R = B
    sparse_mm(A, 'N', -1.0, X, 1.0, R); // R = B - A*X

    // tmp = L^{-1} * R   (forward triangular solve: L * tmp = R)
    DenseMatrix tmp = R;      // copy R
    sparse_trsm(L, 'N', tmp); // tmp = L^{-1} * R

    // [w, sigma] = thin QR of tmp
    DenseMatrix w, sigma;
    thin_qr(tmp, w, sigma); // w: n x nrhs, sigma: nrhs x nrhs

    // s = (L^{-1})^T * w   (backward triangular solve: L^T * s = w)
    DenseMatrix s = w;      // copy w
    sparse_trsm(L, 'T', s); // s = L^{-T} * w

    // ------------------------------------------------------------------
    // Precompute norm of first column of B for convergence check
    // ------------------------------------------------------------------
    double b_norm = cblas_dnrm2(n, B.data.data(), 1);
    if (b_norm == 0.0)
        b_norm = 1.0; // guard against zero rhs

    int iterations = 0;

    // ------------------------------------------------------------------
    // Main iteration loop
    // ------------------------------------------------------------------
    for (int k = 0; k < max_iterations; ++k) {
        ++iterations;

        // xi = (s' * A * s)^{-1}   (nrhs x nrhs matrix)
        // Step 1: As = A * s  (n x nrhs)
        DenseMatrix As = alloc_dense(n, nrhs);
        As.data.assign(As.data.size(), 0.0);
        sparse_mm(A, 'N', 1.0, s, 0.0, As);

        // Step 2: xi_inv = s' * As  (nrhs x nrhs)
        DenseMatrix xi(alloc_dense(nrhs, nrhs));
        dense_mm('T', 'N', nrhs, nrhs, n, 1.0, s.data.data(), n, As.data.data(),
                 n, 0.0, xi.data.data(), nrhs);

        // Step 3: xi = xi_inv^{-1}
        invert_square(xi.data, nrhs);

        // X = X + s * xi * sigma
        // Step a: tmp2 = xi * sigma  (nrhs x nrhs)
        DenseMatrix xi_sigma = alloc_dense(nrhs, nrhs);
        dense_mm('N', 'N', nrhs, nrhs, nrhs, 1.0, xi.data.data(), nrhs,
                 sigma.data.data(), nrhs, 0.0, xi_sigma.data.data(), nrhs);

        // Step b: X += s * xi_sigma  (n x nrhs)
        dense_mm('N', 'N', n, nrhs, nrhs, 1.0, s.data.data(), n,
                 xi_sigma.data.data(), nrhs, 1.0, X.data.data(), n);

        // ------------------------------------------------------------------
        // Convergence check: rrn = ||B(:,1) - A*X(:,1)|| / ||B(:,1)||
        // ------------------------------------------------------------------
        DenseMatrix X_col1 = alloc_dense(n, 1);
        std::copy(X.data.begin(), X.data.begin() + n, X_col1.data.begin());

        DenseMatrix r1 = alloc_dense(n, 1);
        std::copy(B.data.begin(), B.data.begin() + n, r1.data.begin());
        sparse_mm(A, 'N', -1.0, X_col1, 1.0, r1);

        double rrn = cblas_dnrm2(n, r1.data.data(), 1) / b_norm;

        if (rrn < tolerance)
            break;

        // ------------------------------------------------------------------
        // Update w and s for next iteration
        // ------------------------------------------------------------------
        // tmp = L^{-1} * A * s * xi   (n x nrhs)
        //     = L^{-1} * As * xi
        // Step 1: As_xi = As * xi  (n x nrhs)
        DenseMatrix As_xi = alloc_dense(n, nrhs);
        dense_mm('N', 'N', n, nrhs, nrhs, 1.0, As.data.data(), n,
                 xi.data.data(), nrhs, 0.0, As_xi.data.data(), n);

        // Step 2: L^{-1} * As_xi  (forward solve)
        sparse_trsm(L, 'N', As_xi); // As_xi = L^{-1} * A * s * xi

        // w_new_input = w - L^{-1} * A * s * xi
        DenseMatrix w_new_input = alloc_dense(n, nrhs);
        for (size_t i = 0; i < w_new_input.data.size(); ++i)
            w_new_input.data[i] = w.data[i] - As_xi.data[i];

        // [w, zeta] = thin QR(w_new_input)
        DenseMatrix zeta;
        thin_qr(w_new_input, w, zeta); // w: n x nrhs, zeta: nrhs x nrhs

        // s = L^{-T} * w + s * zeta'
        DenseMatrix Linv_T_w = w;
        sparse_trsm(L, 'T', Linv_T_w); // Linv_T_w = L^{-T} * w

        // s = Linv_T_w + s * zeta'
        DenseMatrix s_new = alloc_dense(n, nrhs);
        // s_new = s * zeta^T
        dense_mm('N', 'T', n, nrhs, nrhs, 1.0, s.data.data(), n,
                 zeta.data.data(), nrhs, 0.0, s_new.data.data(), n);
        // s_new += Linv_T_w
        for (size_t i = 0; i < s_new.data.size(); ++i)
            s_new.data[i] += Linv_T_w.data[i];
        s = std::move(s_new);

        // sigma = zeta * sigma
        DenseMatrix sigma_new = alloc_dense(nrhs, nrhs);
        dense_mm('N', 'N', nrhs, nrhs, nrhs, 1.0, zeta.data.data(), nrhs,
                 sigma.data.data(), nrhs, 0.0, sigma_new.data.data(), nrhs);
        sigma = std::move(sigma_new);
    }

    return iterations;
}
} // namespace dr_bcg::mkl
