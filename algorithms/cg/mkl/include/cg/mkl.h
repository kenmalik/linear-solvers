#include <mat_utils/mat_reader.h>
#include <mkl_spblas.h>
#include <mkl_types.h>
#include <vector>

namespace cg::mkl {
struct MKLSparse {
    sparse_matrix_t mat;
    struct matrix_descr descr;
    std::vector<MKL_INT> row_ptr;
    std::vector<MKL_INT> col_idx;

    explicit MKLSparse(const mat_utils::SpMatReader &reader) {
        // SpMatReader accessors are not const-qualified, so cast away const.
        // The accessors are read-only; this is safe.
        auto &A_mut = const_cast<mat_utils::SpMatReader &>(reader);

        // SpMatReader stores the matrix in MATLAB's CSC format:
        //   jc: column pointers (size cols+1)
        //   ir: row indices    (size nnz)
        //
        // Since A is symmetric (SPD), treating the CSC arrays as CSR gives Aᵀ =
        // A, so we can use mkl_sparse_d_create_csr directly with jc/ir.
        //
        // The index arrays are size_t; copy to MKL_INT64 as required by the
        // API.
        const size_t *jc = A_mut.jc();
        const size_t *ir = A_mut.ir();
        row_ptr.assign(jc, jc + A_mut.jc_size());
        col_idx.assign(ir, ir + A_mut.ir_size());

        mkl_sparse_d_create_csr(
            &mat, SPARSE_INDEX_BASE_ZERO, A_mut.rows(), A_mut.cols(),
            row_ptr.data(), row_ptr.data() + 1, col_idx.data(), A_mut.data());

        descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    }

    ~MKLSparse() { mkl_sparse_destroy(mat); }
};

/// Preconditioned Conjugate Gradient solver for sparse A with incomplete
/// Cholesky factor L (preconditioner M = L * L^T).
/// Solves Ax = b for symmetric positive-definite matrix A.
/// Convergence is checked against a relative residual: ||r|| <= tol * ||b||.
///
/// @param A              n x n symmetric positive-definite matrix
/// @param b              right-hand side vector (size n)
/// @param x              initial guess on input, solution on output (size n)
/// @param L              lower triangular Cholesky factor of preconditioner
/// @param tolerance      relative residual convergence tolerance
/// @param max_iterations maximum number of iterations
/// @param real_residual  if true, recompute r = b - A*x exactly each iteration
///                       instead of updating r = r - alpha*q (avoids drift)
/// @return number of iterations performed
int solve(const MKLSparse &A, const std::vector<double> &b,
          std::vector<double> &x, const MKLSparse &L, double tolerance = 1e-6,
          int max_iterations = 100, bool real_residual = false);
} // namespace cg::mkl
