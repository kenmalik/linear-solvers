#ifndef CG_H
#define CG_H

#include <mkl_types.h>
#include <vector>

/// Conjugate Gradient solver using Intel MKL.
/// Solves Ax = b for symmetric positive-definite matrix A.
///
/// @param A           n x n matrix in row-major order (size n*n)
/// @param b           right-hand side vector (size n)
/// @param x           initial guess on input, solution on output (size n)
/// @param tolerance   convergence tolerance for residual norm
/// @param max_iterations maximum number of iterations
/// @return number of iterations performed
int cg(const std::vector<double> &A, const std::vector<double> &b,
       std::vector<double> &x, double tolerance = 1e-6,
       int max_iterations = 100);

/// Sparse CSR matrix representation.
struct CsrMatrix {
    int n;                          ///< matrix dimension (n x n)
    std::vector<double> values;     ///< non-zero values
    std::vector<MKL_INT64> col_idx; ///< column indices of non-zero values
    std::vector<MKL_INT64> row_ptr; ///< row pointers (size n+1)
};

/// Conjugate Gradient solver for sparse CSR matrix A.
/// Solves Ax = b for symmetric positive-definite matrix A.
///
/// @param A           n x n symmetric positive-definite matrix in CSR format
/// @param b           right-hand side vector (size n)
/// @param x           initial guess on input, solution on output (size n)
/// @param tolerance   convergence tolerance for residual norm
/// @param max_iterations maximum number of iterations
/// @return number of iterations performed
int cg(CsrMatrix &A, const std::vector<double> &b, std::vector<double> &x,
       double tolerance = 1e-6, int max_iterations = 100);

#ifdef USE_MAT_UTILS

#include <mat_utils/mat_reader.h>

int cg(const mat_utils::SpMatReader &A, const std::vector<double> &b,
       std::vector<double> &x, double tolerance = 1e-6,
       int max_iterations = 100);

#endif // USE_MAT_UTILS

#endif // CG_H
