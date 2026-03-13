#pragma once

#include <mkl_spblas.h>
#include <mkl_types.h>
#include <vector>

#include "common/mkl_matrices.h"

namespace cg::mkl {
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
int solve(const CSRMatrix &A, const std::vector<double> &b,
          std::vector<double> &x, const CSRMatrix &L, double tolerance = 1e-6,
          int max_iterations = 100, bool real_residual = false);
} // namespace cg::mkl
