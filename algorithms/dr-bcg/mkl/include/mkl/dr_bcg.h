#pragma once

#include <mkl.h>
#include <mkl_spblas.h>

#include "common/mkl_matrices.h"

namespace dr_bcg::mkl {
/// PDR-BCG: Preconditioned Dubrulle-R Block Conjugate Gradient
///
/// Solves A * X = B using L as the Cholesky factor of the preconditioner M = L
/// * L^T. A and L are provided as CSR sparse matrices; B and X are dense
/// column-major matrices. X is updated in-place with the solution.
///
/// @param A symmetric positive definite system matrix (CSR)
/// @param L lower triangular Cholesky factor of preconditioner (CSR)
/// @param B right-hand side matrix (n x nrhs, column-major)
/// @param X initial guess / solution matrix (n x nrhs, column-major), updated
/// in-place
/// @param tolerance convergence threshold on relative residual norm of first
/// column max_iterations- maximum number of iterations
///
/// @return number of iterations performed
int PDR_BCG(const CSRMatrix &A, const CSRMatrix &L, const DenseMatrix &B,
            DenseMatrix &X, double tolerance, int max_iterations);
} // namespace dr_bcg::mkl
