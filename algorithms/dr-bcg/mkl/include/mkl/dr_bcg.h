#pragma once

#include <vector>

#include <mkl.h>
#include <mkl_spblas.h>

namespace dr_bcg::mkl {

// CSR sparse matrix descriptor
struct CSRMatrix {
    MKL_INT rows;
    MKL_INT cols;
    std::vector<double> values;
    std::vector<MKL_INT> row_ptr;
    std::vector<MKL_INT> col_idx;
};

// Dense matrix stored in column-major order (Fortran layout for LAPACK/BLAS)
// Element (i, j) is at data[j * rows + i]
struct DenseMatrix {
    MKL_INT rows;
    MKL_INT cols;
    std::vector<double> data;
};

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
