#ifndef CG_H
#define CG_H

#include <vector>

// Conjugate Gradient solver using Intel MKL
// Solves Ax = b for symmetric positive-definite matrix A
//
// Parameters:
//   A: n x n matrix in row-major order (size n*n)
//   b: right-hand side vector (size n)
//   x: initial guess on input, solution on output (size n)
//   tolerance: convergence tolerance for residual norm
//   max_iterations: maximum number of iterations
//
// Returns: number of iterations performed
int cg(std::vector<double>& A, std::vector<double>& b, std::vector<double>& x,
       double tolerance = 1e-6, int max_iterations = 100);

#endif // CG_H
