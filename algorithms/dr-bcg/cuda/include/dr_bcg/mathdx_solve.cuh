#pragma once

// Entry points for the fused MathDx CholeskyQR2 DR-BCG variant (PLAN.md Stage 2).
//
// These wrap dr_bcg::cuda::solve<double, MathDxCholeskyQr2<double>> so the
// MathDx (cuBLASDx + cuSolverDx) kernels are instantiated and device-linked
// inside the dedicated cuda_dr_bcg_mathdx target, not in general adapter code.
// Only declarations live here; the definitions are in mathdx_solve.cu.

#include <cuda_runtime.h>
#include <cusparse.h>

namespace dr_bcg::cuda {

struct Handles; // defined in dr_bcg/cuda.cuh

// Preconditioned (M = L L^T) DR-BCG with fused CholeskyQR2 orthonormalization.
int solve_cholqr_dx(Handles &handles, cusparseSpMatDescr_t A,
                    cusparseDnMatDescr_t X, cusparseDnMatDescr_t B,
                    cusparseSpMatDescr_t L, double tolerance,
                    int max_iterations, cudaStream_t stream);

// Unpreconditioned DR-BCG with fused CholeskyQR2 orthonormalization.
int solve_cholqr_dx(Handles &handles, cusparseSpMatDescr_t A,
                    cusparseDnMatDescr_t X, cusparseDnMatDescr_t B,
                    double tolerance, int max_iterations, cudaStream_t stream);

// Preconditioned (M = L L^T) fully fused DR-BCG: fused CholeskyQR2 (Stage 2)
// plus the fused reduced-system xi chain (Stage 3).
int solve_fused_dx(Handles &handles, cusparseSpMatDescr_t A,
                   cusparseDnMatDescr_t X, cusparseDnMatDescr_t B,
                   cusparseSpMatDescr_t L, double tolerance, int max_iterations,
                   cudaStream_t stream);

// Unpreconditioned fully fused DR-BCG (fused CholeskyQR2 + fused xi chain).
int solve_fused_dx(Handles &handles, cusparseSpMatDescr_t A,
                   cusparseDnMatDescr_t X, cusparseDnMatDescr_t B,
                   double tolerance, int max_iterations, cudaStream_t stream);

} // namespace dr_bcg::cuda
