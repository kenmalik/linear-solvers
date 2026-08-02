#include "config.h"

#include "mkl_adapter.h"

#ifdef SOLVERS_BUILD_CG
#include "mkl/cg.h"
#endif

#ifdef SOLVERS_BUILD_DR_BCG
#include "mkl/dr_bcg.h"
#endif

#include "common/mkl_matrices.h"

#include <mkl_spblas.h>
#include <mkl_types.h>

#include <cstddef>
#include <span>
#include <utility>

namespace cils {

CSRMatrix read_mkl(const mat_utils::MatReader<mat_utils::Sparsity::Sparse> &reader) {
    const std::size_t nnz = reader.nonzero_count();
    std::span<double> values = reader.values<double>();

    CSRMatrix csr{
        .rows = static_cast<MKL_INT>(reader.rows()),
        .cols = static_cast<MKL_INT>(reader.cols()),
        .values = {values.begin(), values.end()},
        .row_ptr = std::vector<MKL_INT>(reader.rows() + 1, 0),
        .col_idx = std::vector<MKL_INT>(nnz)};

    // Count nnz per row
    std::span<std::size_t> ir = reader.row_indices();
    std::span<std::size_t> jc = reader.column_pointers();

    for (MKL_INT k = 0; std::cmp_less(k, nnz); ++k) {
        ++csr.row_ptr[ir[k] + 1];
    }

    // Exclusive prefix sum → row_ptr
    for (MKL_INT i = 0; i < csr.rows; ++i) {
        csr.row_ptr[i + 1] += csr.row_ptr[i];
    }

    // Scatter CSC columns into CSR rows
    std::vector<MKL_INT> cursor(csr.row_ptr.begin(),
                                csr.row_ptr.begin() + csr.cols);
    for (MKL_INT j = 0; j < csr.cols; ++j) {
        for (size_t k = jc[j]; k < jc[j + 1]; ++k) {
            MKL_INT row = static_cast<MKL_INT>(ir[k]);
            MKL_INT pos = cursor[row]++;
            csr.col_idx[pos] = j;
            csr.values[pos] = values[k];
        }
    }

    mkl_sparse_d_create_csr(&csr.mat, SPARSE_INDEX_BASE_ZERO,
                            static_cast<long long>(reader.rows()),
                            static_cast<long long>(reader.cols()),
                            csr.row_ptr.data(), csr.row_ptr.data() + 1,
                            csr.col_idx.data(), csr.values.data());

    return csr;
}

#ifdef SOLVERS_BUILD_CG

int run_mkl_cg(const mat_utils::MatReader<mat_utils::Sparsity::Sparse> &A,
               const std::vector<double> &b, std::vector<double> &x,
               const mat_utils::MatReader<mat_utils::Sparsity::Sparse> &L,
               double tolerance, int max_iterations) {
    auto A_csr = read_mkl(A);
    A_csr.descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    auto L_csr = read_mkl(L);
    L_csr.descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    L_csr.descr.mode = SPARSE_FILL_MODE_LOWER;
    L_csr.descr.diag = SPARSE_DIAG_NON_UNIT;

    return cils::mkl::cg(A_csr, b, x, L_csr,
                         {.tolerance = tolerance,
                          .max_iterations = max_iterations});
}

#endif // SOLVERS_BUILD_CG

#ifdef SOLVERS_BUILD_DR_BCG

int run_mkl_dr_bcg(const mat_utils::MatReader<mat_utils::Sparsity::Sparse> &A,
                   const std::vector<double> &b, std::vector<double> &x,
                   const mat_utils::MatReader<mat_utils::Sparsity::Sparse> &L,
                   MklDrBcgConfig config) {
    auto A_csr = read_mkl(A);
    A_csr.descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    auto L_csr = read_mkl(L);
    L_csr.descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    L_csr.descr.mode = SPARSE_FILL_MODE_LOWER;
    L_csr.descr.diag = SPARSE_DIAG_NON_UNIT;

    int n = static_cast<int>(A.rows());
    DenseMatrix b_dm{.rows = n, .cols = config.block_size, .data = b};
    DenseMatrix x_dm{.rows = n, .cols = config.block_size, .data = x};

    return cils::mkl::dr_bcg(A_csr, L_csr, b_dm, x_dm,
                             {.tolerance = config.tolerance,
                              .max_iterations = config.max_iterations});
}

#endif // SOLVERS_BUILD_DR_BCG

} // namespace cils
