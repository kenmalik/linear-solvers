#include "config.h"

#include "mkl_adapter.h"

#ifdef SOLVERS_BUILD_CG
#include "cg/mkl.h"
#endif

#ifdef SOLVERS_BUILD_DR_BCG
#include "dr_bcg/mkl.h"
#endif

#include "common/mkl_matrices.h"

CSRMatrix read_mkl(const mat_utils::SpMatReader &reader) {
    const MKL_INT n_rows = static_cast<MKL_INT>(reader.rows());
    const MKL_INT n_cols = static_cast<MKL_INT>(reader.cols());
    const MKL_INT nnz = static_cast<MKL_INT>(reader.nnz());

    const size_t *jc = reader.jc();
    const size_t *ir = reader.ir();
    const double *values = reader.data();

    CSRMatrix csr;
    csr.rows = n_rows;
    csr.cols = n_cols;
    csr.row_ptr.assign(n_rows + 1, 0);
    csr.col_idx.resize(nnz);
    csr.values.resize(nnz);

    // Count nnz per row
    for (MKL_INT k = 0; k < nnz; ++k)
        ++csr.row_ptr[ir[k] + 1];

    // Exclusive prefix sum → row_ptr
    for (MKL_INT i = 0; i < n_rows; ++i)
        csr.row_ptr[i + 1] += csr.row_ptr[i];

    // Scatter CSC columns into CSR rows
    std::vector<MKL_INT> cursor(csr.row_ptr.begin(),
                                csr.row_ptr.begin() + n_rows);
    for (MKL_INT j = 0; j < n_cols; ++j) {
        for (size_t k = jc[j]; k < jc[j + 1]; ++k) {
            MKL_INT row = static_cast<MKL_INT>(ir[k]);
            MKL_INT pos = cursor[row]++;
            csr.col_idx[pos] = j;
            csr.values[pos] = values[k];
        }
    }

    mkl_sparse_d_create_csr(&csr.mat, SPARSE_INDEX_BASE_ZERO, reader.rows(),
                            reader.cols(), csr.row_ptr.data(),
                            csr.row_ptr.data() + 1, csr.col_idx.data(),
                            csr.values.data());

    return csr;
}

#ifdef SOLVERS_BUILD_CG

int run_mkl_cg(const mat_utils::SpMatReader &A, const std::vector<double> &b,
               std::vector<double> &x, const mat_utils::SpMatReader &L,
               double tolerance, int max_iterations) {
    auto A_csr = read_mkl(A);
    A_csr.descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    auto L_csr = read_mkl(L);
    L_csr.descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    L_csr.descr.mode = SPARSE_FILL_MODE_LOWER;
    L_csr.descr.diag = SPARSE_DIAG_NON_UNIT;

    return cg::mkl::solve(A_csr, b, x, L_csr, tolerance, max_iterations);
}

#endif // SOLVERS_BUILD_CG

#ifdef SOLVERS_BUILD_DR_BCG

int run_mkl_dr_bcg(const mat_utils::SpMatReader &A,
                   const std::vector<double> &b, std::vector<double> &x,
                   const mat_utils::SpMatReader &L, double tolerance,
                   int max_iterations, int block_size) {
    auto A_csr = read_mkl(A);
    A_csr.descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    auto L_csr = read_mkl(L);
    L_csr.descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    L_csr.descr.mode = SPARSE_FILL_MODE_LOWER;
    L_csr.descr.diag = SPARSE_DIAG_NON_UNIT;

    int n = A.rows();
    DenseMatrix b_dm{n, block_size, b};
    DenseMatrix x_dm{n, block_size, x};

    return dr_bcg::mkl::solve(A_csr, L_csr, b_dm, x_dm, tolerance,
                              max_iterations);
}

#endif // SOLVERS_BUILD_DR_BCG