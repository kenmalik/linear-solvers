#pragma once

#include <mkl_spblas.h>
#include <vector>

#include <mkl.h>

namespace cils::mkl {

// CSR sparse matrix descriptor
struct CSRMatrix {
    MKL_INT rows{};
    MKL_INT cols{};
    sparse_matrix_t mat{};
    struct matrix_descr descr{.type = SPARSE_MATRIX_TYPE_GENERAL,
                              .mode = SPARSE_FILL_MODE_FULL,
                              .diag = SPARSE_DIAG_NON_UNIT};
    std::vector<double> values;
    std::vector<MKL_INT> row_ptr;
    std::vector<MKL_INT> col_idx;
};

// Dense matrix stored in column-major order (Fortran layout for LAPACK/BLAS)
// Element (i, j) is at data[j * rows + i]
struct DenseMatrix {
    MKL_INT rows{};
    MKL_INT cols{};
    std::vector<double> data;
};

} // namespace cils::mkl
