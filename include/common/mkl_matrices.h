#pragma once

#include <vector>

#include <mkl.h>

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
