#pragma once

#include <vector>

#include <mat_utils/mat_reader.h>

#include "common/mkl_matrices.h"

namespace cils {

mkl::CSRMatrix read_mkl(const mat_utils::MatReader<mat_utils::Sparsity::Sparse> &reader);

int run_mkl_cg(const mat_utils::MatReader<mat_utils::Sparsity::Sparse> &A,
               const std::vector<double> &b, std::vector<double> &x,
               const mat_utils::MatReader<mat_utils::Sparsity::Sparse> &L,
               double tolerance, int max_iterations);

struct MklDrBcgConfig {
    double tolerance;
    int max_iterations;
    int block_size;
};

int run_mkl_dr_bcg(const mat_utils::MatReader<mat_utils::Sparsity::Sparse> &A,
                   const std::vector<double> &b, std::vector<double> &x,
                   const mat_utils::MatReader<mat_utils::Sparsity::Sparse> &L,
                   MklDrBcgConfig config);

} // namespace cils
