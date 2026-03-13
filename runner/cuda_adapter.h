#pragma once

#include <vector>

#include <mat_utils/mat_reader.h>

int run_cuda_cg(const mat_utils::SpMatReader &A, const std::vector<double> &b,
                std::vector<double> &x, const mat_utils::SpMatReader &L,
                double tolerance, int max_iterations);

int run_cuda_dr_bcg(const mat_utils::SpMatReader &A,
                    const std::vector<double> &b, std::vector<double> &x,
                    const mat_utils::SpMatReader &L, double tolerance,
                    int max_iterations, int block_size);
