#pragma once

#include <vector>

#include <mat_utils/mat_reader.h>

int run_cuda(const mat_utils::SpMatReader &A, const std::vector<double> &b,
             std::vector<double> &x, const mat_utils::SpMatReader &L);
