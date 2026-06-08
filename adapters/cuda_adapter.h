#pragma once

#include "qr_backend.h"

#include <vector>

#include <mat_utils/mat_reader.h>

int run_cuda_cg(const mat_utils::SpMatReader &A, const std::vector<double> &b,
                std::vector<double> &x, const mat_utils::SpMatReader &L,
                double tolerance, int max_iterations,
                bool disable_tensor_cores = false);

int run_cuda_dr_bcg(const mat_utils::SpMatReader &A,
                    const std::vector<double> &b, std::vector<double> &x,
                    const mat_utils::SpMatReader &L, double tolerance,
                    int max_iterations, int block_size,
                    bool disable_tensor_cores = false,
                    QrBackend qr_backend = QrBackend::Householder);

int run_cuda_dr_bcg(const mat_utils::SpMatReader &A,
                    const std::vector<double> &b, std::vector<double> &x,
                    double tolerance, int max_iterations, int block_size,
                    bool disable_tensor_cores = false,
                    QrBackend qr_backend = QrBackend::Householder);
