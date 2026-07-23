#pragma once

#include "qr_backend.h"

#include "common/supported_type.h"

#include <mat_utils/mat_reader.h>
#include <mat_utils/supported_type.h>

#include <vector>

int run_cuda_cg(const mat_utils::MatReader<mat_utils::Sparsity::Sparse> &A,
                const std::vector<double> &b, std::vector<double> &x,
                const mat_utils::MatReader<mat_utils::Sparsity::Sparse> &L,
                double tolerance, int max_iterations,
                bool disable_tensor_cores = false);

template <cils::SupportedType T>
struct CudaDrBcgConfig {
    T tolerance;
    int max_iterations;
    int block_size;
    bool disable_tensor_cores;
    QrBackend qr_backend;
    bool fused_xi;
};

template <cils::SupportedType T>
int run_cuda_dr_bcg(const mat_utils::MatReader<mat_utils::Sparsity::Sparse> &A,
                    const std::vector<T> &b, std::vector<T> &x,
                    const mat_utils::MatReader<mat_utils::Sparsity::Sparse> &L,
                    CudaDrBcgConfig<T> config);

template <cils::SupportedType T>
int run_cuda_dr_bcg(const mat_utils::MatReader<mat_utils::Sparsity::Sparse> &A,
                    const std::vector<T> &b, std::vector<T> &x,
                    CudaDrBcgConfig<T> config);

extern template int run_cuda_dr_bcg<double>(
    const mat_utils::MatReader<mat_utils::Sparsity::Sparse> &A,
    const std::vector<double> &b, std::vector<double> &x,
    const mat_utils::MatReader<mat_utils::Sparsity::Sparse> &L,
    CudaDrBcgConfig<double> config);
extern template int run_cuda_dr_bcg<float>(
    const mat_utils::MatReader<mat_utils::Sparsity::Sparse> &A,
    const std::vector<float> &b, std::vector<float> &x,
    const mat_utils::MatReader<mat_utils::Sparsity::Sparse> &L,
    CudaDrBcgConfig<float> config);

extern template int run_cuda_dr_bcg<double>(
    const mat_utils::MatReader<mat_utils::Sparsity::Sparse> &A,
    const std::vector<double> &b, std::vector<double> &x,
    CudaDrBcgConfig<double> config);
extern template int run_cuda_dr_bcg<float>(
    const mat_utils::MatReader<mat_utils::Sparsity::Sparse> &A,
    const std::vector<float> &b, std::vector<float> &x,
    CudaDrBcgConfig<float> config);
