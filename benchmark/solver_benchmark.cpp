#include "config.h"

#include <benchmark/benchmark.h>

#include <mat_utils/mat_reader.h>

#include <vector>

#ifndef BENCHMARK_DATA_DIR
#define BENCHMARK_DATA_DIR "."
#endif

#ifdef SOLVERS_BUILD_MKL
#include "mkl_adapter.h"
#endif

#ifdef SOLVERS_BUILD_CUDA
#include "cuda_adapter.h"
#endif

static constexpr double tolerance = 1e-6;

#ifdef SOLVERS_BUILD_MKL

#ifdef SOLVERS_BUILD_CG

static void BM_CgMkl(benchmark::State &state) {
    mat_utils::MatReader<mat_utils::Sparsity::Sparse> A{BENCHMARK_DATA_DIR "/G2_circuit.mat", {"Problem"}, "A"};
    mat_utils::MatReader<mat_utils::Sparsity::Sparse> L{BENCHMARK_DATA_DIR "/G2_circuit_ichol.mat", {}, "L"};
    int n = A.rows();
    std::vector<double> b(n, 1.0);

    for (auto _ : state) {
        std::vector<double> x(n, 0.0);
        int iters = run_mkl_cg(A, b, x, L, tolerance, n);
        state.counters["iters"] = iters;
    }
}
BENCHMARK(BM_CgMkl)->MinWarmUpTime(0.5);

#endif // SOLVERS_BUILD_CG

#ifdef SOLVERS_BUILD_DR_BCG

static void BM_DrBcgMkl(benchmark::State &state) {
    int block_size = state.range(0);
    mat_utils::MatReader<mat_utils::Sparsity::Sparse> A{BENCHMARK_DATA_DIR "/G2_circuit.mat", {"Problem"}, "A"};
    mat_utils::MatReader<mat_utils::Sparsity::Sparse> L{BENCHMARK_DATA_DIR "/G2_circuit_ichol.mat", {}, "L"};
    int n = A.rows();
    std::vector<double> b(n * block_size, 1.0);

    for (auto _ : state) {
        std::vector<double> x(n * block_size, 0.0);
        int iters = run_mkl_dr_bcg(A, b, x, L, tolerance, n, block_size);
        state.counters["iters"] = iters;
    }
}
BENCHMARK(BM_DrBcgMkl)->ArgsProduct({{1, 2, 4, 8, 16, 32, 64}})->MinWarmUpTime(0.5);

#endif // SOLVERS_BUILD_DR_BCG

#endif // SOLVERS_BUILD_MKL

#ifdef SOLVERS_BUILD_CUDA

#ifdef SOLVERS_BUILD_CG

static void BM_CgCuda(benchmark::State &state) {
    mat_utils::MatReader<mat_utils::Sparsity::Sparse> A{BENCHMARK_DATA_DIR "/G2_circuit.mat", {"Problem"}, "A"};
    mat_utils::MatReader<mat_utils::Sparsity::Sparse> L{BENCHMARK_DATA_DIR "/G2_circuit_ichol.mat", {}, "L"};
    int n = A.rows();
    std::vector<double> b(n, 1.0);

    for (auto _ : state) {
        std::vector<double> x(n, 0.0);
        int iters = run_cuda_cg(A, b, x, L, tolerance, n);
        state.counters["iters"] = iters;
    }
}
BENCHMARK(BM_CgCuda)->MinWarmUpTime(0.5);

#endif // SOLVERS_BUILD_CG

#ifdef SOLVERS_BUILD_DR_BCG

static void BM_DrBcgCuda(benchmark::State &state) {
    int block_size = state.range(0);
    mat_utils::MatReader<mat_utils::Sparsity::Sparse> A{BENCHMARK_DATA_DIR "/G2_circuit.mat", {"Problem"}, "A"};
    mat_utils::MatReader<mat_utils::Sparsity::Sparse> L{BENCHMARK_DATA_DIR "/G2_circuit_ichol.mat", {}, "L"};
    int n = A.rows();
    std::vector<double> b(n * block_size, 1.0);

    for (auto _ : state) {
        std::vector<double> x(n * block_size, 0.0);
        int iters = run_cuda_dr_bcg(A, b, x, L, tolerance, n, block_size);
        state.counters["iters"] = iters;
    }
}
BENCHMARK(BM_DrBcgCuda)->ArgsProduct({{1, 2, 4, 8, 16, 32, 64}})->MinWarmUpTime(0.5);

#endif // SOLVERS_BUILD_DR_BCG

#endif // SOLVERS_BUILD_CUDA
