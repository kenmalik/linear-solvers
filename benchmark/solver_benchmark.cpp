#include <benchmark/benchmark.h>

#include <mat_utils/mat_reader.h>

#include <vector>

#ifndef BENCHMARK_DATA_DIR
#define BENCHMARK_DATA_DIR "."
#endif

#ifdef MKL_ENABLED
#include "mkl_adapter.h"
#endif

#ifdef CUDA_ENABLED
#include "cuda_adapter.h"
#endif

static constexpr double tolerance = 1e-6;
static constexpr int block_size = 4;

#ifdef MKL_ENABLED

static void BM_CgMkl(benchmark::State &state) {
    mat_utils::SpMatReader A{BENCHMARK_DATA_DIR "/G2_circuit.mat", {"Problem"}, "A"};
    mat_utils::SpMatReader L{BENCHMARK_DATA_DIR "/G2_circuit_ichol.mat", {}, "L"};
    int n = A.rows();
    std::vector<double> b(n, 1.0);

    for (auto _ : state) {
        std::vector<double> x(n, 0.0);
        int iters = run_mkl_cg(A, b, x, L, tolerance, n);
        state.counters["iters"] = iters;
    }
}
BENCHMARK(BM_CgMkl)->MinWarmUpTime(0.5);

static void BM_DrBcgMkl(benchmark::State &state) {
    mat_utils::SpMatReader A{BENCHMARK_DATA_DIR "/G2_circuit.mat", {"Problem"}, "A"};
    mat_utils::SpMatReader L{BENCHMARK_DATA_DIR "/G2_circuit_ichol.mat", {}, "L"};
    int n = A.rows();
    std::vector<double> b(n * block_size, 1.0);

    for (auto _ : state) {
        std::vector<double> x(n * block_size, 0.0);
        int iters = run_mkl_dr_bcg(A, b, x, L, tolerance, n, block_size);
        state.counters["iters"] = iters;
    }
}
BENCHMARK(BM_DrBcgMkl)->MinWarmUpTime(0.5);

#endif // MKL_ENABLED

#ifdef CUDA_ENABLED

static void BM_CgCuda(benchmark::State &state) {
    mat_utils::SpMatReader A{BENCHMARK_DATA_DIR "/G2_circuit.mat", {"Problem"}, "A"};
    mat_utils::SpMatReader L{BENCHMARK_DATA_DIR "/G2_circuit_ichol.mat", {}, "L"};
    int n = A.rows();
    std::vector<double> b(n, 1.0);

    for (auto _ : state) {
        std::vector<double> x(n, 0.0);
        int iters = run_cuda_cg(A, b, x, L, tolerance, n);
        state.counters["iters"] = iters;
    }
}
BENCHMARK(BM_CgCuda)->MinWarmUpTime(0.5);

static void BM_DrBcgCuda(benchmark::State &state) {
    mat_utils::SpMatReader A{BENCHMARK_DATA_DIR "/G2_circuit.mat", {"Problem"}, "A"};
    mat_utils::SpMatReader L{BENCHMARK_DATA_DIR "/G2_circuit_ichol.mat", {}, "L"};
    int n = A.rows();
    std::vector<double> b(n * block_size, 1.0);

    for (auto _ : state) {
        std::vector<double> x(n * block_size, 0.0);
        int iters = run_cuda_dr_bcg(A, b, x, L, tolerance, n, block_size);
        state.counters["iters"] = iters;
    }
}
BENCHMARK(BM_DrBcgCuda)->MinWarmUpTime(0.5);

#endif // CUDA_ENABLED
