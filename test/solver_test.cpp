#include <gtest/gtest.h>

#include <mat_utils/mat_reader.h>

#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "."
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

TEST(CgMkl, ConvergesOn1138Bus) {
    mat_utils::SpMatReader A{TEST_DATA_DIR "/1138_bus.mat", {"Problem"}, "A"};
    mat_utils::SpMatReader L{TEST_DATA_DIR "/1138_bus_ichol.mat", {}, "L"};

    int n = A.rows();
    std::vector<double> b(n, 1.0);
    std::vector<double> x(n, 0.0);

    int iters = run_mkl_cg(A, b, x, L, tolerance, n);

    EXPECT_LT(iters, n) << "CG (MKL) did not converge within " << n << " iterations";
}

TEST(DrBcgMkl, ConvergesOn1138Bus) {
    mat_utils::SpMatReader A{TEST_DATA_DIR "/1138_bus.mat", {"Problem"}, "A"};
    mat_utils::SpMatReader L{TEST_DATA_DIR "/1138_bus_ichol.mat", {}, "L"};

    int n = A.rows();
    std::vector<double> b(n * block_size, 1.0);
    std::vector<double> x(n * block_size, 0.0);

    int iters = run_mkl_dr_bcg(A, b, x, L, tolerance, n, block_size);

    EXPECT_LT(iters, n) << "DR-BCG (MKL) did not converge within " << n << " iterations";
}

#endif // MKL_ENABLED

#ifdef CUDA_ENABLED

TEST(CgCuda, ConvergesOn1138Bus) {
    mat_utils::SpMatReader A{TEST_DATA_DIR "/1138_bus.mat", {"Problem"}, "A"};
    mat_utils::SpMatReader L{TEST_DATA_DIR "/1138_bus_ichol.mat", {}, "L"};

    int n = A.rows();
    std::vector<double> b(n, 1.0);
    std::vector<double> x(n, 0.0);

    int iters = run_cuda_cg(A, b, x, L, tolerance, n);

    EXPECT_LT(iters, n) << "CG (CUDA) did not converge within " << n << " iterations";
}

TEST(DrBcgCuda, ConvergesOn1138Bus) {
    mat_utils::SpMatReader A{TEST_DATA_DIR "/1138_bus.mat", {"Problem"}, "A"};
    mat_utils::SpMatReader L{TEST_DATA_DIR "/1138_bus_ichol.mat", {}, "L"};

    int n = A.rows();
    std::vector<double> b(n * block_size, 1.0);
    std::vector<double> x(n * block_size, 0.0);

    int iters = run_cuda_dr_bcg(A, b, x, L, tolerance, n, block_size);

    EXPECT_LT(iters, n) << "DR-BCG (CUDA) did not converge within " << n << " iterations";
}

#endif // CUDA_ENABLED
