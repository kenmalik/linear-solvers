#include "cg/cg.h"
#include <cmath>
#include <gtest/gtest.h>

// 3x3 symmetric positive-definite system used in all tests:
//
//   A = [4  1  0]     b = [1]
//       [1  3  1]         [2]
//       [0  1  2]         [3]

static constexpr double tolerance = 1e-10;

static double residual_norm(const std::vector<double> &A_dense,
                            const std::vector<double> &b,
                            const std::vector<double> &x) {
    int n = static_cast<int>(b.size());
    double norm = 0.0;
    for (int i = 0; i < n; ++i) {
        double r = b[i];
        for (int j = 0; j < n; ++j)
            r -= A_dense[i * n + j] * x[j];
        norm += r * r;
    }
    return std::sqrt(norm);
}

// Dense overload tests

TEST(CgDense, ConvergesWithinMaxIterations) {
    std::vector<double> A = {
        4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0,
    };
    std::vector<double> b = {1.0, 2.0, 3.0};
    std::vector<double> x = {0.0, 0.0, 0.0};

    int iters = cg(A, b, x, tolerance, 100);

    EXPECT_LT(iters, 100) << "solver did not converge";
}

TEST(CgDense, SolutionSatisfiesResidualTolerance) {
    std::vector<double> A = {
        4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0,
    };
    std::vector<double> b = {1.0, 2.0, 3.0};
    std::vector<double> x = {0.0, 0.0, 0.0};

    cg(A, b, x, tolerance, 100);

    EXPECT_LT(residual_norm(A, b, x), tolerance);
}

// Sparse (CSR) overload tests

static CsrMatrix make_sparse_A() {
    // CSR representation of the same 3x3 matrix:
    //   row 0: (0,4) (1,1)
    //   row 1: (0,1) (1,3) (2,1)
    //   row 2: (1,1) (2,2)
    CsrMatrix A;
    A.n = 3;
    A.values = {4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0};
    A.col_idx = {0, 1, 0, 1, 2, 1, 2};
    A.row_ptr = {0, 2, 5, 7};
    return A;
}

TEST(CgSparse, ConvergesWithinMaxIterations) {
    CsrMatrix A = make_sparse_A();
    std::vector<double> b = {1.0, 2.0, 3.0};
    std::vector<double> x = {0.0, 0.0, 0.0};

    int iters = cg(A, b, x, tolerance, 100);

    EXPECT_LT(iters, 100) << "solver did not converge";
}

TEST(CgSparse, SolutionSatisfiesResidualTolerance) {
    CsrMatrix A = make_sparse_A();
    std::vector<double> b = {1.0, 2.0, 3.0};
    std::vector<double> x = {0.0, 0.0, 0.0};

    cg(A, b, x, tolerance, 100);

    std::vector<double> A_dense = {
        4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0,
    };
    EXPECT_LT(residual_norm(A_dense, b, x), tolerance);
}

TEST(CgSparse, MatchesDenseSolution) {
    std::vector<double> A_dense = {
        4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0,
    };
    std::vector<double> b = {1.0, 2.0, 3.0};

    std::vector<double> x_dense = {0.0, 0.0, 0.0};
    cg(A_dense, b, x_dense, tolerance, 100);

    CsrMatrix A_sparse = make_sparse_A();
    std::vector<double> x_sparse = {0.0, 0.0, 0.0};
    cg(A_sparse, b, x_sparse, tolerance, 100);

    for (int i = 0; i < 3; ++i)
        EXPECT_NEAR(x_dense[i], x_sparse[i], 1e-8) << "mismatch at index " << i;
}

#ifdef USE_MAT_UTILS

#include <filesystem>
#include <mat_utils/mat_reader.h>

std::filesystem::path data_dir{DATA_DIR};

TEST(CgSparse, SpMatReader) {
    mat_utils::SpMatReader A{data_dir / "494_bus.mat", {"Problem"}, "A"};

    constexpr int max_iterations = 2000;
    std::vector<double> b(A.rows(), 1);
    std::vector<double> x(A.rows(), 0);

    int iters = cg(A, b, x, tolerance, max_iterations);

    EXPECT_LT(iters, max_iterations) << "solver did not converge";
}

TEST(PcgSparse, SpMatReader) {
    mat_utils::SpMatReader A{data_dir / "494_bus.mat", {"Problem"}, "A"};
    mat_utils::SpMatReader L{data_dir / "494_bus_ichol.mat", {}, "L"};

    int max_iterations = A.rows();
    std::vector<double> b(A.rows(), 1);
    std::vector<double> x(A.rows(), 0);

    int iters = cg(A, b, x, L, tolerance, max_iterations);

    EXPECT_LT(iters, max_iterations) << "solver did not converge";
}

#endif // USE_MAT_UTILS
