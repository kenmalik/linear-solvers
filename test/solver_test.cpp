#include <gtest/gtest.h>

#include <mat_utils/mat_reader.h>
#include <mat_utils/mat_writer.h>

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "."
#endif

#include "parser.h"
#include "rhs.h"

#ifdef MKL_ENABLED
#include "mkl_adapter.h"
#endif

#ifdef CUDA_ENABLED
#include "cuda_adapter.h"
#endif

static constexpr double tolerance = 1e-6;
static constexpr int block_size = 4;

namespace {

std::vector<char *> argv_from(std::vector<std::string> &args) {
    std::vector<char *> argv;
    argv.reserve(args.size());
    for (auto &arg : args) {
        argv.push_back(arg.data());
    }
    return argv;
}

} // namespace

TEST(Rhs, LoadsCgRhsFromMat) {
    std::optional<mat_utils::DnMatReader> reader;
    reader.emplace(TEST_DATA_DIR "/b_vec_test.mat", std::vector<std::string>{}, "b");

    auto b = prepare_rhs(reader, 3, 1);

    EXPECT_EQ(b, (std::vector<double>{1.0, 2.0, 3.0}));
}

TEST(Rhs, LoadsDrBcgRhsFromMat) {
    std::optional<mat_utils::DnMatReader> reader;
    reader.emplace(TEST_DATA_DIR "/b_mat_test.mat", std::vector<std::string>{}, "b");

    auto b = prepare_rhs(reader, 3, 2);

    EXPECT_EQ(b, (std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}));
}

TEST(Rhs, RejectsDimensionMismatch) {
    std::optional<mat_utils::DnMatReader> reader;
    reader.emplace(TEST_DATA_DIR "/b_vec_test.mat", std::vector<std::string>{}, "b");

    EXPECT_THROW(static_cast<void>(prepare_rhs(reader, 3, 2)), std::runtime_error);
}

TEST(Rhs, GeneratesRandomRhsWhenFileNotProvided) {
    auto b = prepare_rhs(std::optional<mat_utils::DnMatReader>{}, 3, 2);

    EXPECT_EQ(b.size(), 6);
}

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
