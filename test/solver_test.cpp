#include <gtest/gtest.h>

#include <mat_utils/mat_reader.h>
#include <mat_utils/mat_writer.h>

#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "."
#endif

#include "common/timer.h"
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

std::vector<std::string> read_csv_ranges(const std::filesystem::path &path) {
    std::ifstream in(path);
    std::string line;
    std::vector<std::string> ranges;

    std::getline(in, line); // header
    while (std::getline(in, line)) {
        auto comma = line.find(',');
        ranges.push_back(line.substr(0, comma));
    }

    return ranges;
}

} // namespace

TEST(Timer, ReportsSectionsInFirstOccurrenceOrder) {
    CpuTimer<true> timer;

    timer.start("iteration");
    timer.stop("iteration");
    timer.start("setup");
    timer.stop("setup");
    timer.start("iteration");
    timer.stop("iteration");
    timer.start("solve");
    timer.stop("solve");

    auto path = std::filesystem::temp_directory_path() /
                ("timer-order-" + std::to_string(getpid()) + ".csv");
    timer.report(path.string());

    EXPECT_EQ(read_csv_ranges(path),
              (std::vector<std::string>{"iteration", "setup", "solve"}));

    std::filesystem::remove(path);
}

TEST(Timer, SuppressesDuplicateRowsBeforeReportAggregation) {
    CpuTimer<true> timer;

    timer.start("iteration");
    timer.stop("iteration");
    timer.start("inner");
    timer.stop("inner");
    timer.start("iteration");
    timer.stop("iteration");
    timer.start("inner");
    timer.stop("inner");
    timer.start("finalize");
    timer.stop("finalize");

    auto path = std::filesystem::temp_directory_path() /
                ("timer-dedup-" + std::to_string(getpid()) + ".csv");
    timer.report(path.string());

    EXPECT_EQ(read_csv_ranges(path),
              (std::vector<std::string>{"iteration", "inner", "finalize"}));

    std::filesystem::remove(path);
}

TEST(Timer, ReportsRaiiSectionsInFirstOccurrenceOrder) {
    CpuTimer<true> timer;

    {
        CpuTimer<true>::ScopedRange iteration(timer, "iteration");
    }
    {
        CpuTimer<true>::ScopedRange setup(timer, "setup");
    }
    {
        CpuTimer<true>::ScopedRange iteration(timer, "iteration");
    }
    {
        CpuTimer<true>::ScopedRange solve(timer, "solve");
    }

    auto path = std::filesystem::temp_directory_path() /
                ("timer-raii-order-" + std::to_string(getpid()) + ".csv");
    timer.report(path.string());

    EXPECT_EQ(read_csv_ranges(path),
              (std::vector<std::string>{"iteration", "setup", "solve"}));

    std::filesystem::remove(path);
}

TEST(Timer, ReportsNestedRaiiSectionsOnceInEntryOrder) {
    CpuTimer<true> timer;

    {
        CpuTimer<true>::ScopedRange outer(timer, "outer");
        {
            CpuTimer<true>::ScopedRange inner(timer, "inner");
        }
    }
    {
        CpuTimer<true>::ScopedRange outer(timer, "outer");
        {
            CpuTimer<true>::ScopedRange inner(timer, "inner");
        }
    }

    auto path = std::filesystem::temp_directory_path() /
                ("timer-raii-nested-" + std::to_string(getpid()) + ".csv");
    timer.report(path.string());

    EXPECT_EQ(read_csv_ranges(path),
              (std::vector<std::string>{"outer", "inner"}));

    std::filesystem::remove(path);
}

TEST(Rhs, LoadsCgRhsFromMat) {
    std::optional<mat_utils::DnMatReader> reader;
    reader.emplace(TEST_DATA_DIR "/b_vec_test.mat", std::vector<std::string>{}, "b");

    auto b = prepare_rhs(reader, std::optional<mat_utils::DnMatReader>{}, 3, 1);

    EXPECT_EQ(b, (std::vector<double>{1.0, 2.0, 3.0}));
}

TEST(Rhs, LoadsDrBcgRhsFromMat) {
    std::optional<mat_utils::DnMatReader> reader;
    reader.emplace(TEST_DATA_DIR "/b_mat_test.mat", std::vector<std::string>{}, "b");

    auto b = prepare_rhs(reader, std::optional<mat_utils::DnMatReader>{}, 3, 2);

    EXPECT_EQ(b, (std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}));
}

TEST(Rhs, CombinesVectorAndMatrixRhsForDrBcg) {
    std::optional<mat_utils::DnMatReader> b_reader;
    b_reader.emplace(TEST_DATA_DIR "/b_vec_test.mat", std::vector<std::string>{}, "b");
    std::optional<mat_utils::DnMatReader> B_reader;
    B_reader.emplace(TEST_DATA_DIR "/b_mat_test.mat", std::vector<std::string>{}, "b");

    auto b = prepare_rhs(b_reader, B_reader, 3, 3);

    EXPECT_EQ(b, (std::vector<double>{1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0}));
}

TEST(Rhs, CombinesVectorRhsWithSeededDefaultMatrixForDrBcg) {
    std::optional<mat_utils::DnMatReader> b_reader;
    b_reader.emplace(TEST_DATA_DIR "/b_vec_test.mat", std::vector<std::string>{}, "b");

    auto b = prepare_rhs(b_reader, std::optional<mat_utils::DnMatReader>{}, 3, 3);
    auto fallback = prepare_rhs(std::optional<mat_utils::DnMatReader>{},
                                std::optional<mat_utils::DnMatReader>{}, 3, 2);

    EXPECT_EQ(b, (std::vector<double>{1.0, 2.0, 3.0,
                                      fallback[0], fallback[1], fallback[2],
                                      fallback[3], fallback[4], fallback[5]}));
}

TEST(Rhs, CombinesSeededDefaultVectorWithMatrixRhsForDrBcg) {
    std::optional<mat_utils::DnMatReader> B_reader;
    B_reader.emplace(TEST_DATA_DIR "/b_mat_test.mat", std::vector<std::string>{}, "b");

    auto b = prepare_rhs(std::optional<mat_utils::DnMatReader>{}, B_reader, 3, 3);
    auto fallback = prepare_rhs(std::optional<mat_utils::DnMatReader>{},
                                std::optional<mat_utils::DnMatReader>{}, 3, 1);

    EXPECT_EQ(b, (std::vector<double>{fallback[0], fallback[1], fallback[2],
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0}));
}

TEST(Rhs, RejectsDimensionMismatch) {
    std::optional<mat_utils::DnMatReader> reader;
    reader.emplace(TEST_DATA_DIR "/b_mat_test.mat", std::vector<std::string>{}, "b");

    EXPECT_THROW(
        static_cast<void>(prepare_rhs(reader, std::optional<mat_utils::DnMatReader>{}, 3, 3)),
        std::runtime_error);
}

TEST(Rhs, RejectsSplitRhsDimensionMismatch) {
    std::optional<mat_utils::DnMatReader> b_reader;
    b_reader.emplace(TEST_DATA_DIR "/b_vec_test.mat", std::vector<std::string>{}, "b");
    std::optional<mat_utils::DnMatReader> B_reader;
    B_reader.emplace(TEST_DATA_DIR "/b_vec_test.mat", std::vector<std::string>{}, "b");

    EXPECT_THROW(static_cast<void>(prepare_rhs(b_reader, B_reader, 3, 3)),
                 std::runtime_error);
}

TEST(Rhs, GeneratesRandomRhsWhenFileNotProvided) {
    auto b = prepare_rhs(std::optional<mat_utils::DnMatReader>{},
                         std::optional<mat_utils::DnMatReader>{}, 3, 2);

    EXPECT_EQ(b.size(), 6);
}

TEST(Rhs, GeneratesSeededRandomRhsWhenFileNotProvided) {
    auto first = prepare_rhs(std::optional<mat_utils::DnMatReader>{},
                             std::optional<mat_utils::DnMatReader>{}, 3, 2);
    auto second = prepare_rhs(std::optional<mat_utils::DnMatReader>{},
                              std::optional<mat_utils::DnMatReader>{}, 3, 2);

    EXPECT_EQ(first, second);
    EXPECT_NE(first, (std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
}

TEST(Rhs, LoadsCgInitialGuessFromMat) {
    std::optional<mat_utils::DnMatReader> reader;
    reader.emplace(TEST_DATA_DIR "/x_vec_test.mat", std::vector<std::string>{}, "x");

    auto initial_guess =
        prepare_initial_guess(reader, std::optional<mat_utils::DnMatReader>{}, 3, 1);

    EXPECT_EQ(initial_guess, (std::vector<double>{1.0, 2.0, 3.0}));
}

TEST(Rhs, LoadsDrBcgInitialGuessFromMat) {
    std::optional<mat_utils::DnMatReader> reader;
    reader.emplace(TEST_DATA_DIR "/x_mat_test.mat", std::vector<std::string>{}, "x");

    auto initial_guess =
        prepare_initial_guess(reader, std::optional<mat_utils::DnMatReader>{}, 3, 2);

    EXPECT_EQ(initial_guess, (std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}));
}

TEST(Rhs, CombinesVectorAndMatrixInitialGuessForDrBcg) {
    std::optional<mat_utils::DnMatReader> x_reader;
    x_reader.emplace(TEST_DATA_DIR "/x_vec_test.mat", std::vector<std::string>{}, "x");
    std::optional<mat_utils::DnMatReader> X_reader;
    X_reader.emplace(TEST_DATA_DIR "/x_mat_test.mat", std::vector<std::string>{}, "x");

    auto initial_guess = prepare_initial_guess(x_reader, X_reader, 3, 3);

    EXPECT_EQ(initial_guess,
              (std::vector<double>{1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0}));
}

TEST(Rhs, CombinesVectorInitialGuessWithDefaultMatrixForDrBcg) {
    std::optional<mat_utils::DnMatReader> x_reader;
    x_reader.emplace(TEST_DATA_DIR "/x_vec_test.mat", std::vector<std::string>{}, "x");

    auto initial_guess =
        prepare_initial_guess(x_reader, std::optional<mat_utils::DnMatReader>{}, 3, 3);

    EXPECT_EQ(initial_guess,
              (std::vector<double>{1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
}

TEST(Rhs, CombinesDefaultVectorWithMatrixInitialGuessForDrBcg) {
    std::optional<mat_utils::DnMatReader> X_reader;
    X_reader.emplace(TEST_DATA_DIR "/x_mat_test.mat", std::vector<std::string>{}, "x");

    auto initial_guess =
        prepare_initial_guess(std::optional<mat_utils::DnMatReader>{}, X_reader, 3, 3);

    EXPECT_EQ(initial_guess,
              (std::vector<double>{0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0}));
}

TEST(Rhs, RejectsInitialGuessDimensionMismatch) {
    std::optional<mat_utils::DnMatReader> reader;
    reader.emplace(TEST_DATA_DIR "/x_mat_test.mat", std::vector<std::string>{}, "x");

    EXPECT_THROW(static_cast<void>(prepare_initial_guess(
                     reader, std::optional<mat_utils::DnMatReader>{}, 3, 3)),
                 std::runtime_error);
}

TEST(Rhs, RejectsSplitInitialGuessDimensionMismatch) {
    std::optional<mat_utils::DnMatReader> x_reader;
    x_reader.emplace(TEST_DATA_DIR "/x_vec_test.mat", std::vector<std::string>{}, "x");
    std::optional<mat_utils::DnMatReader> X_reader;
    X_reader.emplace(TEST_DATA_DIR "/x_vec_test.mat", std::vector<std::string>{}, "x");

    EXPECT_THROW(static_cast<void>(prepare_initial_guess(x_reader, X_reader, 3, 3)),
                 std::runtime_error);
}

TEST(Rhs, GeneratesZeroInitialGuessWhenFileNotProvided) {
    auto initial_guess = prepare_initial_guess(
        std::optional<mat_utils::DnMatReader>{}, std::optional<mat_utils::DnMatReader>{}, 3, 2);

    EXPECT_EQ(initial_guess,
              (std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
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
