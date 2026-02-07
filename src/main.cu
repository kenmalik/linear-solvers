#include <cstring>
#include <filesystem>
#include <format>
#include <iostream>
#include <optional>
#include <string>

#include <cublas_v2.h>
#include <cusparse_v2.h>

#include <cxxopts.hpp>

#include <mat_utils/mat_reader.h>

#include "cg_run/cg.h"
#include "cg_run/device_sparse_matrix.h"
#include "cg_run/device_vector.h"

namespace fs = std::filesystem;

struct Args {
    cg_run::DeviceSparseMatrix<double> A;
    cg_run::DeviceSparseMatrix<double> R;
    cg_run::DeviceVector x;
    cg_run::DeviceVector f;
    double tolerance;
    int max_iterations;
    bool real_residual;
};

void print_error(std::string_view message) {
    std::cerr << "Error: " << message << std::endl;
}

std::optional<Args> validate(int argc, char **argv) {
    cxxopts::Options options(argv[0], "CUDA Conjugate Gradient solver");

    // clang-format off
    options.add_options()
        ("h,help", "Print this help menu.")
        ("A", "Path to matrix file.", cxxopts::value<std::string>())
        ("R", "Path to preconditioner file.", cxxopts::value<std::string>())
        ("B", "Path to B file.", cxxopts::value<std::string>())
        ("tolerance", "Convergence tolerance.", cxxopts::value<double>()->default_value("1e-6"))
        ("max-iterations", "Maximum iterations (default: n).", cxxopts::value<int>())
        ("real-residual", "Whether to fully recalculate residual every iteration. Uses formula r = b - A * x.");
    options.parse_positional({"A", "R"});
    // clang-format on

    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    } catch (const std::exception &e) {
        print_error(e.what());
        std::cerr << options.help() << std::endl;
        return std::nullopt;
    }

    Args args;

    if (!result.count("A")) {
        print_error("Missing required argument A.");
        return std::nullopt;
    }

    auto A_path = result["A"].as<std::string>();
    if (!fs::exists(A_path) && !fs::is_regular_file(A_path)) {
        print_error("A is an invalid file.");
    }
    mat_utils::SpMatReader A_reader{A_path, {"Problem"}, "A"};
    args.A = cg_run::DeviceSparseMatrix<double>{A_reader};

    if (result.count("R")) {
        auto R_path = result["R"].as<std::string>();
        if (!fs::exists(R_path) && !fs::is_regular_file(R_path)) {
            print_error("R is an invalid file.");
        }

        mat_utils::SpMatReader R_reader{R_path, {}, "L"};
        args.R = cg_run::DeviceSparseMatrix<double>{R_reader};
    } else {
        std::vector<std::int64_t> ir(A_reader.ir_size());
        std::vector<std::int64_t> jc(A_reader.jc_size());
        std::int64_t i;
        for (i = 0; i < static_cast<std::int64_t>(A_reader.ir_size()); ++i) {
            ir[i] = i;
            jc[i] = i;
        }
        ir.push_back(i);
        std::vector<double> vals(A_reader.rows(), 1.0);
        args.R = cg_run::DeviceSparseMatrix<double>{jc, ir, vals};
    }

    // TODO: Add f and x reading
    args.x = std::vector<double>(A_reader.rows(), 0);
    args.f = std::vector<double>(A_reader.rows(), 1);

    args.tolerance = result["tolerance"].as<double>();

    if (!result.count("max-iterations")) {
        args.max_iterations = A_reader.rows();
    } else {
        args.max_iterations = result["max-iterations"].as<int>();
    }

    args.real_residual = result.count("real-residual");

    return args;
}

int main(int argc, char **argv) {
    if (auto validated = validate(argc, argv)) {
        auto &args = *validated;

        cusparseFillMode_t R_fill_mode = CUSPARSE_FILL_MODE_LOWER;
        CUSPARSE_CHECK(
            cusparseSpMatSetAttribute(args.R.get(), CUSPARSE_SPMAT_FILL_MODE,
                                      &R_fill_mode, sizeof(R_fill_mode)));

        cusparseHandle_t cusparse;
        cublasHandle_t cublas;

        CUSPARSE_CHECK(cusparseCreate(&cusparse));
        CUBLAS_CHECK(cublasCreate_v2(&cublas));

        int iterations =
            cg_run::cg(cusparse, cublas, args.A.get(), args.f.get(),
                       args.x.get(), args.R.get(), args.tolerance,
                       args.max_iterations, args.real_residual);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUBLAS_CHECK(cublasDestroy_v2(cublas));
        CUSPARSE_CHECK(cusparseDestroy(cusparse));

        std::cout << iterations << std::endl;

    } else {
        return -1;
    }

    return 0;
}
