#include <cstring>
#include <filesystem>
#include <format>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <nvtx3/nvtx3.hpp>

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
    cg_run::DeviceVector b;
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
        ("x", "Path to x file.", cxxopts::value<std::string>())
        ("b", "Path to b file.", cxxopts::value<std::string>())
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
        return std::nullopt;
    }
    mat_utils::SpMatReader A_reader{A_path, {"Problem"}, "A"};
    args.A = cg_run::DeviceSparseMatrix<double>{A_reader};

    if (result.count("R")) {
        auto R_path = result["R"].as<std::string>();
        if (!fs::exists(R_path) && !fs::is_regular_file(R_path)) {
            print_error("R is an invalid file.");
            return std::nullopt;
        }

        mat_utils::SpMatReader R_reader{R_path, {}, "L"};
        args.R = cg_run::DeviceSparseMatrix<double>{R_reader};
    } else {
        std::int64_t rows = A_reader.rows();
        std::int64_t cols = rows;

        std::vector<double> vals(rows, 1.0);
        std::vector<std::int64_t> rowPtr(rows + 1);
        std::vector<std::int64_t> colInd(vals.size());

        std::int64_t i;
        for (i = 0; i < vals.size(); ++i) {
            rowPtr.at(i) = i;
            colInd.at(i) = i;
        }
        rowPtr.at(i) = vals.size();

        args.R = cg_run::DeviceSparseMatrix<double>{rows, cols, rowPtr, colInd,
                                                    vals};
    }

    if (result.count("x")) {
        auto x_path = result["x"].as<std::string>();
        if (!fs::exists(x_path) && !fs::is_regular_file(x_path)) {
            print_error("x is an invalid file.");
            return std::nullopt;
        }
        mat_utils::DnMatReader x_reader{x_path, {}, "X"};
        args.x = std::vector<double>(x_reader.data(),
                                     x_reader.data() + x_reader.rows());
    } else {
        args.x = std::vector<double>(A_reader.rows(), 0);
    }

    if (result.count("b")) {
        auto b_path = result["b"].as<std::string>();
        if (!fs::exists(b_path) && !fs::is_regular_file(b_path)) {
            print_error("b is an invalid file.");
            return std::nullopt;
        }
        mat_utils::DnMatReader b_reader{b_path, {}, "B"};
        args.b = std::vector<double>(b_reader.data(),
                                     b_reader.data() + b_reader.rows());
    } else {
        args.b = std::vector<double>(A_reader.rows(), 1);
    }

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
    NVTX3_FUNC_RANGE();
    nvtx3::event_attributes attr{"pre-cg"};
    auto pre_cg = nvtx3::start_range(attr);

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

        nvtx3::end_range(pre_cg);
        int iterations =
            cg_run::cg(cusparse, cublas, args.A.get(), args.b.get(),
                       args.x.get(), args.R.get(), args.tolerance,
                       args.max_iterations, args.real_residual);
        CUDA_CHECK(cudaDeviceSynchronize());

        nvtx3::event_attributes attr{"post-cg"};
        auto post_cg = nvtx3::start_range(attr);

        CUBLAS_CHECK(cublasDestroy_v2(cublas));
        CUSPARSE_CHECK(cusparseDestroy(cusparse));

        std::cout << iterations << std::endl;

        std::int64_t size = 0;
        void *values = nullptr;
        cudaDataType_t data_type;
        CUSPARSE_CHECK(
            cusparseDnVecGet(args.x.get(), &size, &values, &data_type));

        std::vector<double> h_x(size);
        CUDA_CHECK(cudaMemcpy(h_x.data(), values, sizeof(double) * size,
                              cudaMemcpyDeviceToHost));

        std::cout << "\nSolution:" << std::endl;
        for (int i = 0; i < std::min(static_cast<int>(h_x.size()), 5); ++i) {
            std::cout << "x[" << i + 1 << "]=" << h_x.at(i) << std::endl;
        }
        std::cout << "..." << std::endl;
        for (int i = std::max(static_cast<int>(h_x.size()) - 5, 5);
             i < static_cast<int>(h_x.size()); ++i) {
            std::cout << "x[" << i + 1 << "]=" << h_x.at(i) << std::endl;
        }

        nvtx3::end_range(post_cg);
    } else {
        return -1;
    }

    return 0;
}
