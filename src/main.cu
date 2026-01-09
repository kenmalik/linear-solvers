#include <cstring>
#include <filesystem>
#include <format>
#include <iostream>
#include <mat_utils/mat_reader.h>
#include <optional>
#include <string>

#include <cublas_v2.h>
#include <cusparse_v2.h>

#include <cxxopts.hpp>

#include "cg_run/cg.h"
#include "cg_run/device_sparse_matrix.h"

namespace fs = std::filesystem;

struct Args {
    fs::path A;
    fs::path R;
    double tolerance;
    int max_iterations;
};

std::optional<Args> validate(int argc, char **argv) {
    cxxopts::Options options(argv[0], "CUDA Conjugate Gradient solver");

    // clang-format off
    options.add_options()
        ("A", "Path to matrix file", cxxopts::value<std::string>())
        ("R", "Path to preconditioner file", cxxopts::value<std::string>())
        ("tolerance", "Convergence tolerance (default: 1e-6)", cxxopts::value<double>()->default_value("1e-6"))
        ("max-iterations", "Maximum iterations (default: 1)", cxxopts::value<int>()->default_value("1"))
        ("h,help", "Print help");
    options.parse_positional({"A", "R"});
    // clang-format on

    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << options.help() << std::endl;
        return std::nullopt;
    }

    if (result.count("help")) {
        std::cerr << options.help() << std::endl;
        return std::nullopt;
    }

    if (!result.count("A") || !result.count("R")) {
        std::cerr << "Error: Both matrix and preconditioner files are required."
                  << std::endl;
        std::cerr << options.help() << std::endl;
        return std::nullopt;
    }

    Args args;

    args.A = result["A"].as<std::string>();
    if (!fs::exists(args.A)) {
        std::cerr << "Error: File " << args.A << " does not exist."
                  << std::endl;
        return std::nullopt;
    }

    args.R = result["R"].as<std::string>();
    if (!fs::exists(args.R)) {
        std::cerr << "Error: File " << args.R << " does not exist."
                  << std::endl;
        return std::nullopt;
    }

    args.tolerance = result["tolerance"].as<double>();
    args.max_iterations = result["max-iterations"].as<int>();

    return args;
}

int main(int argc, char **argv) {
    if (const auto validated = validate(argc, argv)) {
        const auto &args = *validated;
        std::cerr << std::format("{} {} {} {}", args.A.string(),
                                 args.R.string(), args.tolerance,
                                 args.max_iterations)
                  << std::endl;

        mat_utils::SpMatReader A_reader{args.A.string(), {"Problem"}, "A"};
        mat_utils::SpMatReader R_reader{args.R.string(), {}, "L"};

        std::cerr << std::format("A: {} ({}x{})", args.A.stem().string(),
                                 A_reader.rows(), A_reader.cols())
                  << std::endl;
        std::cerr << std::format("R: {} ({}x{})", args.R.stem().string(),
                                 R_reader.rows(), R_reader.cols())
                  << std::endl;

        cg_run::DeviceSparseMatrix<double> A{A_reader};
        cg_run::DeviceSparseMatrix<double> R{R_reader};

        cusparseFillMode_t R_fill_mode = CUSPARSE_FILL_MODE_LOWER;
        cusparseSpMatSetAttribute(R.get(), CUSPARSE_SPMAT_FILL_MODE,
                                  &R_fill_mode, sizeof(R_fill_mode));

        double *x_d = nullptr;
        cusparseDnVecDescr_t x;
        cudaMalloc(&x_d, sizeof(double) * A_reader.rows());
        std::vector<double> x_initial(A_reader.rows(), 0);
        cudaMemcpy(x_d, x_initial.data(), sizeof(double) * A_reader.rows(),
                   cudaMemcpyHostToDevice);

        double *f_d = nullptr;
        cusparseDnVecDescr_t f;
        cudaMalloc(&f_d, sizeof(double) * A_reader.rows());
        std::vector<double> f_initial(A_reader.rows(), 1);
        cudaMemcpy(f_d, f_initial.data(), sizeof(double) * A_reader.rows(),
                   cudaMemcpyHostToDevice);

        cusparseCreateDnVec(&x, A_reader.rows(), x_d, CUDA_R_64F);
        cusparseCreateDnVec(&f, A_reader.rows(), f_d, CUDA_R_64F);

        cusparseHandle_t cusparse;
        cublasHandle_t cublas;

        cusparseCreate(&cusparse);
        cublasCreate_v2(&cublas);

        int iterations = cg_run::cg(cusparse, cublas, A.get(), x, f, R.get());

        cusparseDestroyDnVec(f);
        cudaFree(f_d);

        cusparseDestroyDnVec(x);
        cudaFree(x_d);

        cublasDestroy_v2(cublas);
        cusparseDestroy(cusparse);

        std::cout << iterations << std::endl;

    } else {
        return -1;
    }

    return 0;
}
