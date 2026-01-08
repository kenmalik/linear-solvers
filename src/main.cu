#include <cstring>
#include <filesystem>
#include <format>
#include <iostream>
#include <mat_utils/mat_reader.h>
#include <optional>
#include <string>

#include <cublas_v2.h>
#include <cusparse_v2.h>

#include "cg_run/cg.h"
#include "cg_run/device_sparse_matrix.h"

namespace fs = std::filesystem;

struct Args {
    fs::path A;
    fs::path R;
    double tolerance;
    int max_iterations;
};

void print_usage(const char *progname) {
    std::cerr << "Usage: " << progname
              << " <matrix> <preconditioner> [tolerance] [max_iterations]"
              << std::endl;
}

std::optional<Args> validate(int argc, char **argv) {
    if (argc < 3 || argc > 5) {
        print_usage(argv[0]);
        return std::nullopt;
    }

    Args args;

    args.A = argv[1];
    if (!fs::exists(args.A)) {
        std::cerr << "Error: File " << argv[1] << " does not exist."
                  << std::endl;
        return std::nullopt;
    }

    args.R = argv[2];
    if (!fs::exists(args.R)) {
        std::cerr << "Error: File " << argv[2] << " does not exist."
                  << std::endl;
        return std::nullopt;
    }

    args.tolerance = 1e-6;
    args.max_iterations = 1;

    try {
        if (argc >= 4) {
            args.tolerance = std::stof(argv[3]);
        }

        if (argc == 5) {
            args.max_iterations = std::stoi(argv[4]);
        }
    } catch (const std::invalid_argument &e) {
        std::cerr << "Error: "
                  << (std::strcmp(e.what(), "stof") == 0
                          ? "Invalid tolerance"
                          : "Invalid max_iterations")
                  << std::endl;
        return std::nullopt;
    }

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

        cusparseHandle_t cusparse;
        cublasHandle_t cublas;

        cusparseCreate(&cusparse);
        cublasCreate_v2(&cublas);

        int iterations = cg_run::cg(cusparse, cublas, A.get(), R.get());

        std::cout << iterations << std::endl;

        cublasDestroy_v2(cublas);
        cusparseDestroy(cusparse);
    } else {
        return -1;
    }

    return 0;
}
