#include "parser.h"

#include <cxxopts.hpp>
#include <exception>
#include <iostream>
#include <string>
#include <utility>

static std::optional<Algorithm> parse_algorithm(const std::string &s) {
    if (s == "cg") return Algorithm::CG;
    if (s == "dr-bcg") return Algorithm::DR_BCG;
    return std::nullopt;
}

static std::optional<Implementation> parse_implementation(const std::string &s) {
    if (s == "mkl") return Implementation::MKL;
    if (s == "cuda") return Implementation::CUDA;
    return std::nullopt;
}

std::optional<Args> parse_args(int argc, char *argv[]) {
    cxxopts::Options options("cgrun",
                             "Run conjugate gradient variants on .mat files");

    // clang-format off
    options.add_options()
        ("algorithm", "Algorithm to run (cg)", cxxopts::value<std::string>())
        ("implementation", "Implementation to use (mkl, cuda)", cxxopts::value<std::string>())
        ("A", "A matrix's .mat file", cxxopts::value<std::string>())
        ("L", "L matrix's .mat file", cxxopts::value<std::string>())
        ("b", "b matrix's .mat file containing top-level dense variable b", cxxopts::value<std::string>())
        ("B", "B matrix's .mat file containing top-level dense variable B", cxxopts::value<std::string>())
        ("x", "x matrix's .mat file containing top-level dense variable x", cxxopts::value<std::string>())
        ("X", "X matrix's .mat file containing top-level dense variable X", cxxopts::value<std::string>())
        ("timer-out", "Output file for timings CSV", cxxopts::value<std::string>()->default_value("timings.csv"))
        ("t,tolerance", "Convergence tolerance", cxxopts::value<double>()->default_value("1e-6"))
        ("i,max-iterations", "Maximum number of iterations (default: n)", cxxopts::value<int>())
        ("s,block-size", "Block size (DR-BCG only)", cxxopts::value<int>()->default_value("1"))
        ("no-tensor-cores", "Disable tensor-core-eligible cuBLAS math for CUDA runs",
         cxxopts::value<bool>()->default_value("false")->implicit_value("true"));
    // clang-format on

    options.parse_positional({"algorithm", "implementation", "A", "L"});
    options.positional_help("<algorithm> <implementation> <A> [L]");

    try {
        auto result = options.parse(argc, argv);

        if (!result.count("algorithm")) {
            std::cerr << "Missing required argument: algorithm\n" << std::endl;
            std::cerr << options.help();
            return std::nullopt;
        }

        auto algorithm = parse_algorithm(result["algorithm"].as<std::string>());
        if (!algorithm) {
            std::cerr << "Unknown algorithm: " << result["algorithm"].as<std::string>() << "\n"
                      << "Available: cg, dr-bcg\n" << std::endl;
            std::cerr << options.help();
            return std::nullopt;
        }

        if (!result.count("implementation")) {
            std::cerr << "Missing required argument: implementation\n" << std::endl;
            std::cerr << options.help();
            return std::nullopt;
        }

        auto implementation = parse_implementation(result["implementation"].as<std::string>());
        if (!implementation) {
            std::cerr << "Unknown implementation: " << result["implementation"].as<std::string>() << "\n"
                      << "Available: mkl, cuda\n" << std::endl;
            std::cerr << options.help();
            return std::nullopt;
        }

        if (!result.count("A")) {
            std::cerr << "Missing required argument: A\n" << std::endl;
            std::cerr << options.help();
            return std::nullopt;
        }

        mat_utils::SpMatReader A_reader{
            result["A"].as<std::string>(), {"Problem"}, "A"};

        double tolerance = result["tolerance"].as<double>();
        std::optional<int> max_iterations;
        if (result.count("max-iterations"))
            max_iterations = result["max-iterations"].as<int>();
        int block_size = result["block-size"].as<int>();
        bool disable_tensor_cores = result["no-tensor-cores"].as<bool>();
        std::optional<mat_utils::DnMatReader> b_reader;
        if (result.count("b")) {
            b_reader.emplace(result["b"].as<std::string>(), std::vector<std::string>{}, "b");
        }
        std::optional<mat_utils::DnMatReader> B_reader;
        if (result.count("B")) {
            B_reader.emplace(result["B"].as<std::string>(), std::vector<std::string>{}, "B");
        }
        std::optional<mat_utils::DnMatReader> x_reader;
        if (result.count("x")) {
            x_reader.emplace(result["x"].as<std::string>(), std::vector<std::string>{}, "x");
        }
        std::optional<mat_utils::DnMatReader> X_reader;
        if (result.count("X")) {
            X_reader.emplace(result["X"].as<std::string>(), std::vector<std::string>{}, "X");
        }
        auto timer_out = result["timer-out"].as<std::string>();
        if (!timer_out.ends_with(".csv")) {
            timer_out += ".csv";
        }

        if (result.count("L")) {
            mat_utils::SpMatReader L_reader{
                result["L"].as<std::string>(), {}, "L"};
            return Args{*algorithm, *implementation, std::move(A_reader), std::move(L_reader),
                        std::move(b_reader), std::move(B_reader), std::move(x_reader),
                        std::move(X_reader), timer_out, tolerance,
                        max_iterations, block_size, disable_tensor_cores};
        }

        return Args{*algorithm, *implementation, std::move(A_reader), std::nullopt,
                    std::move(b_reader), std::move(B_reader), std::move(x_reader),
                    std::move(X_reader), timer_out, tolerance,
                    max_iterations, block_size, disable_tensor_cores};
    } catch (const cxxopts::exceptions::exception &e) {
        std::cerr << e.what() << '\n' << std::endl;
        std::cerr << options.help();
        return std::nullopt;
    } catch (const std::exception &e) {
        std::cerr << "Data loading failed: " << e.what() << std::endl;
        return std::nullopt;
    }
}
