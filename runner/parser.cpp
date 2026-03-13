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
        ("t,tolerance", "Convergence tolerance", cxxopts::value<double>()->default_value("1e-6"))
        ("i,max-iterations", "Maximum number of iterations (default: n)", cxxopts::value<int>())
        ("s,block-size", "Block size (DR-BCG only)", cxxopts::value<int>()->default_value("1"));
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
                      << "Available: cg\n" << std::endl;
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

        if (result.count("L")) {
            mat_utils::SpMatReader L_reader{
                result["L"].as<std::string>(), {}, "L"};
            return Args{*algorithm, *implementation, std::move(A_reader), std::move(L_reader),
                        tolerance, max_iterations, block_size};
        }

        return Args{*algorithm, *implementation, std::move(A_reader), std::nullopt,
                    tolerance, max_iterations, block_size};
    } catch (const cxxopts::exceptions::exception &e) {
        std::cerr << e.what() << '\n' << std::endl;
        std::cerr << options.help();
        return std::nullopt;
    } catch (const std::exception &e) {
        std::cerr << "Data loading failed" << std::endl;
        return std::nullopt;
    }
}
