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
        ("L", "L matrix's .mat file", cxxopts::value<std::string>());
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

        if (result.count("L")) {
            mat_utils::SpMatReader L_reader{
                result["L"].as<std::string>(), {}, "L"};
            return Args{*algorithm, *implementation, std::move(A_reader), std::move(L_reader)};
        }

        return Args{*algorithm, *implementation, std::move(A_reader), std::nullopt};
    } catch (const cxxopts::exceptions::exception &e) {
        std::cerr << e.what() << '\n' << std::endl;
        std::cerr << options.help();
        return std::nullopt;
    } catch (const std::exception &e) {
        std::cerr << "Data loading failed" << std::endl;
        return std::nullopt;
    }
}
