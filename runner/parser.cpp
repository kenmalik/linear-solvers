#include "parser.h"

#include <cxxopts.hpp>
#include <exception>
#include <iostream>
#include <string>
#include <utility>

std::optional<Args> parse_args(int argc, char *argv[]) {
    cxxopts::Options options("cgrun",
                             "Run conjugate gradient variants on .mat files");

    // clang-format off
    options.add_options()
        ("A", "A matrix's .mat file", cxxopts::value<std::string>())
        ("L", "L matrix's .mat file", cxxopts::value<std::string>());
    // clang-format on

    options.parse_positional({"A", "L"});
    try {
        auto result = options.parse(argc, argv);

        if (!result.count("A")) {
            std::cerr << "Missing required argument A\n" << std::endl;
            std::cerr << options.help();
            return std::nullopt;
        }

        mat_utils::SpMatReader A_reader{
            result["A"].as<std::string>(), {"Problem"}, "A"};

        if (result.count("L")) {
            mat_utils::SpMatReader L_reader{
                result["L"].as<std::string>(), {}, "L"};
            return Args{std::move(A_reader), std::move(L_reader)};
        }

        return Args{std::move(A_reader), std::nullopt};
    } catch (const cxxopts::exceptions::exception &e) {
        std::cerr << e.what() << '\n' << std::endl;
        std::cerr << options.help();
        return std::nullopt;
    } catch (const std::exception &e) {
        std::cerr << "Data loading failed" << std::endl;
        return std::nullopt;
    }
}
