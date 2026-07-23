#include "parser.h"

#include <cstddef>
#include <cxxopts.hpp>
#include <exception>
#include <iostream>
#include <optional>
#include <regex>
#include <string>
#include <utility>
#include <vector>

namespace {

std::optional<Algorithm> parse_algorithm(const std::string &s) {
    if (s == "cg") {
        return Algorithm::CG;
    }
    if (s == "dr-bcg") {
        return Algorithm::DR_BCG;
    }
    return std::nullopt;
}

std::optional<Implementation> parse_implementation(const std::string &s) {
    if (s == "mkl") {
        return Implementation::MKL;
    }
    if (s == "cuda") {
        return Implementation::CUDA;
    }
    return std::nullopt;
}

std::optional<QrBackend> parse_qr_backend(const std::string &s) {
    if (s == "householder") {
        return QrBackend::Householder;
    }
    if (s == "cholqr") {
        return QrBackend::CholQR;
    }
    if (s == "cholqr-dx") {
        return QrBackend::CholQRDx;
    }
    return std::nullopt;
}

struct MatArg {
    std::string file;
    std::vector<std::string> parent_arrays;
    std::string field;
};

// Format: mat_file_path[:parent_arrays/field]
// If the `:parent_arrays/field` suffix is omitted, default_parent_arrays and
// default_field are used instead.
std::optional<MatArg> parse_mat_arg(const std::string &s,
                                    std::vector<std::string> default_parent_arrays,
                                    std::string default_field) {
    if (s.find(':') == std::string::npos) {
        if (s.empty()) {
            return std::nullopt;
        }
        return MatArg{.file = s, .parent_arrays = std::move(default_parent_arrays), .field = std::move(default_field)};
    }

    static const std::regex pattern(R"(^([^:]+):((?:/[^/]+)+)$)");
    std::smatch match;
    if (!std::regex_match(s, match, pattern)) {
        return std::nullopt;
    }

    std::string file = match[1];
    std::string path = match[2];

    std::vector<std::string> components;
    size_t start = 1; // skip leading '/'
    while (start < path.size()) {
        size_t end = path.find('/', start);
        if (end == std::string::npos) {
            end = path.size();
        }
        components.push_back(path.substr(start, end - start));
        start = end + 1;
    }

    std::string field = std::move(components.back());
    components.pop_back();

    return MatArg{.file = std::move(file), .parent_arrays = std::move(components), .field = std::move(field)};
}

} // namespace

std::optional<Args> parse_args(int argc, char *argv[]) { // NOLINT(*avoid-c-arrays)
    using mat_utils::MatReader;

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
        ("o,output", "Output .mat file for the solution X", cxxopts::value<std::string>())
        ("output-b", "Also write the used B into the --output .mat file",
         cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
        ("t,tolerance", "Convergence tolerance", cxxopts::value<double>()->default_value("1e-6"))
        ("i,max-iterations", "Maximum number of iterations (default: n)", cxxopts::value<int>())
        ("s,block-size", "Block size (DR-BCG only)", cxxopts::value<int>()->default_value("1"))
        ("qr-backend", "CUDA DR-BCG orthonormalization backend (householder, cholqr, cholqr-dx)",
         cxxopts::value<std::string>()->default_value("householder"))
        ("fused-xi", "Enable the fused MathDx reduced-system (xi) chain for CUDA DR-BCG (requires SOLVERS_BUILD_MATHDX)",
         cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
        ("no-tensor-cores", "Disable tensor-core-eligible cuBLAS math for CUDA runs",
         cxxopts::value<bool>()->default_value("false")->implicit_value("true"));
    // clang-format on

    options.parse_positional({"algorithm", "implementation", "A", "L"});
    options.positional_help("<algorithm> <implementation> <A> [L]");

    auto parse_mat_option = [&options](const std::string &name, const std::string &value,
                                       std::vector<std::string> default_parent_arrays,
                                       std::string default_field) -> std::optional<MatArg> {
        auto arg = parse_mat_arg(value, std::move(default_parent_arrays), std::move(default_field));
        if (!arg) {
            std::cerr << "Invalid format for --" << name << ": " << value << "\n"
                      << "Expected format: file[:/parent_arrays/field]\n\n";
            std::cerr << options.help();
        }
        return arg;
    };

    try {
        auto parsed = options.parse(argc, argv);

        if (!parsed.contains("algorithm")) {
            std::cerr << "Missing required argument: algorithm\n\n";
            std::cerr << options.help();
            return std::nullopt;
        }

        auto algorithm = parse_algorithm(parsed["algorithm"].as<std::string>());
        if (!algorithm) {
            std::cerr << "Unknown algorithm: " << parsed["algorithm"].as<std::string>() << "\n"
                      << "Available: cg, dr-bcg\n\n";
            std::cerr << options.help();
            return std::nullopt;
        }

        if (!parsed.contains("implementation")) {
            std::cerr << "Missing required argument: implementation\n\n";
            std::cerr << options.help();
            return std::nullopt;
        }

        auto implementation = parse_implementation(parsed["implementation"].as<std::string>());
        if (!implementation) {
            std::cerr << "Unknown implementation: " << parsed["implementation"].as<std::string>() << "\n"
                      << "Available: mkl, cuda\n\n";
            std::cerr << options.help();
            return std::nullopt;
        }

        if (!parsed.contains("A")) {
            std::cerr << "Missing required argument: A\n\n";
            std::cerr << options.help();
            return std::nullopt;
        }

        auto A_arg = parse_mat_option("A", parsed["A"].as<std::string>(), {"Problem"}, "A");
        if (!A_arg) {
            return std::nullopt;
        }
        MatReader<mat_utils::Sparsity::Sparse> A_reader{A_arg->file, A_arg->parent_arrays, A_arg->field};

        double tolerance = parsed["tolerance"].as<double>();
        std::optional<int> max_iterations;
        if (parsed.contains("max-iterations")) {
            max_iterations = parsed["max-iterations"].as<int>();
        }
        int block_size = parsed["block-size"].as<int>();
        bool disable_tensor_cores = parsed["no-tensor-cores"].as<bool>();
        bool fused_xi = parsed["fused-xi"].as<bool>();
        auto qr_backend = parse_qr_backend(parsed["qr-backend"].as<std::string>());
        if (!qr_backend) {
            std::cerr << "Unknown QR backend: "
                      << parsed["qr-backend"].as<std::string>() << "\n"
                      << "Available: householder, cholqr, cholqr-dx\n\n";
            std::cerr << options.help();
            return std::nullopt;
        }
        std::optional<MatReader<>> b_reader;
        if (parsed.contains("b")) {
            auto b_arg = parse_mat_option("b", parsed["b"].as<std::string>(), {}, "b");
            if (!b_arg) {
                return std::nullopt;
            }
            b_reader.emplace(b_arg->file, b_arg->parent_arrays, b_arg->field);
        }
        std::optional<MatReader<>> B_reader;
        if (parsed.contains("B")) {
            auto B_arg = parse_mat_option("B", parsed["B"].as<std::string>(), {}, "B");
            if (!B_arg) {
                return std::nullopt;
            }
            B_reader.emplace(B_arg->file, B_arg->parent_arrays, B_arg->field);
        }
        std::optional<MatReader<>> x_reader;
        if (parsed.contains("x")) {
            auto x_arg = parse_mat_option("x", parsed["x"].as<std::string>(), {}, "x");
            if (!x_arg) {
                return std::nullopt;
            }
            x_reader.emplace(x_arg->file, x_arg->parent_arrays, x_arg->field);
        }
        std::optional<MatReader<>> X_reader;
        if (parsed.contains("X")) {
            auto X_arg = parse_mat_option("X", parsed["X"].as<std::string>(), {}, "X");
            if (!X_arg) {
                return std::nullopt;
            }
            X_reader.emplace(X_arg->file, X_arg->parent_arrays, X_arg->field);
        }
        auto timer_out = parsed["timer-out"].as<std::string>();
        if (!timer_out.ends_with(".csv")) {
            timer_out += ".csv";
        }

        std::optional<std::string> output;
        if (parsed.contains("output")) {
            output = parsed["output"].as<std::string>();
            if (!output->ends_with(".mat")) {
                *output += ".mat";
            }
        }
        bool output_b = parsed["output-b"].as<bool>();

        Args res{.algorithm = *algorithm,
                 .implementation = *implementation,
                 .A = std::move(A_reader),
                 .L = std::nullopt,
                 .b = std::move(b_reader),
                 .B = std::move(B_reader),
                 .x = std::move(x_reader),
                 .X = std::move(X_reader),
                 .timer_out = timer_out,
                 .output = std::move(output),
                 .output_b = output_b,
                 .tolerance = tolerance,
                 .max_iterations = max_iterations,
                 .block_size = block_size,
                 .disable_tensor_cores = disable_tensor_cores,
                 .qr_backend = *qr_backend,
                 .fused_xi = fused_xi};

        if (parsed.contains("L")) {
            auto L_arg = parse_mat_option("L", parsed["L"].as<std::string>(), {}, "L");
            if (!L_arg) {
                return std::nullopt;
            }
            MatReader<mat_utils::Sparsity::Sparse> L_reader{L_arg->file, L_arg->parent_arrays, L_arg->field};
            res.L = std::move(L_reader);
        }

        return res;
    } catch (const cxxopts::exceptions::exception &e) {
        std::cerr << e.what() << "\n\n";
        std::cerr << options.help();
        return std::nullopt;
    } catch (const std::exception &e) {
        std::cerr << "Data loading failed: " << e.what() << '\n';
        return std::nullopt;
    }
}
