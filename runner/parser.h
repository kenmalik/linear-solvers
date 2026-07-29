#pragma once

#include <mat_utils/mat_reader.h>

#include <cstdint>
#include <optional>
#include <string>

enum class QrBackend : std::uint8_t;

enum class Algorithm : std::uint8_t { CG,
                                      DR_BCG };
enum class Implementation : std::uint8_t { MKL,
                                           CUDA };

struct Args {
    Algorithm algorithm;
    Implementation implementation;
    mat_utils::MatReader<mat_utils::Sparsity::Sparse> A;
    std::optional<mat_utils::MatReader<mat_utils::Sparsity::Sparse>> L;
    std::optional<mat_utils::MatReader<>> b;
    std::optional<mat_utils::MatReader<>> B;
    std::optional<mat_utils::MatReader<>> x;
    std::optional<mat_utils::MatReader<>> X;
    std::string timer_out;
    std::optional<std::string> output;
    bool output_b;
    double tolerance;
    std::optional<int> max_iterations;
    int block_size;
    bool disable_tensor_cores;
    QrBackend qr_backend;
    bool fused_xi;
};

std::optional<Args> parse_args(int argc, char *argv[]); // NOLINT(*avoid-c-arrays)
