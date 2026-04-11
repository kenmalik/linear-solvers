#pragma once

#include <optional>

#include <mat_utils/mat_reader.h>

enum class Algorithm { CG, DR_BCG };
enum class Implementation { MKL, CUDA };

struct Args {
    Algorithm algorithm;
    Implementation implementation;
    mat_utils::SpMatReader A;
    std::optional<mat_utils::SpMatReader> L;
    std::optional<mat_utils::DnMatReader> b;
    std::optional<mat_utils::DnMatReader> x;
    double tolerance;
    std::optional<int> max_iterations;
    int block_size;
};

std::optional<Args> parse_args(int argc, char *argv[]);
