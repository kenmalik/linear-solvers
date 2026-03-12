#pragma once

#include <optional>

#include <mat_utils/mat_reader.h>

struct Args {
    mat_utils::SpMatReader A;
    std::optional<mat_utils::SpMatReader> L;
};

std::optional<Args> parse_args(int argc, char *argv[]);
