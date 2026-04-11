#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include <mat_utils/mat_reader.h>

std::vector<double>
prepare_rhs(const std::optional<mat_utils::DnMatReader> &rhs_reader,
            std::size_t expected_rows, std::size_t expected_cols);
