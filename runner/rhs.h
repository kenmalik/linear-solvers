#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include <mat_utils/mat_reader.h>

std::vector<double>
prepare_rhs(const std::optional<mat_utils::MatReader<>> &rhs_reader,
            const std::optional<mat_utils::MatReader<>> &rhs_rest_reader,
            std::size_t expected_rows, std::size_t expected_cols);

std::vector<double>
prepare_initial_guess(const std::optional<mat_utils::MatReader<>> &x_reader,
                      const std::optional<mat_utils::MatReader<>> &x_rest_reader,
                      std::size_t expected_rows, std::size_t expected_cols);
