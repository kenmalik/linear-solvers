#pragma once

#include "common/supported_type.h"

#include <mat_utils/mat_reader.h>

#include <cstddef>
#include <optional>
#include <vector>

template <cils::detail::SupportedType T>
std::vector<T>
prepare_rhs(const std::optional<mat_utils::MatReader<>> &rhs_reader,
            const std::optional<mat_utils::MatReader<>> &rhs_rest_reader,
            std::size_t expected_rows, std::size_t expected_cols);

template <cils::detail::SupportedType T>
std::vector<T>
prepare_initial_guess(const std::optional<mat_utils::MatReader<>> &x_reader,
                      const std::optional<mat_utils::MatReader<>> &x_rest_reader,
                      std::size_t expected_rows, std::size_t expected_cols);

extern template std::vector<double> prepare_rhs<double>(
    const std::optional<mat_utils::MatReader<>> &, const std::optional<mat_utils::MatReader<>> &,
    std::size_t, std::size_t);
extern template std::vector<float> prepare_rhs<float>(
    const std::optional<mat_utils::MatReader<>> &, const std::optional<mat_utils::MatReader<>> &,
    std::size_t, std::size_t);

extern template std::vector<double> prepare_initial_guess<double>(
    const std::optional<mat_utils::MatReader<>> &, const std::optional<mat_utils::MatReader<>> &,
    std::size_t, std::size_t);
extern template std::vector<float> prepare_initial_guess<float>(
    const std::optional<mat_utils::MatReader<>> &, const std::optional<mat_utils::MatReader<>> &,
    std::size_t, std::size_t);
