#include "rhs.h"

#include <algorithm>
#include <cstddef>
#include <random>
#include <sstream>
#include <stdexcept>

namespace {

constexpr std::uint32_t rhs_seed = 0x5EED1234U;

std::vector<double> generate_random_rhs(std::size_t size) {
    std::mt19937 gen(rhs_seed); // NOLINT
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> rhs(size);
    std::ranges::generate(rhs, [&] { return dist(gen); });
    return rhs;
}

std::vector<double> generate_zero_initial_guess(std::size_t size) {
    return std::vector<double>(size, 0.0);
}

std::vector<double>
prepare_dense_input(const std::optional<mat_utils::DnMatReader> &reader,
                    std::size_t expected_rows, std::size_t expected_cols,
                    const char *name, std::vector<double> (*fallback)(std::size_t)) {
    if (!reader.has_value()) {
        return fallback(expected_rows * expected_cols);
    }

    if (reader->rows() != expected_rows || reader->cols() != expected_cols) {
        std::ostringstream oss;
        oss << name << " has shape " << reader->rows() << "x" << reader->cols()
            << ", expected " << expected_rows << "x" << expected_cols;
        throw std::runtime_error(oss.str());
    }

    auto *data = reader->data();
    return {data, data + reader->size()};
}

void validate_dense_input(const mat_utils::DnMatReader &reader,
                          std::size_t expected_rows,
                          std::size_t expected_cols,
                          const char *name) {
    if (reader.rows() != expected_rows || reader.cols() != expected_cols) {
        std::ostringstream oss;
        oss << name << " has shape " << reader.rows() << "x" << reader.cols()
            << ", expected " << expected_rows << "x" << expected_cols;
        throw std::runtime_error(oss.str());
    }
}

std::vector<double> prepare_split_dense_input(
    const std::optional<mat_utils::DnMatReader> &first_reader,
    const std::optional<mat_utils::DnMatReader> &rest_reader,
    std::size_t expected_rows, std::size_t expected_cols, const char *first_name,
    const char *rest_name, std::vector<double> (*fallback)(std::size_t)) {
    if (!first_reader.has_value() && !rest_reader.has_value()) {
        return fallback(expected_rows * expected_cols);
    }

    if (first_reader.has_value() && !rest_reader.has_value()) {
        if (first_reader->cols() == expected_cols) {
            return prepare_dense_input(first_reader, expected_rows, expected_cols,
                                       first_name, fallback);
        }

        if (first_reader->cols() != 1) {
            validate_dense_input(*first_reader, expected_rows, expected_cols,
                                 first_name);
        }
    }

    if (!first_reader.has_value() && rest_reader.has_value() && expected_cols < 2) {
        std::ostringstream oss;
        oss << rest_name << " was provided, but expected column count is "
            << expected_cols;
        throw std::runtime_error(oss.str());
    }

    if (first_reader.has_value() && !rest_reader.has_value() && expected_cols < 2) {
        return prepare_dense_input(first_reader, expected_rows, expected_cols, first_name,
                                   fallback);
    }

    if (expected_cols < 2) {
        std::ostringstream oss;
        oss << rest_name << " was provided, but expected column count is "
            << expected_cols;
        throw std::runtime_error(oss.str());
    }

    std::vector<double> dense_input(expected_rows * expected_cols);

    if (first_reader.has_value()) {
        validate_dense_input(*first_reader, expected_rows, 1, first_name);
        std::copy(first_reader->data(), first_reader->data() + first_reader->size(),
                  dense_input.begin());
    } else {
        auto first_column = fallback(expected_rows);
        std::ranges::copy(first_column, dense_input.begin());
    }

    if (rest_reader.has_value()) {
        validate_dense_input(*rest_reader, expected_rows, expected_cols - 1, rest_name);
        std::copy(rest_reader->data(), rest_reader->data() + rest_reader->size(),
                  dense_input.begin() + static_cast<std::ptrdiff_t>(expected_rows));
    } else {
        auto remaining_columns = fallback(expected_rows * (expected_cols - 1));
        std::ranges::copy(remaining_columns,
                          dense_input.begin() + static_cast<std::ptrdiff_t>(expected_rows));
    }

    return dense_input;
}

} // namespace

std::vector<double>
prepare_rhs(const std::optional<mat_utils::DnMatReader> &rhs_reader,
            const std::optional<mat_utils::DnMatReader> &rhs_rest_reader,
            std::size_t expected_rows, std::size_t expected_cols) {
    return prepare_split_dense_input(rhs_reader, rhs_rest_reader, expected_rows,
                                     expected_cols, "RHS b", "RHS B",
                                     generate_random_rhs);
}

std::vector<double>
prepare_initial_guess(const std::optional<mat_utils::DnMatReader> &x_reader,
                      const std::optional<mat_utils::DnMatReader> &x_rest_reader,
                      std::size_t expected_rows, std::size_t expected_cols) {
    return prepare_split_dense_input(x_reader, x_rest_reader, expected_rows,
                                     expected_cols, "Initial guess x",
                                     "Initial guess X", generate_zero_initial_guess);
}
