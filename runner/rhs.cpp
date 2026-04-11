#include "rhs.h"

#include <algorithm>
#include <random>
#include <sstream>
#include <stdexcept>

namespace {

std::vector<double> generate_random_rhs(std::size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> rhs(size);
    std::generate(rhs.begin(), rhs.end(), [&] { return dist(gen); });
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
    return std::vector<double>(data, data + reader->size());
}

} // namespace

std::vector<double>
prepare_rhs(const std::optional<mat_utils::DnMatReader> &rhs_reader,
            std::size_t expected_rows, std::size_t expected_cols) {
    return prepare_dense_input(rhs_reader, expected_rows, expected_cols, "RHS b",
                               generate_random_rhs);
}

std::vector<double>
prepare_initial_guess(const std::optional<mat_utils::DnMatReader> &x_reader,
                      std::size_t expected_rows, std::size_t expected_cols) {
    return prepare_dense_input(x_reader, expected_rows, expected_cols,
                               "Initial guess x", generate_zero_initial_guess);
}
