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

} // namespace

std::vector<double>
prepare_rhs(const std::optional<mat_utils::DnMatReader> &rhs_reader,
            std::size_t expected_rows, std::size_t expected_cols) {
    if (!rhs_reader.has_value()) {
        return generate_random_rhs(expected_rows * expected_cols);
    }

    if (rhs_reader->rows() != expected_rows || rhs_reader->cols() != expected_cols) {
        std::ostringstream oss;
        oss << "RHS b has shape " << rhs_reader->rows() << "x" << rhs_reader->cols()
            << ", expected " << expected_rows << "x" << expected_cols;
        throw std::runtime_error(oss.str());
    }

    auto *data = rhs_reader->data();
    return std::vector<double>(data, data + rhs_reader->size());
}
