#pragma once

#include <cstdint>

namespace dr_bcg {

int dr_bcg(float *d_A, float *d_X, float *d_B, std::int64_t n, std::int64_t s,
           float tolerance = 1e-6, std::int64_t max_iterations = 100);

int dr_bcg(double *d_A, double *d_X, double *d_B, std::int64_t n,
           std::int64_t s, double tolerance = 1e-6,
           std::int64_t max_iterations = 100);

} // namespace dr_bcg
