#pragma once

#include <cstdint>

namespace dr_bcg {

int dr_bcg(float *A, float *X, float *B, std::int64_t n, std::int64_t s,
           float tolerance = 1e-6, std::int64_t max_iterations = 100);

}
