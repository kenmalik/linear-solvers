#pragma once

#ifdef CUDA_DR_BCG_DEBUG_BUILD

#include <cuda/std/cmath>
#include <thrust/device_vector.h>

#include <cassert>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <type_traits>

#define DEBUG_LOG(val) std::cerr << val << std::endl;
#define DEBUG_SLOG(val, stream) stream << val << std::endl;

#define DEBUG_LOG_DMAT(A, m, n, lda) print_device_matrix(A, m, n, lda);
#define DEBUG_SLOG_DMAT(A, m, n, lda, stream)                                  \
    print_device_matrix(A, m, n, lda, stream);

// Print m by n column-major device matrix
template <typename T>
void print_device_matrix(const T *d_A, int m, int n, int lda,
                         std::ostream &os = std::cerr) {
    static_assert(std::is_same<T, float>::value ||
                      std::is_same<T, double>::value,
                  "print_device_matrix only supports float or double");
    assert(m <= lda && "m must be less than or equal to lda");

    thrust::device_ptr<const T> begin{d_A};

    auto original_precision = std::cerr.precision();
    os << std::scientific << std::setprecision(5);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            os << std::setw(12) << *(begin + j * lda + i) << " ";
        }
        os << std::endl;
    }

    os << std::defaultfloat << std::setprecision(original_precision);
}

#define NON_FINITE_CHECK(mat, size, step, iteration)                           \
    check_non_finite(mat, size, step, iteration);

struct is_non_finite {
    __device__ bool operator()(const float x) {
        return !cuda::std::isfinite(x);
    }
};

/**
 * @brief Checks for NaN values in a device array.
 *
 * @param d_arr Device pointer to the array to check.
 * @param size Number of elements in the array.
 * @param step Description of the step after which the check is performed.
 * @param iteration Iteration in which the check is performed.
 *
 * @throws std::runtime_error if a NaN value is detected in the array.
 */
void check_non_finite(const float *d_arr, size_t size, const char *step,
                      int iteration) {
    thrust::device_ptr<const float> begin{d_arr};
    auto first_nan = thrust::find_if(begin, begin + size, is_non_finite{});
    if (first_nan != begin + size) {
        std::ostringstream oss;
        oss << "Non-finite detected after step: " << step << " (iteration "
            << iteration << ") at value "
            << std::to_string(thrust::distance(begin, first_nan)) << " ("
            << std::to_string(*first_nan) << ")";
        throw std::runtime_error(oss.str());
    }
}

#else

#define DEBUG_LOG(val)                                                         \
    do {                                                                       \
    } while (0);

#define DEBUG_SLOG(val, stream)                                                \
    do {                                                                       \
    } while (0);

#define DEBUG_LOG_DMAT(A, m, n, lda)                                           \
    do {                                                                       \
    } while (0);

#define DEBUG_SLOG_DMAT(A, m, n, lda, stream)                                  \
    do {                                                                       \
    } while (0);

#define NON_FINITE_CHECK(mat, size, step, iteration)                           \
    do {                                                                       \
    } while (0);

#endif
