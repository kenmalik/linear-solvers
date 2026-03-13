#pragma once

#include <iostream>
#include <mkl_lapacke.h>
#include <mkl_spblas.h>

#define MKL_SPARSE_CHECK(call)                                                 \
    do {                                                                       \
        sparse_status_t _err = (call);                                         \
        if (_err != SPARSE_STATUS_SUCCESS) {                                   \
            std::cerr << "MKL sparse error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << _err << std::endl;                            \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

#define MKL_LAPACKE_CHECK(call)                                                \
    do {                                                                       \
        lapack_int _err = (call);                                              \
        if (_err != 0) {                                                       \
            std::cerr << "MKL LAPACKE error at " << __FILE__ << ":"            \
                      << __LINE__ << ": " << _err << std::endl;                \
            std::abort();                                                      \
        }                                                                      \
    } while (0)
