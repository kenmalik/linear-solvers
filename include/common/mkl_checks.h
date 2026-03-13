#pragma once

#include <iostream>
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
