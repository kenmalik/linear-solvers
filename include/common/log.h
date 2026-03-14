#pragma once

#ifdef SOLVERS_VERBOSE
#include <iostream>

#define LOG_TRACE(v)                                                           \
    do {                                                                       \
        std::cerr << v << std::endl;                                           \
    } while (0)
#else
#define LOG_TRACE(v)
#endif
