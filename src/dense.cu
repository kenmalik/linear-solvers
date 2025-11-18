#include <dr_bcg/dense.h>

#include <iostream>

int dr_bcg(float *A, float *X, float *B, std::int64_t n, std::int64_t s,
           float tolerance, std::int64_t max_iterations) {
    std::cout << "Dense dr_bcg" << std::endl;

    int iterations = 0;
    while (iterations < max_iterations) {
        ++iterations;
    }

    return iterations;
}