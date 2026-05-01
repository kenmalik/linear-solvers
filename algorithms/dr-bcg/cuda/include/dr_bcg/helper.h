#pragma once

#include <cuda_runtime.h>

void copy_upper_triangular(float *dst, const float *src, int ld_src, int n,
                           cudaStream_t stream);

void copy_upper_triangular(double *dst, const double *src, int ld_src, int n,
                           cudaStream_t stream);
