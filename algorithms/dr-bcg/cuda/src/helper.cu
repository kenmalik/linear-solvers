#include <nvtx3/nvtx3.hpp>

#include "dr_bcg/helper.h"
#include "dr_bcg/internal/math.h"

/**
 * @brief CUDA kernel to copy upper triangular of a matrix stored in
 * column-major order.
 *
 * @param dst Pointer to destination device matrix (n x n)
 * @param src Pointer to source device matrix (m x n)
 * @param m Matrix dimension m
 * @param n Matrix dimension n
 */
__global__ void copy_upper_triangular_kernel(float *dst, float *src,
                                             const int m, const int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col <= row && row < n && col < n) {
        dst[row * n + col] = src[row * m + col];
    }
}

void copy_upper_triangular(float *dst, float *src, const int m, const int n) {
    constexpr int block_n = 16;
    constexpr dim3 block_dim(block_n, block_n);
    dim3 grid_dim((n + block_n - 1) / block_n, (n + block_n - 1) / block_n);
    copy_upper_triangular_kernel<<<grid_dim, block_dim>>>(dst, src, m, n);
}

__global__ void copy_upper_triangular_kernel_double(double *dst, double *src,
                                                    const int m, const int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col <= row && row < n && col < n) {
        dst[row * n + col] = src[row * m + col];
    }
}

void copy_upper_triangular(double *dst, double *src, const int m, const int n) {
    constexpr int block_n = 16;
    constexpr dim3 block_dim(block_n, block_n);
    dim3 grid_dim((n + block_n - 1) / block_n, (n + block_n - 1) / block_n);
    copy_upper_triangular_kernel_double<<<grid_dim, block_dim>>>(dst, src, m,
                                                                 n);
}
