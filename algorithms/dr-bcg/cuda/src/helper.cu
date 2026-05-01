#include <nvtx3/nvtx3.hpp>

#include "dr_bcg/helper.h"
#include "dr_bcg/internal/math.h"

__global__ void copy_upper_triangular_kernel(float *dst, const float *src,
                                             const int ld_src, const int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        dst[row + col * n] = (row <= col) ? src[row + col * ld_src] : 0.0f;
    }
}

void copy_upper_triangular(float *dst, const float *src, int ld_src, int n,
                           cudaStream_t stream) {
    constexpr int block_n = 16;
    constexpr dim3 block_dim(block_n, block_n);
    dim3 grid_dim((n + block_n - 1) / block_n, (n + block_n - 1) / block_n);
    copy_upper_triangular_kernel<<<grid_dim, block_dim, 0, stream>>>(
        dst, src, ld_src, n);
}

__global__ void copy_upper_triangular_kernel_double(double *dst,
                                                    const double *src,
                                                    const int ld_src,
                                                    const int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        dst[row + col * n] = (row <= col) ? src[row + col * ld_src] : 0.0;
    }
}

void copy_upper_triangular(double *dst, const double *src, int ld_src, int n,
                           cudaStream_t stream) {
    constexpr int block_n = 16;
    constexpr dim3 block_dim(block_n, block_n);
    dim3 grid_dim((n + block_n - 1) / block_n, (n + block_n - 1) / block_n);
    copy_upper_triangular_kernel_double<<<grid_dim, block_dim, 0, stream>>>(
        dst, src, ld_src, n);
}
