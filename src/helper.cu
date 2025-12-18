#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

#include <nvtx3/nvtx3.hpp>

#include "dr_bcg/helper.h"
#include "dr_bcg/internal/math.h"

/**
 * @brief Checks CUDA runtime API results and throws an exception on error.
 *
 * @param err CUDA error code returned by a CUDA runtime API call.
 * @param func Name of the function where the check is performed.
 * @param file Source file name where the check is performed.
 * @param line Line number in the source file where the check is performed.
 *
 * @throws std::runtime_error if the CUDA error code is not cudaSuccess.
 */
void check(cudaError_t err, const char *const func, const char *const file,
           const int line) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error " << err << " at " << file << " line " << line
            << ": " << func << ": " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

/**
 * @brief Checks cuSOLVER API results and throws an exception on error.
 *
 * @param err CUDA error code returned by a CUDA runtime API call.
 * @param func Name of the function where the check is performed.
 * @param file Source file name where the check is performed.
 * @param line Line number in the source file where the check is performed.
 *
 * @throws std::runtime_error if the cuSOLVER error code is not
 * CUSOLVER_STATUS_SUCCESS.
 */
void check(cusolverStatus_t err, const char *const func, const char *const file,
           const int line) {
    if (err != CUSOLVER_STATUS_SUCCESS) {
        std::ostringstream oss;
        oss << "cuSOLVER error " << err << " at " << file << " line " << line
            << ": " << func;
        throw std::runtime_error(oss.str());
    }
}

/**
 * @brief Checks cuBLAS API results and throws an exception on error.
 *
 * @param err cuBLAS error code returned by a cuBLAS API call.
 * @param func Name of the function where the check is performed.
 * @param file Source file name where the check is performed.
 * @param line Line number in the source file where the check is performed.
 *
 * @throws std::runtime_error if the cuBLAS error code is not
 * CUBLAS_STATUS_SUCCESS.
 */
void check(cublasStatus_t err, const char *const func, const char *const file,
           const int line) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        std::ostringstream oss;
        oss << "cuBLAS error " << err << " at " << file << " line " << line
            << ": " << func;
        throw std::runtime_error(oss.str());
    }
}

/**
 * @brief Checks cuSPARSE API results and throws an exception on error.
 *
 * @param err cuSPARSE error code returned by a cuSPARSE API call.
 * @param func Name of the function where the check is performed.
 * @param file Source file name where the check is performed.
 * @param line Line number in the source file where the check is performed.
 *
 * @throws std::runtime_error if the cuSPARSE error code is not
 * CUSPARSE_STATUS_SUCCESS.
 */
void check(cusparseStatus_t err, const char *const func, const char *const file,
           const int line) {
    if (err != CUSPARSE_STATUS_SUCCESS) {
        std::ostringstream oss;
        oss << "cuSPARSE error " << err << " at " << file << " line " << line
            << ": " << func << ": " << cusparseGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

/**
 * @brief Prints a matrix stored in column-major order.
 *
 * @param mat Pointer to the matrix data (column-major)
 * @param rows Number of rows in the matrix
 * @param cols Number of columns in the matrix
 */
void print_matrix(const float *mat, const int rows, const int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%7.2f ", mat[j * rows + i]);
        }
        std::cout << std::endl;
    }
}

/**
 * @brief Prints a device matrix by copying it to host and calling print_matrix.
 *
 * @param d_mat Device pointer to the matrix (column-major)
 * @param rows Number of rows in the matrix
 * @param cols Number of columns in the matrix
 */
void print_device_matrix(const float *d_mat, const int rows, const int cols) {
    std::vector<float> h_mat(rows * cols);
    CUDA_CHECK(cudaMemcpy(h_mat.data(), d_mat, sizeof(float) * rows * cols,
                          cudaMemcpyDeviceToHost));
    print_matrix(h_mat.data(), rows, cols);
}

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

void print_sparse_matrix(const cusparseHandle_t &cusparseH,
                         const cusparseSpMatDescr_t &sp_mat) {
    constexpr cusparseSparseToDenseAlg_t ALG =
        CUSPARSE_SPARSETODENSE_ALG_DEFAULT;

    size_t buffer_size = 0;
    void *buffer = nullptr;

    int64_t rows = 0;
    int64_t cols = 0;
    int64_t nnz = 0;

    float *dense_d = nullptr;
    cusparseDnMatDescr_t dense{};

    CUSPARSE_CHECK(cusparseSpMatGetSize(sp_mat, &rows, &cols, &nnz));
    CUDA_CHECK(cudaMalloc(&dense_d, sizeof(float) * rows * cols));
    CUSPARSE_CHECK(cusparseCreateDnMat(&dense, rows, cols, rows, dense_d,
                                       CUDA_R_32F, CUSPARSE_ORDER_COL));

    CUSPARSE_CHECK(cusparseSparseToDense_bufferSize(cusparseH, sp_mat, dense,
                                                    ALG, &buffer_size));
    CUDA_CHECK(cudaMalloc(&buffer, buffer_size));
    CUSPARSE_CHECK(
        cusparseSparseToDense(cusparseH, sp_mat, dense, ALG, buffer));

    print_device_matrix(dense_d, rows, cols);

    CUDA_CHECK(cudaFree(buffer));
    CUDA_CHECK(cudaFree(dense_d));
    CUSPARSE_CHECK(cusparseDestroyDnMat(dense));
}

void sptri_left_multiply(const cusparseHandle_t &cusparseH,
                         cusparseDnMatDescr_t &C, cusparseOperation_t opA,
                         const cusparseSpMatDescr_t &A,
                         const cusparseDnMatDescr_t &B,
                         const cudaDataType compute_type) {
    if (compute_type == CUDA_R_32F) {
        sptri_left_multiply<float>(cusparseH, C, opA, A, B);
    } else {
        sptri_left_multiply<double>(cusparseH, C, opA, A, B);
    }
}