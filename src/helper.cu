#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

#include <nvtx3/nvtx3.hpp>

#include "dr_bcg/helper.h"

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

std::vector<double> read_matrix_bin(std::string filename) {
    std::ifstream input_file(filename, std::ios::binary);
    if (!input_file.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        exit(1);
    }

    input_file.seekg(0, std::ios::end);
    long long file_size = input_file.tellg();
    input_file.seekg(0, std::ios::beg);

    size_t num_doubles = file_size / sizeof(double);
    std::vector<double> matrix(num_doubles);
    input_file.read(reinterpret_cast<char *>(matrix.data()), file_size);

    return matrix;
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

/**
 * @brief Computes the inverse of a matrix using LU factorization.
 *
 * @param cusolverH cuSOLVER handle
 * @param params cuSOLVER params
 * @param d_A Device pointer to the symmetric positive definite matrix to
 * invert. Result is overwritten to pointed location.
 * @param n Matrix dimension
 */
void invert_square_matrix(cusolverDnHandle_t &cusolverH,
                          cusolverDnParams_t &params, float *d_A, const int n) {
    constexpr cudaDataType_t data_type = CUDA_R_32F;

    // LU Decomposition
    size_t d_work_size = 0;
    void *d_work = nullptr;
    size_t h_work_size = 0;
    void *h_work = nullptr;

    int info = 0;
    int *d_info = nullptr;

    int64_t *d_Ipiv = nullptr;

    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_Ipiv), sizeof(int64_t) * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    CUSOLVER_CHECK(cusolverDnXgetrf_bufferSize(cusolverH, params, n, n,
                                               data_type, d_A, n, data_type,
                                               &d_work_size, &h_work_size));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), d_work_size));
    if (h_work_size > 0) {
        h_work = reinterpret_cast<void *>(malloc(h_work_size));
        if (h_work == nullptr) {
            throw std::runtime_error("Error: h_work not allocated.");
        }
    }

    CUSOLVER_CHECK(cusolverDnXgetrf(cusolverH, params, n, n, data_type, d_A, n,
                                    d_Ipiv, data_type, d_work, d_work_size,
                                    h_work, h_work_size, d_info));

    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (0 > info) {
        throw std::runtime_error(std::to_string(-info) +
                                 "-th parameter is wrong \n");
    }

    CUDA_CHECK(cudaFree(d_work));
    free(h_work);

    // Solve A * X = I for inverse
    std::vector<float> h_I(n * n, 0);
    float *d_I = nullptr;

    for (int i = 0; i < n; i++) {
        h_I.at(i * n + i) = 1;
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_I),
                          sizeof(float) * h_I.size()));
    CUDA_CHECK(cudaMemcpy(d_I, h_I.data(), sizeof(float) * h_I.size(),
                          cudaMemcpyHostToDevice));

    CUSOLVER_CHECK(cusolverDnXgetrs(cusolverH, params, CUBLAS_OP_N, n, n,
                                    data_type, d_A, n, d_Ipiv, data_type, d_I,
                                    n, d_info));

    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (0 > info) {
        throw std::runtime_error(std::to_string(-info) +
                                 "-th parameter is wrong \n");
    }

    CUDA_CHECK(cudaMemcpy(d_A, d_I, sizeof(float) * h_I.size(),
                          cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaFree(d_I));
    CUDA_CHECK(cudaFree(d_Ipiv));
    CUDA_CHECK(cudaFree(d_info));
}

void invert_square_matrix(cusolverDnHandle_t &cusolverH,
                          cusolverDnParams_t &params, double *d_A,
                          const int n) {
    constexpr cudaDataType_t data_type = CUDA_R_64F;

    // LU Decomposition
    size_t d_work_size = 0;
    void *d_work = nullptr;
    size_t h_work_size = 0;
    void *h_work = nullptr;

    int info = 0;
    int *d_info = nullptr;

    int64_t *d_Ipiv = nullptr;

    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_Ipiv), sizeof(int64_t) * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    CUSOLVER_CHECK(cusolverDnXgetrf_bufferSize(cusolverH, params, n, n,
                                               data_type, d_A, n, data_type,
                                               &d_work_size, &h_work_size));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), d_work_size));
    if (h_work_size > 0) {
        h_work = reinterpret_cast<void *>(malloc(h_work_size));
        if (h_work == nullptr) {
            throw std::runtime_error("Error: h_work not allocated.");
        }
    }

    CUSOLVER_CHECK(cusolverDnXgetrf(cusolverH, params, n, n, data_type, d_A, n,
                                    d_Ipiv, data_type, d_work, d_work_size,
                                    h_work, h_work_size, d_info));

    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (0 > info) {
        throw std::runtime_error(std::to_string(-info) +
                                 "-th parameter is wrong \n");
    }

    CUDA_CHECK(cudaFree(d_work));
    free(h_work);

    // Solve A * X = I for inverse
    std::vector<double> h_I(n * n, 0);
    double *d_I = nullptr;

    for (int i = 0; i < n; i++) {
        h_I.at(i * n + i) = 1;
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_I),
                          sizeof(double) * h_I.size()));
    CUDA_CHECK(cudaMemcpy(d_I, h_I.data(), sizeof(double) * h_I.size(),
                          cudaMemcpyHostToDevice));

    CUSOLVER_CHECK(cusolverDnXgetrs(cusolverH, params, CUBLAS_OP_N, n, n,
                                    data_type, d_A, n, d_Ipiv, data_type, d_I,
                                    n, d_info));

    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (0 > info) {
        throw std::runtime_error(std::to_string(-info) +
                                 "-th parameter is wrong \n");
    }

    CUDA_CHECK(cudaMemcpy(d_A, d_I, sizeof(double) * h_I.size(),
                          cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaFree(d_I));
    CUDA_CHECK(cudaFree(d_Ipiv));
    CUDA_CHECK(cudaFree(d_info));
}

/**
 * @brief Computes the QR factorization of matrix A.
 *
 * @param cusolverH cuSOLVER handle
 * @param params Params for the cuSOLVER handle
 * @param d_Q Pointer to device memory to store Q result in
 * @param d_R Pointer to device memory to store R result in. Note that the lower
 * triangular still contains householder vectors and must be handled accordingly
 * (e.g. by using trmm in future multiplications using the R factor)
 * @param m m-dimension (leading dimension) of A
 * @param n n-dimension (second dimension) of A
 * @param d_A The matrix to factorize (device pointer)
 */
void qr_factorization(cusolverDnHandle_t &cusolverH, cusolverDnParams_t &params,
                      float *d_Q, float *d_R, const int m, const int n,
                      const float *d_A) {
    NVTX3_FUNC_RANGE();

    assert(n < m && "Expect cols to be less than rows for DR-BCG");

    int info = 0;

    float *d_tau = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tau, sizeof(float) * n));

    int *d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    void *d_work = nullptr;
    std::size_t lwork_geqrf_d = 0;

    void *h_work = nullptr;
    std::size_t lwork_geqrf_h = 0;

    CUDA_CHECK(
        cudaMemcpy(d_Q, d_A, sizeof(float) * m * n, cudaMemcpyDeviceToDevice));

    // Create device buffer
    CUSOLVER_CHECK(cusolverDnXgeqrf_bufferSize(
        cusolverH, params, m, n, CUDA_R_32F, d_Q, m, CUDA_R_32F, d_tau,
        CUDA_R_32F, &lwork_geqrf_d, &lwork_geqrf_h));

    int numfloats_orgqr_d = 0;
    CUSOLVER_CHECK(cusolverDnSorgqr_bufferSize(cusolverH, m, n, n, d_Q, m,
                                               d_tau, &numfloats_orgqr_d));
    const std::size_t lwork_orgqr_d = numfloats_orgqr_d * sizeof(float);

    // Note: The legacy cuSOLVER API returns lwork number of array values
    // while the generic API returns lwork in bytes.
    // This is why we multiply lwork_orgqr by sizeof(float) to get a
    // proper comparison in workspace sizes.
    const std::size_t lwork_bytes_d = std::max(lwork_geqrf_d, lwork_orgqr_d);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), lwork_bytes_d));

    if (lwork_geqrf_h > 0) {
        h_work = reinterpret_cast<void *>(malloc(lwork_geqrf_h));
        if (h_work == nullptr) {
            throw std::runtime_error("Error: h_work not allocated.");
        }
    }

    CUSOLVER_CHECK(cusolverDnXgeqrf(
        cusolverH, params, m, n, CUDA_R_32F, d_Q, m, CUDA_R_32F, d_tau,
        CUDA_R_32F, d_work, lwork_geqrf_d, h_work, lwork_geqrf_h, d_info));

    if (h_work) {
        free(h_work); // No longer needed
    }

    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (0 > info) {
        throw std::runtime_error(std::to_string(-info) +
                                 "-th parameter is wrong \n");
    }

    copy_upper_triangular(d_R, d_Q, m, n);

    // Explicitly compute Q
    CUSOLVER_CHECK(cusolverDnSorgqr(cusolverH, m, n, n, d_Q, m, d_tau,
                                    reinterpret_cast<float *>(d_work),
                                    numfloats_orgqr_d, d_info));
    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (0 > info) {
        throw std::runtime_error(std::to_string(-info) +
                                 "-th parameter is wrong \n");
    }

    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_tau));
    CUDA_CHECK(cudaFree(d_work));
}

void qr_factorization(cusolverDnHandle_t &cusolverH, cusolverDnParams_t &params,
                      double *d_Q, double *d_R, const int m, const int n,
                      const double *d_A) {
    NVTX3_FUNC_RANGE();

    assert(n < m && "Expect cols to be less than rows for DR-BCG");

    int info = 0;

    double *d_tau = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tau, sizeof(double) * n));

    int *d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    void *d_work = nullptr;
    std::size_t lwork_geqrf_d = 0;

    void *h_work = nullptr;
    std::size_t lwork_geqrf_h = 0;

    CUDA_CHECK(
        cudaMemcpy(d_Q, d_A, sizeof(double) * m * n, cudaMemcpyDeviceToDevice));

    // Create device buffer
    CUSOLVER_CHECK(cusolverDnXgeqrf_bufferSize(
        cusolverH, params, m, n, CUDA_R_64F, d_Q, m, CUDA_R_64F, d_tau,
        CUDA_R_64F, &lwork_geqrf_d, &lwork_geqrf_h));

    int numdoubles_orgqr_d = 0;
    CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(cusolverH, m, n, n, d_Q, m,
                                               d_tau, &numdoubles_orgqr_d));
    const std::size_t lwork_orgqr_d = numdoubles_orgqr_d * sizeof(double);

    const std::size_t lwork_bytes_d = std::max(lwork_geqrf_d, lwork_orgqr_d);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), lwork_bytes_d));

    if (lwork_geqrf_h > 0) {
        h_work = reinterpret_cast<void *>(malloc(lwork_geqrf_h));
        if (h_work == nullptr) {
            throw std::runtime_error("Error: h_work not allocated.");
        }
    }

    CUSOLVER_CHECK(cusolverDnXgeqrf(
        cusolverH, params, m, n, CUDA_R_64F, d_Q, m, CUDA_R_64F, d_tau,
        CUDA_R_64F, d_work, lwork_geqrf_d, h_work, lwork_geqrf_h, d_info));

    if (h_work) {
        free(h_work); // No longer needed
    }

    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (0 > info) {
        throw std::runtime_error(std::to_string(-info) +
                                 "-th parameter is wrong \n");
    }

    copy_upper_triangular(d_R, d_Q, m, n);

    // Explicitly compute Q
    CUSOLVER_CHECK(cusolverDnDorgqr(cusolverH, m, n, n, d_Q, m, d_tau,
                                    reinterpret_cast<double *>(d_work),
                                    numdoubles_orgqr_d, d_info));
    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (0 > info) {
        throw std::runtime_error(std::to_string(-info) +
                                 "-th parameter is wrong \n");
    }

    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_tau));
    CUDA_CHECK(cudaFree(d_work));
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

namespace {
template <typename T>
void sptri_left_multiply(const cusparseHandle_t &cusparseH,
                         cusparseDnMatDescr_t &C, cusparseOperation_t opA,
                         const cusparseSpMatDescr_t &A,
                         const cusparseDnMatDescr_t &B) {
    constexpr cusparseOperation_t OP_B = CUSPARSE_OPERATION_NON_TRANSPOSE;
    constexpr cudaDataType_t compute_type = [] {
        if constexpr (std::is_same_v<T, float>) {
            return CUDA_R_32F;
        } else {
            return CUDA_R_64F;
        }
    }();
    constexpr T alpha = 1;
    constexpr cusparseSpSMAlg_t ALG_TYPE = CUSPARSE_SPSM_ALG_DEFAULT;

    cusparseSpSMDescr_t spsm{};
    CUSPARSE_CHECK(cusparseSpSM_createDescr(&spsm));

    void *buffer = nullptr;
    size_t buffer_size = 0;

    CUSPARSE_CHECK(cusparseSpSM_bufferSize(
        cusparseH, opA, OP_B, reinterpret_cast<const void *>(&alpha), A, B, C,
        compute_type, ALG_TYPE, spsm, &buffer_size));

    if (buffer_size > 0) {
        CUDA_CHECK(cudaMalloc(&buffer, buffer_size));
    } else {
        throw std::runtime_error("s solve: buffer not allocated");
    }

    CUSPARSE_CHECK(cusparseSpSM_analysis(
        cusparseH, opA, OP_B, reinterpret_cast<const void *>(&alpha), A, B, C,
        compute_type, ALG_TYPE, spsm, buffer));

    CUSPARSE_CHECK(cusparseSpSM_solve(cusparseH, opA, OP_B,
                                      reinterpret_cast<const void *>(&alpha), A,
                                      B, C, compute_type, ALG_TYPE, spsm));

    CUDA_CHECK(cudaFree(buffer));
    CUSPARSE_CHECK(cusparseSpSM_destroyDescr(spsm));
}
} // namespace

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