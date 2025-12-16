#pragma once

#include <optional>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusparse_v2.h>

#define CUDA_CHECK(val) check((val), #val, __FILE__, __LINE__);
#define CUSOLVER_CHECK(val) check((val), #val, __FILE__, __LINE__);
#define CUBLAS_CHECK(val) check((val), #val, __FILE__, __LINE__);
#define CUSPARSE_CHECK(val) check((val), #val, __FILE__, __LINE__);

void check(cudaError_t err, const char *const func, const char *const file,
           const int line);
void check(cusolverStatus_t err, const char *const func, const char *const file,
           const int line);
void check(cublasStatus_t err, const char *const func, const char *const file,
           const int line);
void check(cusparseStatus_t err, const char *const func, const char *const file,
           const int line);

void print_matrix(const float *mat, const int rows, const int cols);

void print_device_matrix(const float *d_mat, const int rows, const int cols);

void print_sparse_matrix(const cusparseHandle_t &cusparseH,
                         const cusparseSpMatDescr_t &sp_mat);

void copy_upper_triangular(float *dst, float *src, const int m, const int n);

void copy_upper_triangular(double *dst, double *src, const int m, const int n);

void invert_square_matrix(cusolverDnHandle_t &cusolverH,
                          cusolverDnParams_t &params, float *A, const int n);
void invert_square_matrix(cusolverDnHandle_t &cusolverH,
                          cusolverDnParams_t &params, double *A, const int n);

void qr_factorization(cusolverDnHandle_t &cusolverH, cusolverDnParams_t &params,
                      float *Q, float *R, const int m, const int n,
                      const float *A);

void qr_factorization(cusolverDnHandle_t &cusolverH, cusolverDnParams_t &params,
                      double *Q, double *R, const int m, const int n,
                      const double *A);

void sptri_left_multiply(const cusparseHandle_t &cusparseH,
                         cusparseDnMatDescr_t &C, cusparseOperation_t opA,
                         const cusparseSpMatDescr_t &A,
                         const cusparseDnMatDescr_t &B,
                         const cudaDataType compute_type = CUDA_R_32F);