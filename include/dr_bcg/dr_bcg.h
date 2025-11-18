#pragma once

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusparse_v2.h>

#include "dr_bcg/device_buffer.h"
#include "dr_bcg/helper.h"

namespace dr_bcg {
cusolverStatus_t dr_bcg(cusolverDnHandle_t cusolverH,
                        cusolverDnParams_t cusolverParams,
                        cublasHandle_t cublasH, cusparseHandle_t cusparseH,
                        cusparseSpMatDescr_t &A, cusparseDnMatDescr_t &X,
                        cusparseDnMatDescr_t &B, cusparseSpMatDescr_t &L,
                        float tolerance, int max_iterations,
                        int *iterations = nullptr);

void get_xi(cusolverDnHandle_t &cusolverH, cusolverDnParams_t &cusolverParams,
            cublasHandle_t &cublasH, const int m, const int n, DeviceBuffer &d,
            const float *d_A);

void get_xi(cublasHandle_t &cublasH, cusolverDnHandle_t &cusolverH,
            cusolverDnParams_t &cusolverParams, cusparseHandle_t &cusparseH,
            cusparseSpMatDescr_t &A, const int n, const int s, DeviceBuffer &d);

void get_sigma(cublasHandle_t cublasH, int n, DeviceBuffer &d);

void get_s(cublasHandle_t cublasH, const int m, const int n, DeviceBuffer &d);

void get_s(cusparseHandle_t cusparseH, cublasHandle_t cublasH, const int n,
           const int s, DeviceBuffer &d, cusparseSpMatDescr_t &L);

void get_w_zeta(cusolverDnHandle_t &cusolverH,
                cusolverDnParams_t &cusolverParams, cublasHandle_t &cublasH,
                const int m, const int n, DeviceBuffer &d, const float *d_A);

void get_w_zeta(cusolverDnHandle_t &cusolverH,
                cusolverDnParams_t &cusolverParams, cublasHandle_t &cublasH,
                cusparseHandle_t &cusparseH, const int m, const int n,
                DeviceBuffer &d, cusparseSpMatDescr_t &A);

void get_w_zeta(cusolverDnHandle_t &cusolverH,
                cusolverDnParams_t &cusolverParams, cublasHandle_t &cublasH,
                cusparseHandle_t &cusparseH, const int n, const int s,
                DeviceBuffer &d, cusparseSpMatDescr_t &A,
                cusparseSpMatDescr_t &L);

void residual(cublasHandle_t &cublasH, float *d_residual, const float *B,
              const int m, const float *d_A, const float *d_X);

void residual(cusparseHandle_t &cusparseH, cusparseDnVecDescr_t &residual,
              const float *B, cusparseSpMatDescr_t &A, cusparseDnMatDescr_t &X);

void get_next_X(cublasHandle_t &cublasH, const int m, const int n,
                const float *d_s, const float *d_xi, float *d_temp,
                const float *d_sigma, float *d_X);

void quadratic_form(cublasHandle_t &cublasH, const int m, const int n,
                    const float *d_s, const float *d_A, float *d_work,
                    float *d_y);

void quadratic_form(cublasHandle_t &cublasH, cusparseHandle_t &cusparseH,
                    const int n, const int s, cusparseDnMatDescr_t &X,
                    cusparseSpMatDescr_t &A, float *d_work, float *d_y);

void get_R(cublasHandle_t &cublasH, float *h_R, const int n, const int m,
           const float *A, const float *X, const float *B);

void get_R(cusparseHandle_t &cusparseH, cusparseDnMatDescr_t &R,
           cusparseSpMatDescr_t &A, cusparseDnMatDescr_t &X,
           cusparseDnMatDescr_t &B);
} // namespace dr_bcg
