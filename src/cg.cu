#include <cg_run/cg.h>
#include <cstdint>
#include <iostream>
#include <type_traits>

namespace {
template <typename T> struct Device_buffers {
    cudaDataType_t cuda_type =
        std::is_same_v<T, float> ? CUDA_R_32F : CUDA_R_64F;

    Device_buffers(std::int64_t n) : n(n) {
        cudaMalloc(&r_d, sizeof(T) * n);
        cudaMalloc(&z_d, sizeof(T) * n);
        cudaMalloc(&p_d, sizeof(T) * n);
        cudaMalloc(&q_d, sizeof(T) * n);
        cudaMalloc(&t_d, sizeof(T) * n);

        cusparseCreateDnVec(&r, n, r_d, cuda_type);
        cusparseCreateDnVec(&z, n, z_d, cuda_type);
        cusparseCreateDnVec(&p, n, p_d, cuda_type);
        cusparseCreateDnVec(&q, n, q_d, cuda_type);
        cusparseCreateDnVec(&t, n, t_d, cuda_type);
    }

    ~Device_buffers() {
        cudaFree(r_d);
        cudaFree(z_d);
        cudaFree(p_d);
        cudaFree(q_d);
        cudaFree(t_d);

        cusparseDestroyDnVec(r);
        cusparseDestroyDnVec(z);
        cusparseDestroyDnVec(p);
        cusparseDestroyDnVec(q);
        cusparseDestroyDnVec(t);
    }

    std::int64_t n;
    cusparseDnVecDescr_t r, z, p, q, t;
    T *r_d, *z_d, *p_d, *q_d, *t_d;
};
} // namespace

namespace cg_run {
// R is lower triangular incomplete Cholesky where A ~= M = R^TR
int cg(cusparseHandle_t cusparse, cublasHandle_t cublas, cusparseSpMatDescr_t A,
       cusparseDnVecDescr_t x, cusparseDnVecDescr_t f, cusparseSpMatDescr_t R,
       double tolerance, int max_iterations) {
    std::int64_t n;
    double *x_d = nullptr;
    cudaDataType_t compute_type;
    cusparseDnVecGet(x, &n, reinterpret_cast<void **>(&x_d), &compute_type);

    double *f_d = nullptr;
    cusparseDnVecGetValues(f, reinterpret_cast<void **>(&f_d));

    // Allocate buffer for SpSV operations
    cusparseSpSVDescr_t descr_SV_R, descr_SV_Rt;
    cusparseSpSV_createDescr(&descr_SV_R);
    cusparseSpSV_createDescr(&descr_SV_Rt);

    // Analysis phase for R (lower triangular, non-transpose)
    constexpr double alpha_SpSV = 1.0;
    std::size_t bufferSizeR = 0;
    cusparseSpSV_bufferSize(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_SpSV, R, f, f,
        compute_type, CUSPARSE_SPSV_ALG_DEFAULT, descr_SV_R, &bufferSizeR);

    // Analysis phase for R^T (transpose)
    std::size_t bufferSizeRt = 0;
    cusparseSpSV_bufferSize(cusparse, CUSPARSE_OPERATION_TRANSPOSE, &alpha_SpSV,
                            R, f, f, compute_type, CUSPARSE_SPSV_ALG_DEFAULT,
                            descr_SV_Rt, &bufferSizeRt);

    std::size_t bufferSizeSV = std::max(bufferSizeR, bufferSizeRt);
    void *buffer_SV = nullptr;
    cudaMalloc(&buffer_SV, bufferSizeSV);

    // Analyze R for non-transpose solve
    cusparseSpSV_analysis(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          &alpha_SpSV, R, f, f, compute_type,
                          CUSPARSE_SPSV_ALG_DEFAULT, descr_SV_R, buffer_SV);

    // Analyze R for transpose solve
    cusparseSpSV_analysis(cusparse, CUSPARSE_OPERATION_TRANSPOSE, &alpha_SpSV,
                          R, f, f, compute_type, CUSPARSE_SPSV_ALG_DEFAULT,
                          descr_SV_Rt, buffer_SV);

    Device_buffers<double> d{n};

    // r = f - A x_0
    constexpr double alpha_residual = -1.0;
    constexpr double beta_residual = 1.0;

    cudaMemcpy(d.r_d, f_d, sizeof(double) * n, cudaMemcpyDeviceToDevice);

    std::size_t buffer_size = 0;
    cusparseSpMV_bufferSize(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha_residual, A, x, &beta_residual, d.r,
                            compute_type, CUSPARSE_SPMV_ALG_DEFAULT,
                            &buffer_size);

    void *buffer = nullptr;
    cudaMalloc(&buffer, buffer_size);

    cusparseSpMV(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_residual, A,
                 x, &beta_residual, d.r, compute_type,
                 CUSPARSE_SPMV_ALG_DEFAULT, buffer);

    cudaFree(buffer);

    double norm_r_0 = 0;
    cublasDnrm2_v2(cublas, n, d.r_d, 1, &norm_r_0);

    // Allocate buffer for SpMV (A*p operation)
    constexpr double alpha_SpMV = 1.0;
    constexpr double beta_SpMV = 0.0;
    std::size_t buffer_size_SpMV = 0;
    cusparseSpMV_bufferSize(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha_SpMV, A, d.p, &beta_SpMV, d.q, compute_type,
                            CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size_SpMV);
    void *buffer_MV = nullptr;
    cudaMalloc(&buffer_MV, buffer_size_SpMV);

    double rho = 0.0, rhop = 0.0, alpha, beta, temp;

    // CG iteration loop
    int iteration;
    for (iteration = 0; iteration < max_iterations; ++iteration) {
        // Solve M z = r, where M = R R^T
        // First solve: R t = r (lower triangular)
        cusparseSpSV_solve(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           &alpha_SpSV, R, d.r, d.t, compute_type,
                           CUSPARSE_SPSV_ALG_DEFAULT, descr_SV_R);

        // Second solve: R^T z = t (upper triangular)
        cusparseSpSV_solve(cusparse, CUSPARSE_OPERATION_TRANSPOSE, &alpha_SpSV,
                           R, d.t, d.z, compute_type, CUSPARSE_SPSV_ALG_DEFAULT,
                           descr_SV_Rt);

        // rho = r^T z
        rhop = rho;
        cublasDdot_v2(cublas, n, d.r_d, 1, d.z_d, 1, &rho);

        if (iteration == 0) {
            // p = z
            cublasDcopy_v2(cublas, n, d.z_d, 1, d.p_d, 1);
        } else {
            // beta = rho_i / rho_{i-1}
            beta = rho / rhop;

            // p = z + beta * p
            cublasDaxpy_v2(cublas, n, &beta, d.p_d, 1, d.z_d, 1);
            cublasDcopy_v2(cublas, n, d.z_d, 1, d.p_d, 1);
        }

        // q = A p
        cusparseSpMV(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_SpMV, A,
                     d.p, &beta_SpMV, d.q, compute_type,
                     CUSPARSE_SPMV_ALG_DEFAULT, buffer_MV);

        // alpha = rho_i / (p^T q)
        cublasDdot_v2(cublas, n, d.p_d, 1, d.q_d, 1, &temp);
        alpha = rho / temp;

        // x = x + alpha * p
        cublasDaxpy_v2(cublas, n, &alpha, d.p_d, 1, x_d, 1);

        // r = r - alpha * q
        double neg_alpha = -alpha;
        cublasDaxpy_v2(cublas, n, &neg_alpha, d.q_d, 1, d.r_d, 1);

        // Check for convergence
        double norm_r;
        cublasDnrm2_v2(cublas, n, d.r_d, 1, &norm_r);

        if (norm_r / norm_r_0 < tolerance) {
            break;
        }
    }

    // Cleanup
    cudaFree(buffer_SV);
    cudaFree(buffer_MV);
    cusparseSpSV_destroyDescr(descr_SV_R);
    cusparseSpSV_destroyDescr(descr_SV_Rt);

    return iteration + 1; // Return number of iterations performed
}
} // namespace cg_run