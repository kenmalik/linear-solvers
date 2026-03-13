#include "cuda_adapter.h"

#include <cg/cuda.h>

int run_cuda(const mat_utils::SpMatReader &A, const std::vector<double> &b,
             std::vector<double> &x, const mat_utils::SpMatReader &L) {
    cusparseHandle_t cusparse;
    cusparseCreate(&cusparse);

    cublasHandle_t cublas;
    cublasCreate_v2(&cublas);

    double *b_d = nullptr;
    cudaMalloc(&b_d, sizeof(double) * b.size());
    cudaMemcpy(b_d, b.data(), sizeof(double) * b.size(), cudaMemcpyHostToDevice);

    cusparseDnVecDescr_t b_descr;
    cusparseCreateDnVec(&b_descr, b.size(), b_d, CUDA_R_64F);

    double *x_d = nullptr;
    cudaMalloc(&x_d, sizeof(double) * x.size());
    cudaMemcpy(x_d, x.data(), sizeof(double) * x.size(), cudaMemcpyHostToDevice);

    cusparseDnVecDescr_t x_descr;
    cusparseCreateDnVec(&x_descr, x.size(), x_d, CUDA_R_64F);

    // int iters = cg::cuda::solve(cusparse, cublas, cusparseSpMatDescr_t A,
    //                             b, x, cusparseSpMatDescr_t L)

    cusparseDestroyDnVec(x_descr);
    cudaFree(x_d);

    cusparseDestroyDnVec(b_descr);
    cudaFree(b_d);

    cublasDestroy_v2(cublas);
    cusparseDestroy(cusparse);

    return 0;
}
