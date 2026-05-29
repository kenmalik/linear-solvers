#include "config.h"

#include "cuda_adapter.h"

#include <iostream>

#include "common/cuda_checks.h"
#include "common/device_sparse_matrix.h"

#ifdef SOLVERS_BUILD_CG
#include "cg/cuda.h"
#endif

#ifdef SOLVERS_BUILD_DR_BCG
#include "dr_bcg/cuda.h"
#endif

namespace {

void configure_cublas_math_mode(cublasHandle_t cublas,
                                bool disable_tensor_cores) {
    if (disable_tensor_cores) {
        CUBLAS_CHECK(cublasSetMathMode(cublas, CUBLAS_PEDANTIC_MATH));
    }
}

const char *qr_backend_name(dr_bcg::cuda::QrBackend qr_backend) {
    switch (qr_backend) {
    case dr_bcg::cuda::QrBackend::Householder:
        return "householder";
    case dr_bcg::cuda::QrBackend::CholQR:
        return "cholqr";
    default:
        return "unknown";
    }
}

} // namespace

#ifdef SOLVERS_BUILD_CG

int run_cuda_cg(const mat_utils::SpMatReader &A, const std::vector<double> &b,
                std::vector<double> &x, const mat_utils::SpMatReader &L,
                double tolerance, int max_iterations,
                bool disable_tensor_cores) {
    cusparseHandle_t cusparse;
    CUSPARSE_CHECK(cusparseCreate(&cusparse));

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate_v2(&cublas));
    configure_cublas_math_mode(cublas, disable_tensor_cores);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUSPARSE_CHECK(cusparseSetStream(cusparse, stream));
    CUBLAS_CHECK(cublasSetStream_v2(cublas, stream));

    double *b_d = nullptr;
    CUDA_CHECK(cudaMalloc(&b_d, sizeof(double) * b.size()));
    CUDA_CHECK(cudaMemcpyAsync(b_d, b.data(), sizeof(double) * b.size(),
                               cudaMemcpyHostToDevice, stream));

    cusparseDnVecDescr_t b_descr;
    CUSPARSE_CHECK(cusparseCreateDnVec(&b_descr, b.size(), b_d, CUDA_R_64F));

    double *x_d = nullptr;
    CUDA_CHECK(cudaMalloc(&x_d, sizeof(double) * x.size()));
    CUDA_CHECK(cudaMemcpyAsync(x_d, x.data(), sizeof(double) * x.size(),
                               cudaMemcpyHostToDevice, stream));

    cusparseDnVecDescr_t x_descr;
    CUSPARSE_CHECK(cusparseCreateDnVec(&x_descr, x.size(), x_d, CUDA_R_64F));

    DeviceSparseMatrixDouble A_mat{A};
    DeviceSparseMatrixDouble L_mat{L};

    CUDA_CHECK(cudaDeviceSynchronize());

    int iters = cg::cuda::solve(cusparse, cublas, A_mat.get(), b_descr, x_descr,
                                L_mat.get(), tolerance, max_iterations, true,
                                stream);

    CUDA_CHECK(cudaMemcpyAsync(x.data(), x_d, sizeof(double) * x.size(),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUSPARSE_CHECK(cusparseDestroyDnVec(x_descr));
    CUDA_CHECK(cudaFree(x_d));

    CUSPARSE_CHECK(cusparseDestroyDnVec(b_descr));
    CUDA_CHECK(cudaFree(b_d));

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUBLAS_CHECK(cublasDestroy_v2(cublas));
    CUSPARSE_CHECK(cusparseDestroy(cusparse));

    return iters;
}

#endif // SOLVERS_BUILD_CG

#ifdef SOLVERS_BUILD_DR_BCG

int run_cuda_dr_bcg(const mat_utils::SpMatReader &A,
                    const std::vector<double> &b, std::vector<double> &x,
                    const mat_utils::SpMatReader &L, double tolerance,
                    int max_iterations, int block_size,
                    bool disable_tensor_cores,
                    dr_bcg::cuda::QrBackend qr_backend) {
    auto n = A.rows();

    dr_bcg::cuda::Handles handles;
    configure_cublas_math_mode(handles.cublas, disable_tensor_cores);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    double *b_d = nullptr;
    CUDA_CHECK(cudaMalloc(&b_d, sizeof(double) * b.size()));
    CUDA_CHECK(cudaMemcpyAsync(b_d, b.data(), sizeof(double) * b.size(),
                               cudaMemcpyHostToDevice, stream));

    cusparseDnMatDescr_t b_descr;
    CUSPARSE_CHECK(cusparseCreateDnMat(&b_descr, n, block_size, n, b_d,
                                       CUDA_R_64F, CUSPARSE_ORDER_COL));

    double *x_d = nullptr;
    CUDA_CHECK(cudaMalloc(&x_d, sizeof(double) * x.size()));
    CUDA_CHECK(cudaMemcpyAsync(x_d, x.data(), sizeof(double) * x.size(),
                               cudaMemcpyHostToDevice, stream));

    cusparseDnMatDescr_t x_descr;
    CUSPARSE_CHECK(cusparseCreateDnMat(&x_descr, n, block_size, n, x_d,
                                       CUDA_R_64F, CUSPARSE_ORDER_COL));

    DeviceSparseMatrixDouble A_mat{A};
    DeviceSparseMatrixDouble L_mat{L};

    CUDA_CHECK(cudaDeviceSynchronize());

    int iters = -1;
    try {
        iters = dr_bcg::cuda::solve(handles, A_mat.get(), x_descr, b_descr,
                                    L_mat.get(), tolerance, max_iterations,
                                    stream, qr_backend);
        CUDA_CHECK(cudaMemcpyAsync(x.data(), x_d, sizeof(double) * x.size(),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } catch (const std::exception &e) {
        std::cerr << "CUDA DR-BCG failed with QR backend '"
                  << qr_backend_name(qr_backend) << "': " << e.what()
                  << std::endl;
    }

    CUSPARSE_CHECK(cusparseDestroyDnMat(x_descr));
    CUDA_CHECK(cudaFree(x_d));

    CUSPARSE_CHECK(cusparseDestroyDnMat(b_descr));
    CUDA_CHECK(cudaFree(b_d));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return iters;
}

int run_cuda_dr_bcg(const mat_utils::SpMatReader &A,
                    const std::vector<double> &b, std::vector<double> &x,
                    double tolerance, int max_iterations, int block_size,
                    bool disable_tensor_cores,
                    dr_bcg::cuda::QrBackend qr_backend) {
    auto n = A.rows();

    dr_bcg::cuda::Handles handles;
    configure_cublas_math_mode(handles.cublas, disable_tensor_cores);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    double *b_d = nullptr;
    CUDA_CHECK(cudaMalloc(&b_d, sizeof(double) * b.size()));
    CUDA_CHECK(cudaMemcpyAsync(b_d, b.data(), sizeof(double) * b.size(),
                               cudaMemcpyHostToDevice, stream));

    cusparseDnMatDescr_t b_descr;
    CUSPARSE_CHECK(cusparseCreateDnMat(&b_descr, n, block_size, n, b_d,
                                       CUDA_R_64F, CUSPARSE_ORDER_COL));

    double *x_d = nullptr;
    CUDA_CHECK(cudaMalloc(&x_d, sizeof(double) * x.size()));
    CUDA_CHECK(cudaMemcpyAsync(x_d, x.data(), sizeof(double) * x.size(),
                               cudaMemcpyHostToDevice, stream));

    cusparseDnMatDescr_t x_descr;
    CUSPARSE_CHECK(cusparseCreateDnMat(&x_descr, n, block_size, n, x_d,
                                       CUDA_R_64F, CUSPARSE_ORDER_COL));

    DeviceSparseMatrixDouble A_mat{A};

    CUDA_CHECK(cudaDeviceSynchronize());

    int iters = -1;
    try {
        iters = dr_bcg::cuda::solve(handles, A_mat.get(), x_descr, b_descr,
                                    tolerance, max_iterations, stream,
                                    qr_backend);
        CUDA_CHECK(cudaMemcpyAsync(x.data(), x_d, sizeof(double) * x.size(),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } catch (const std::exception &e) {
        std::cerr << "CUDA DR-BCG failed with QR backend '"
                  << qr_backend_name(qr_backend) << "': " << e.what()
                  << std::endl;
    }

    CUSPARSE_CHECK(cusparseDestroyDnMat(x_descr));
    CUDA_CHECK(cudaFree(x_d));

    CUSPARSE_CHECK(cusparseDestroyDnMat(b_descr));
    CUDA_CHECK(cudaFree(b_d));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return iters;
}

#endif // SOLVERS_BUILD_DR_BCG
