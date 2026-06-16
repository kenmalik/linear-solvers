#include "config.h"

#include "cuda_adapter.h"

#include <iostream>

#include "common/cuda_checks.h"
#include "common/device_sparse_matrix.h"

#ifdef SOLVERS_BUILD_CG
#include "cg/cuda.h"
#endif

#ifdef SOLVERS_BUILD_DR_BCG
#include "dr_bcg/cuda.cuh"
#ifdef SOLVERS_BUILD_MATHDX
#include "dr_bcg/mathdx_solve.cuh"
#endif
#endif

namespace {

void configure_cublas_math_mode(cublasHandle_t cublas,
                                bool disable_tensor_cores) {
    if (disable_tensor_cores) {
        CUBLAS_CHECK(cublasSetMathMode(cublas, CUBLAS_PEDANTIC_MATH));
    }
}

const char *qr_backend_name(QrBackend qr_backend) {
    switch (qr_backend) {
    case QrBackend::Householder:
        return "householder";
    case QrBackend::CholQR:
        return "cholqr";
    case QrBackend::CholQRDx:
        return "cholqr-dx";
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

    double *d_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(double) * b.size()));
    CUDA_CHECK(cudaMemcpyAsync(d_b, b.data(), sizeof(double) * b.size(),
                               cudaMemcpyHostToDevice, stream));

    cusparseDnVecDescr_t b_descr;
    CUSPARSE_CHECK(cusparseCreateDnVec(&b_descr, b.size(), d_b, CUDA_R_64F));

    double *d_x = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, sizeof(double) * x.size()));
    CUDA_CHECK(cudaMemcpyAsync(d_x, x.data(), sizeof(double) * x.size(),
                               cudaMemcpyHostToDevice, stream));

    cusparseDnVecDescr_t x_descr;
    CUSPARSE_CHECK(cusparseCreateDnVec(&x_descr, x.size(), d_x, CUDA_R_64F));

    DeviceSparseMatrixDouble A_mat{A};
    DeviceSparseMatrixDouble L_mat{L};

    CUDA_CHECK(cudaDeviceSynchronize());

    int iters = cg::cuda::solve(cusparse, cublas, A_mat.get(), b_descr, x_descr,
                                L_mat.get(), tolerance, max_iterations, true,
                                stream);

    CUDA_CHECK(cudaMemcpyAsync(x.data(), d_x, sizeof(double) * x.size(),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUSPARSE_CHECK(cusparseDestroyDnVec(x_descr));
    CUDA_CHECK(cudaFree(d_x));

    CUSPARSE_CHECK(cusparseDestroyDnVec(b_descr));
    CUDA_CHECK(cudaFree(d_b));

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
                    QrBackend qr_backend) {
    auto n = A.rows();

    dr_bcg::cuda::Handles handles;
    configure_cublas_math_mode(handles.cublas, disable_tensor_cores);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    double *d_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(double) * b.size()));
    CUDA_CHECK(cudaMemcpyAsync(d_b, b.data(), sizeof(double) * b.size(),
                               cudaMemcpyHostToDevice, stream));

    cusparseDnMatDescr_t b_descr;
    CUSPARSE_CHECK(cusparseCreateDnMat(&b_descr, n, block_size, n, d_b,
                                       CUDA_R_64F, CUSPARSE_ORDER_COL));

    double *d_x = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, sizeof(double) * x.size()));
    CUDA_CHECK(cudaMemcpyAsync(d_x, x.data(), sizeof(double) * x.size(),
                               cudaMemcpyHostToDevice, stream));

    cusparseDnMatDescr_t x_descr;
    CUSPARSE_CHECK(cusparseCreateDnMat(&x_descr, n, block_size, n, d_x,
                                       CUDA_R_64F, CUSPARSE_ORDER_COL));

    DeviceSparseMatrixDouble A_mat{A};
    DeviceSparseMatrixDouble L_mat{L};

    CUDA_CHECK(cudaDeviceSynchronize());

    int iters = -1;
    try {
        if (qr_backend == QrBackend::CholQR) {
            iters = dr_bcg::cuda::solve<double, CholeskyQr<double>>(handles, A_mat.get(), x_descr, b_descr,
                                                                    L_mat.get(), tolerance, max_iterations,
                                                                    stream);
        } else if (qr_backend == QrBackend::CholQRDx) {
#ifdef SOLVERS_BUILD_MATHDX
            iters = dr_bcg::cuda::solve_cholqr_dx(handles, A_mat.get(), x_descr, b_descr,
                                                  L_mat.get(), tolerance, max_iterations,
                                                  stream);
#else
            throw std::runtime_error("QR backend 'cholqr-dx' requires building with SOLVERS_BUILD_MATHDX=ON");
#endif
        } else {
            iters = dr_bcg::cuda::solve<double>(handles, A_mat.get(), x_descr, b_descr,
                                                L_mat.get(), tolerance, max_iterations,
                                                stream);
        }
        CUDA_CHECK(cudaMemcpyAsync(x.data(), d_x, sizeof(double) * x.size(),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } catch (const std::exception &e) {
        std::cerr << "CUDA DR-BCG failed with QR backend '"
                  << qr_backend_name(qr_backend) << "': " << e.what()
                  << std::endl;
    }

    CUSPARSE_CHECK(cusparseDestroyDnMat(x_descr));
    CUDA_CHECK(cudaFree(d_x));

    CUSPARSE_CHECK(cusparseDestroyDnMat(b_descr));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return iters;
}

int run_cuda_dr_bcg(const mat_utils::SpMatReader &A,
                    const std::vector<double> &b, std::vector<double> &x,
                    double tolerance, int max_iterations, int block_size,
                    bool disable_tensor_cores,
                    QrBackend qr_backend) {
    auto n = A.rows();

    dr_bcg::cuda::Handles handles;
    configure_cublas_math_mode(handles.cublas, disable_tensor_cores);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    double *d_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(double) * b.size()));
    CUDA_CHECK(cudaMemcpyAsync(d_b, b.data(), sizeof(double) * b.size(),
                               cudaMemcpyHostToDevice, stream));

    cusparseDnMatDescr_t b_descr;
    CUSPARSE_CHECK(cusparseCreateDnMat(&b_descr, n, block_size, n, d_b,
                                       CUDA_R_64F, CUSPARSE_ORDER_COL));

    double *d_x = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, sizeof(double) * x.size()));
    CUDA_CHECK(cudaMemcpyAsync(d_x, x.data(), sizeof(double) * x.size(),
                               cudaMemcpyHostToDevice, stream));

    cusparseDnMatDescr_t x_descr;
    CUSPARSE_CHECK(cusparseCreateDnMat(&x_descr, n, block_size, n, d_x,
                                       CUDA_R_64F, CUSPARSE_ORDER_COL));

    DeviceSparseMatrixDouble A_mat{A};

    CUDA_CHECK(cudaDeviceSynchronize());

    int iters = -1;
    try {
        if (qr_backend == QrBackend::CholQR) {
            iters = dr_bcg::cuda::solve<double, CholeskyQr<double>>(handles, A_mat.get(), x_descr, b_descr,
                                                                    tolerance, max_iterations, stream);
        } else if (qr_backend == QrBackend::CholQRDx) {
#ifdef SOLVERS_BUILD_MATHDX
            iters = dr_bcg::cuda::solve_cholqr_dx(handles, A_mat.get(), x_descr, b_descr,
                                                  tolerance, max_iterations, stream);
#else
            throw std::runtime_error("QR backend 'cholqr-dx' requires building with SOLVERS_BUILD_MATHDX=ON");
#endif
        } else {
            iters = dr_bcg::cuda::solve<double>(handles, A_mat.get(), x_descr, b_descr,
                                                tolerance, max_iterations, stream);
        }
        CUDA_CHECK(cudaMemcpyAsync(x.data(), d_x, sizeof(double) * x.size(),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } catch (const std::exception &e) {
        std::cerr << "CUDA DR-BCG failed with QR backend '"
                  << qr_backend_name(qr_backend) << "': " << e.what()
                  << std::endl;
    }

    CUSPARSE_CHECK(cusparseDestroyDnMat(x_descr));
    CUDA_CHECK(cudaFree(d_x));

    CUSPARSE_CHECK(cusparseDestroyDnMat(b_descr));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return iters;
}

#endif // SOLVERS_BUILD_DR_BCG
