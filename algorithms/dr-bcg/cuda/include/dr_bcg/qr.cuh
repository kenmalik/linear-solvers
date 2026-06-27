#pragma once

#include "common/cuda_checks.h"
#include "common/cuda_event_timer.h"
#include "common/type_info.h"

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <concepts>

template <typename P, typename T>
concept QrPolicy = requires(P &p, T *&d_Q, T *&d_R, const T *d_A, const int &m, const int &n,
                            cublasHandle_t &cublasH, cusolverDnHandle_t &cusolverH,
                            cusolverDnParams_t &params, cudaStream_t &stream, const char *stage) {
    P{cusolverH, params, m, n};
    { p.solve(d_Q, d_R, d_A, m, n, cublasH, cusolverH, params, stream) } -> std::same_as<void>;
    { p.check(n, stage, stream) } -> std::same_as<void>;
};

template <SupportedType T>
__global__ void copy_upper_triangular_kernel(T *dst, const T *src,
                                             const int ld_src, const int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        dst[row + col * n] = (row <= col) ? src[row + col * ld_src] : 0.0;
    }
}

template <SupportedType T>
void copy_upper_triangular(T *dst, const T *src, int ld_src, int n,
                           cudaStream_t stream) {
    constexpr int block_n = 16;
    constexpr dim3 block_dim(block_n, block_n);
    dim3 grid_dim((n + block_n - 1) / block_n, (n + block_n - 1) / block_n);
    copy_upper_triangular_kernel<<<grid_dim, block_dim, 0, stream>>>(
        dst, src, ld_src, n);
}

template <SupportedType T>
class HouseholderQr {
  public:
    HouseholderQr(const HouseholderQr &) = delete;
    HouseholderQr &operator=(const HouseholderQr &) = delete;

    // m: rows of Q (problem size n), n: cols of Q (block size s)
    HouseholderQr(cusolverDnHandle_t &cusolverH, cusolverDnParams_t &params,
                  int m, int n) {
        CUDA_CHECK(cudaMalloc(&d_tau, sizeof(T) * n));
        CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&h_info), sizeof(int)));

        // Dummy m×n device buffer needed to query workspace sizes.
        // Buffer size queries are dimension/type-driven; values are not read.
        T *d_dummy = nullptr;
        CUDA_CHECK(cudaMalloc(&d_dummy, sizeof(T) * m * n));

        CUSOLVER_CHECK(cusolverDnXgeqrf_bufferSize(
            cusolverH, params, m, n, cuda_type<T>, d_dummy, m, cuda_type<T>, d_tau,
            cuda_type<T>, &d_lwork_geqrf, &h_lwork_geqrf));

        if constexpr (std::is_same_v<T, float>) {
            CUSOLVER_CHECK(cusolverDnSorgqr_bufferSize(
                cusolverH, m, n, n, d_dummy, m, d_tau, &d_numfloats_orgqr));
        } else {
            CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(
                cusolverH, m, n, n, d_dummy, m, d_tau, &d_numfloats_orgqr));
        }

        CUDA_CHECK(cudaFree(d_dummy));

        const std::size_t d_lwork_orgqr = d_numfloats_orgqr * sizeof(T);
        CUDA_CHECK(cudaMalloc(&d_work, std::max(d_lwork_geqrf, d_lwork_orgqr)));

        if (h_lwork_geqrf > 0) {
            h_work = malloc(h_lwork_geqrf);
            if (!h_work)
                throw std::runtime_error("Error: HouseholderQr h_work not allocated.");
        }
    }

    ~HouseholderQr() {
        if (d_tau)
            CUDA_CHECK(cudaFree(d_tau));
        if (d_work)
            CUDA_CHECK(cudaFree(d_work));
        if (d_info)
            CUDA_CHECK(cudaFree(d_info));
        if (h_info)
            CUDA_CHECK(cudaFreeHost(h_info));
        if (h_work)
            free(h_work);
    }

    void solve(T *&d_Q, T *&d_R, const T *d_A, const int &m, const int &n,
               cublasHandle_t &cublasH, cusolverDnHandle_t &cusolverH,
               cusolverDnParams_t &params, cudaStream_t &stream) {
        assert(n < m && "Expect cols to be less than rows for DR-BCG");

        CudaTimerRange rng{g_event_timer, "QR:func", stream};

        CUDA_CHECK(cudaMemcpyAsync(d_Q, d_A, sizeof(T) * m * n,
                                   cudaMemcpyDeviceToDevice, stream));

        {
            CudaTimerRange rng{g_event_timer, "QR:geqrf", stream};
            CUSOLVER_CHECK(cusolverDnXgeqrf(
                cusolverH, params, m, n, cuda_type<T>, d_Q, m, cuda_type<T>,
                d_tau, cuda_type<T>, d_work, d_lwork_geqrf, h_work,
                h_lwork_geqrf, d_info));
        }

        {
            CudaTimerRange rng{g_event_timer, "QR:copy_upper_triangular", stream};
            copy_upper_triangular(d_R, d_Q, m, n, stream);
        }

        if constexpr (std::is_same_v<T, float>) {
            CudaTimerRange rng{g_event_timer, "QR:orgqr", stream};
            CUSOLVER_CHECK(cusolverDnSorgqr(
                cusolverH, m, n, n, d_Q, m, d_tau,
                reinterpret_cast<T *>(d_work), d_numfloats_orgqr, d_info));
        } else {
            CudaTimerRange rng{g_event_timer, "QR:orgqr", stream};
            CUSOLVER_CHECK(cusolverDnDorgqr(
                cusolverH, m, n, n, d_Q, m, d_tau,
                reinterpret_cast<T *>(d_work), d_numfloats_orgqr, d_info));
        }

        CUDA_CHECK(cudaMemcpyAsync(h_info, d_info, sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
    }

    void check(int n, const char *stage, cudaStream_t stream) {
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (*h_info < 0) {
            throw std::runtime_error(std::string(stage) + ": " +
                                     std::to_string(-*h_info) +
                                     "-th parameter is wrong in QR");
        }
    }

  private:
    T *d_tau = nullptr;
    void *d_work = nullptr;
    int *d_info = nullptr;
    int *h_info = nullptr;
    void *h_work = nullptr;

    std::size_t d_lwork_geqrf = 0;
    std::size_t h_lwork_geqrf = 0;
    int d_numfloats_orgqr = 0;
};

template <SupportedType T>
class CholeskyQr {
  public:
    CholeskyQr(const CholeskyQr &) = delete;
    CholeskyQr &operator=(const CholeskyQr &) = delete;

    CholeskyQr(cusolverDnHandle_t &cusolverH, cusolverDnParams_t &params,
               int _, int n) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_gram),
                              sizeof(T) * n * n));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&h_info), sizeof(int)));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&h_factor),
                                  sizeof(T) * n * n));

        T *d_dummy = nullptr;
        CUDA_CHECK(cudaMalloc(&d_dummy, sizeof(T) * n * n));
        CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(
            cusolverH, params, CUBLAS_FILL_MODE_UPPER, n, cuda_type<T>, d_dummy,
            n, cuda_type<T>, &d_work_size, &h_work_size));
        CUDA_CHECK(cudaFree(d_dummy));

        if (d_work_size > 0) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), d_work_size));
        }
        if (h_work_size > 0) {
            h_work = malloc(h_work_size);
            if (!h_work)
                throw std::runtime_error("Error: CholeskyQr h_work not allocated.");
        }
    }

    ~CholeskyQr() {
        if (d_gram)
            CUDA_CHECK(cudaFree(d_gram));
        if (d_info)
            CUDA_CHECK(cudaFree(d_info));
        if (h_info)
            CUDA_CHECK(cudaFreeHost(h_info));
        if (d_work)
            CUDA_CHECK(cudaFree(d_work));
        if (h_factor)
            CUDA_CHECK(cudaFreeHost(h_factor));
        if (h_work)
            free(h_work);
    }

    void solve(T *&d_Q, T *&d_R, const T *d_A, const int &m, const int &n,
               cublasHandle_t &cublasH, cusolverDnHandle_t &cusolverH,
               cusolverDnParams_t &params, cudaStream_t &stream) {
        assert(n < m && "Expect cols to be less than rows for DR-BCG");

        CudaTimerRange rng{g_event_timer, "QR:func", stream};

        constexpr T alpha = 1;
        constexpr T beta = 0;
        CUDA_CHECK(cudaMemcpyAsync(d_Q, d_A, sizeof(T) * m * n,
                                   cudaMemcpyDeviceToDevice, stream));
        CUBLAS_CHECK(cublasSetPointerMode(cublasH, CUBLAS_POINTER_MODE_HOST));

        // Gram = A^T * A via GEMM, not SYRK. cuBLAS {S,D}syrk picks a
        // suboptimal kernel for this tall-skinny shape (small n, large
        // k = m, OP_T): measured ~50x slower than the equivalent GEMM and the
        // dominant cost of the whole QR. GEMM is far better tuned; computing
        // the full n*n Gram instead of just the triangle is negligible at
        // these n, and POTRF reads only the upper triangle anyway.
        if constexpr (std::is_same_v<T, float>) {
            CudaTimerRange rng{g_event_timer, "QR:gram", stream};
            CUBLAS_CHECK(cublasSgemm_v2(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, n, n,
                                        m, &alpha, d_A, m, d_A, m, &beta, d_gram,
                                        n));
        } else {
            CudaTimerRange rng{g_event_timer, "QR:gram", stream};
            CUBLAS_CHECK(cublasDgemm_v2(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, n, n,
                                        m, &alpha, d_A, m, d_A, m, &beta, d_gram,
                                        n));
        }

        {
            CudaTimerRange rng{g_event_timer, "QR:potrf", stream};
            CUSOLVER_CHECK(cusolverDnXpotrf(
                cusolverH, params, CUBLAS_FILL_MODE_UPPER, n, cuda_type<T>,
                d_gram, n, cuda_type<T>, d_work, d_work_size, h_work,
                h_work_size, d_info));
        }

        {
            CudaTimerRange rng{g_event_timer, "QR:copy_upper_triangular", stream};
            copy_upper_triangular(d_R, d_gram, n, n, stream);
        }

        if constexpr (std::is_same_v<T, float>) {
            CudaTimerRange rng{g_event_timer, "QR:trsm", stream};
            CUBLAS_CHECK(cublasStrsm_v2(
                cublasH, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha,
                d_gram, n, d_Q, m));
        } else {
            CudaTimerRange rng{g_event_timer, "QR:trsm", stream};
            CUBLAS_CHECK(cublasDtrsm_v2(
                cublasH, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha,
                d_gram, n, d_Q, m));
        }

        CUDA_CHECK(cudaMemcpyAsync(h_info, d_info, sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_factor, d_gram, sizeof(T) * n * n,
                                   cudaMemcpyDeviceToHost, stream));
        CUBLAS_CHECK(cublasSetPointerMode(cublasH, CUBLAS_POINTER_MODE_DEVICE));
    }

    void check(int n, const char *stage, cudaStream_t stream) {
        CUDA_CHECK(cudaStreamSynchronize(stream));

        for (int i = 0; i < n; ++i) {
            if (h_factor[i + i * n] == T{0}) {
                throw std::runtime_error(std::string(stage) +
                                         ": CholeskyQR produced a zero diagonal in R");
            }
        }
        if (*h_info < 0) {
            throw std::runtime_error(std::string(stage) + ": " +
                                     std::to_string(-*h_info) +
                                     "-th parameter is wrong in CholeskyQR");
        }
        if (*h_info > 0) {
            throw std::runtime_error(
                std::string(stage) + ": CholeskyQR failed, Gram matrix lost positive "
                                     "definiteness at leading minor " +
                std::to_string(*h_info));
        }
    }

  private:
    T *d_gram = nullptr;
    int *d_info = nullptr;
    int *h_info = nullptr;
    void *d_work = nullptr;
    std::size_t d_work_size = 0;
    void *h_work = nullptr;
    std::size_t h_work_size = 0;
    T *h_factor = nullptr;
};
