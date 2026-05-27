#pragma once

#include <common/cuda_event_timer.h>

template <typename T>
struct HouseholderQrWorkspace {
    T *d_tau = nullptr;
    void *d_work = nullptr;
    int *d_info = nullptr;
    int *h_info = nullptr;
    void *h_work = nullptr;

    std::size_t lwork_geqrf_d = 0;
    std::size_t lwork_geqrf_h = 0;
    int numfloats_orgqr_d = 0;

    HouseholderQrWorkspace() = default;
    HouseholderQrWorkspace(const HouseholderQrWorkspace &) = delete;
    HouseholderQrWorkspace &operator=(const HouseholderQrWorkspace &) = delete;

    // m: rows of Q (problem size n), n: cols of Q (block size s)
    void allocate(cusolverDnHandle_t &cusolverH, cusolverDnParams_t &params,
                  int m, int n) {
        constexpr cudaDataType_t data_type = Type_info<T>::cuda;

        CUDA_CHECK(cudaMalloc(&d_tau, sizeof(T) * n));
        CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
        CUDA_CHECK(
            cudaMallocHost(reinterpret_cast<void **>(&h_info), sizeof(int)));

        // Dummy m×n device buffer needed to query workspace sizes.
        // Buffer size queries are dimension/type-driven; values are not read.
        T *d_dummy = nullptr;
        CUDA_CHECK(cudaMalloc(&d_dummy, sizeof(T) * m * n));

        CUSOLVER_CHECK(cusolverDnXgeqrf_bufferSize(
            cusolverH, params, m, n, data_type, d_dummy, m, data_type, d_tau,
            data_type, &lwork_geqrf_d, &lwork_geqrf_h));

        if constexpr (std::is_same_v<T, float>) {
            CUSOLVER_CHECK(cusolverDnSorgqr_bufferSize(
                cusolverH, m, n, n, d_dummy, m, d_tau, &numfloats_orgqr_d));
        } else {
            CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(
                cusolverH, m, n, n, d_dummy, m, d_tau, &numfloats_orgqr_d));
        }

        CUDA_CHECK(cudaFree(d_dummy));

        const std::size_t lwork_orgqr_d = numfloats_orgqr_d * sizeof(T);
        CUDA_CHECK(cudaMalloc(&d_work, std::max(lwork_geqrf_d, lwork_orgqr_d)));

        if (lwork_geqrf_h > 0) {
            h_work = malloc(lwork_geqrf_h);
            if (!h_work)
                throw std::runtime_error(
                    "Error: QrWorkspace h_work not allocated.");
        }
    }

    ~HouseholderQrWorkspace() {
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
};

template <typename T>
struct CholQrWorkspace {
    T *d_gram = nullptr;
    int *d_info = nullptr;
    int *h_info = nullptr;
    void *d_work = nullptr;
    std::size_t d_work_size = 0;
    void *h_work = nullptr;
    std::size_t h_work_size = 0;
    T *h_factor = nullptr;

    CholQrWorkspace() = default;
    CholQrWorkspace(const CholQrWorkspace &) = delete;
    CholQrWorkspace &operator=(const CholQrWorkspace &) = delete;

    void allocate(cusolverDnHandle_t &cusolverH, cusolverDnParams_t &params,
                  int n) {
        constexpr cudaDataType_t data_type = Type_info<T>::cuda;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_gram),
                              sizeof(T) * n * n));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));
        CUDA_CHECK(
            cudaMallocHost(reinterpret_cast<void **>(&h_info), sizeof(int)));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&h_factor),
                                  sizeof(T) * n * n));

        T *d_dummy = nullptr;
        CUDA_CHECK(cudaMalloc(&d_dummy, sizeof(T) * n * n));
        CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(
            cusolverH, params, CUBLAS_FILL_MODE_UPPER, n, data_type, d_dummy,
            n, data_type, &d_work_size, &h_work_size));
        CUDA_CHECK(cudaFree(d_dummy));

        if (d_work_size > 0) {
            CUDA_CHECK(
                cudaMalloc(reinterpret_cast<void **>(&d_work), d_work_size));
        }
        if (h_work_size > 0) {
            h_work = malloc(h_work_size);
            if (!h_work)
                throw std::runtime_error(
                    "Error: CholQrWorkspace h_work not allocated.");
        }
    }

    ~CholQrWorkspace() {
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
};

template <typename T>
void cholesky_qr(T *&d_Q, const T *&d_A, const int &m, const int &n, cudaStream_t &stream, cublasHandle_t &cublasH, CholQrWorkspace<T> &cholqr_ws, cusolverDnHandle_t &cusolverH, cusolverDnParams_t &params, const cudaDataType_t &data_type, T *&d_R) {
    CudaTimerRange rng{g_event_timer, "QR:func", stream};

    constexpr T alpha = 1;
    constexpr T beta = 0;
    CUDA_CHECK(cudaMemcpyAsync(d_Q, d_A, sizeof(T) * m * n,
                               cudaMemcpyDeviceToDevice, stream));
    CUBLAS_CHECK(cublasSetPointerMode(cublasH, CUBLAS_POINTER_MODE_HOST));

    if constexpr (std::is_same_v<T, float>) {
        CudaTimerRange rng{g_event_timer, "QR:syrk", stream};
        CUBLAS_CHECK(cublasSsyrk(cublasH, CUBLAS_FILL_MODE_UPPER,
                                 CUBLAS_OP_T, n, m, &alpha, d_A, m, &beta,
                                 cholqr_ws.d_gram, n));
    } else {
        CudaTimerRange rng{g_event_timer, "QR:syrk", stream};
        CUBLAS_CHECK(cublasDsyrk(cublasH, CUBLAS_FILL_MODE_UPPER,
                                 CUBLAS_OP_T, n, m, &alpha, d_A, m, &beta,
                                 cholqr_ws.d_gram, n));
    }

    {
        CudaTimerRange rng{g_event_timer, "QR:potrf", stream};
        CUSOLVER_CHECK(cusolverDnXpotrf(
            cusolverH, params, CUBLAS_FILL_MODE_UPPER, n, data_type,
            cholqr_ws.d_gram, n, data_type, cholqr_ws.d_work,
            cholqr_ws.d_work_size, cholqr_ws.h_work, cholqr_ws.h_work_size,
            cholqr_ws.d_info));
    }

    {
        CudaTimerRange rng{g_event_timer, "QR:copy_upper_triangular", stream};
        copy_upper_triangular(d_R, cholqr_ws.d_gram, n, n, stream);
    }

    if constexpr (std::is_same_v<T, float>) {
        CudaTimerRange rng{g_event_timer, "QR:trsm", stream};
        CUBLAS_CHECK(cublasStrsm_v2(
            cublasH, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha,
            cholqr_ws.d_gram, n, d_Q, m));
    } else {
        CudaTimerRange rng{g_event_timer, "QR:trsm", stream};
        CUBLAS_CHECK(cublasDtrsm_v2(
            cublasH, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha,
            cholqr_ws.d_gram, n, d_Q, m));
    }

    CUDA_CHECK(cudaMemcpyAsync(cholqr_ws.h_info, cholqr_ws.d_info,
                               sizeof(int), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(cholqr_ws.h_factor, cholqr_ws.d_gram,
                               sizeof(T) * n * n, cudaMemcpyDeviceToHost,
                               stream));
    CUBLAS_CHECK(cublasSetPointerMode(cublasH, CUBLAS_POINTER_MODE_DEVICE));
}

template <typename T>
void householder_qr(T *&d_Q, const T *&d_A, const int &m, const int &n, cudaStream_t &stream, cusolverDnHandle_t &cusolverH, cusolverDnParams_t &params, const cudaDataType_t &data_type, HouseholderQrWorkspace<T> &householder_ws, T *&d_R) {
    CudaTimerRange rng{g_event_timer, "QR:func", stream};

    CUDA_CHECK(cudaMemcpyAsync(d_Q, d_A, sizeof(T) * m * n,
                               cudaMemcpyDeviceToDevice, stream));

    {
        CudaTimerRange rng{g_event_timer, "QR:geqrf", stream};
        CUSOLVER_CHECK(cusolverDnXgeqrf(
            cusolverH, params, m, n, data_type, d_Q, m, data_type,
            householder_ws.d_tau, data_type, householder_ws.d_work,
            householder_ws.lwork_geqrf_d, householder_ws.h_work,
            householder_ws.lwork_geqrf_h, householder_ws.d_info));
    }

    {
        CudaTimerRange rng{g_event_timer, "QR:copy_upper_triangular", stream};
        copy_upper_triangular(d_R, d_Q, m, n, stream);
    }

    if constexpr (std::is_same_v<T, float>) {
        CudaTimerRange rng{g_event_timer, "QR:orgqr", stream};
        CUSOLVER_CHECK(cusolverDnSorgqr(
            cusolverH, m, n, n, d_Q, m, householder_ws.d_tau,
            reinterpret_cast<T *>(householder_ws.d_work),
            householder_ws.numfloats_orgqr_d, householder_ws.d_info));
    } else {
        CudaTimerRange rng{g_event_timer, "QR:orgqr", stream};
        CUSOLVER_CHECK(cusolverDnDorgqr(
            cusolverH, m, n, n, d_Q, m, householder_ws.d_tau,
            reinterpret_cast<T *>(householder_ws.d_work),
            householder_ws.numfloats_orgqr_d, householder_ws.d_info));
    }

    CUDA_CHECK(cudaMemcpyAsync(householder_ws.h_info,
                               householder_ws.d_info, sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
}

template <typename T>
void orthonormalize_block(
    cublasHandle_t &cublasH, cusolverDnHandle_t &cusolverH,
    cusolverDnParams_t &params, T *d_Q, T *d_R, const int m, const int n,
    const T *d_A, dr_bcg::cuda::QrBackend backend,
    HouseholderQrWorkspace<T> &householder_ws, CholQrWorkspace<T> &cholqr_ws,
    cudaStream_t stream) {
    NVTX3_FUNC_RANGE();

    constexpr cudaDataType_t data_type = Type_info<T>::cuda;

    assert(n < m && "Expect cols to be less than rows for DR-BCG");

    switch (backend) {
    case dr_bcg::cuda::QrBackend::Householder:
        householder_qr(d_Q, d_A, m, n, stream, cusolverH, params, data_type, householder_ws, d_R);
        break;
    case dr_bcg::cuda::QrBackend::CholQR:
        cholesky_qr(d_Q, d_A, m, n, stream, cublasH, cholqr_ws, cusolverH, params, data_type, d_R);
        break;
    default:
        throw std::runtime_error("Unknown QR backend");
    }
}
