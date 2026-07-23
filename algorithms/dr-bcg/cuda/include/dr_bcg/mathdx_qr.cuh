#pragma once

// MathDx-fused CholeskyQR2 orthonormalization policy for DR-BCG.
//
// The pass is run twice (CholQR2) because a single CholQR squares
// the condition number of the input; the two triangular factors are combined so
// the returned R is the true factor with A = Q * R.

#include "config.h"

#ifdef SOLVERS_BUILD_MATHDX

#include <cassert>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#include "common/cuda_checks.h"
#include "common/cuda_event_timer.h"
#include "common/cuda_type.cuh"
#include "dr_bcg/qr.cuh"

#include <cublasdx.hpp>
#include <cusolverdx.hpp>

namespace dr_bcg::mathdx_detail {

#ifndef MATHDX_TARGET_SM
#error "MATHDX_TARGET_SM must be defined by CMake (derived from CMAKE_CUDA_ARCHITECTURES)"
#endif
inline constexpr unsigned MATHDX_CHOLQR2_SM = MATHDX_TARGET_SM;

inline constexpr int MATHDX_GEMM_TILE_ROWS = 128;
inline constexpr int MATHDX_GEMM_THREADS_PER_BLOCK = 128;

// Raise a kernel's dynamic shared-memory cap above the 48 KB static limit. The
// k=64 fp64 factor + panel can exceed 48 KB, requiring the opt-in attribute.
template <typename Kernel>
void enable_dynamic_smem(Kernel kernel, unsigned int bytes) {
    constexpr unsigned int smem_static_limit_bytes = 48 * 1024;
    if (bytes > smem_static_limit_bytes) {
        CUDA_CHECK(cudaFuncSetAttribute(
            reinterpret_cast<const void *>(kernel),
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(bytes)));
    }
}

// cuBLASDx GEMM descriptor: C[M,N] = A[M,K] * B[K,N], real, column-major,
// block execution on the target SM.
template <typename T, int M, int N, int K>
using GemmOp =
    decltype(cublasdx::Size<M, N, K>() + cublasdx::Precision<T>() +
             cublasdx::Type<cublasdx::type::real>() +
             cublasdx::Function<cublasdx::function::MM>() +
             cublasdx::SM<MATHDX_CHOLQR2_SM>() + cublasdx::Block() +
             cublasdx::BlockDim<MATHDX_GEMM_THREADS_PER_BLOCK>());

// cuSolverDx POTRF descriptor: upper Cholesky of an N*N SPD matrix, one block.
template <typename T, int N>
using PotrfOp =
    decltype(cusolverdx::Size<N, N>() + cusolverdx::Precision<T>() +
             cusolverdx::Type<cusolverdx::type::real>() +
             cusolverdx::Function<cusolverdx::function::potrf>() +
             cusolverdx::FillMode<cusolverdx::fill_mode::upper>() +
             cusolverdx::SM<MATHDX_CHOLQR2_SM>() + cusolverdx::Block());

// Gram: G += A^T * A, accumulated across row-tiles.
//
// Each block owns a TILE-row panel of the m*N column-major matrix A. It stages
// the panel into shared memory as both A (TILE*N) and A^T (N*TILE), runs a
// cuBLASDx block GEMM to form the partial N*N Gram, then atomically reduces it
// into the global accumulator (pre-zeroed by the launcher).
//
template <typename T, int N, int TILE>
__global__ void gram_kernel(const T *__restrict__ A, int m, int ldA,
                            T *__restrict__ G) {
    using GEMM = GemmOp<T, N, N, TILE>;

    extern __shared__ __align__(16) char smem[]; // NOLINT
    auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(smem);
    auto at = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a()); // N*TILE
    auto a = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());  // TILE*N
    auto g = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());  // N*N

    const int row0 = blockIdx.x * TILE;
    for (unsigned int e = threadIdx.x; e < TILE * N; e += blockDim.x) {
        const int r = e % TILE;
        const int col = e / TILE;
        const int gr = row0 + r;
        const T v = (gr < m) ? A[gr + (col * ldA)] : T{0};
        at(col, r) = v; // A^T : logical (N x TILE)
        a(r, col) = v;  // A   : logical (TILE x N)
    }
    __syncthreads();

    GEMM().execute(T{1}, at, a, T{0}, g);
    __syncthreads();

    for (unsigned int e = threadIdx.x; e < N * N; e += blockDim.x) {
        const int i = e % N;
        const int j = e / N;
        atomicAdd(&G[i + (j * N)], g(i, j));
    }
}

// POTRF: R = chol(G), upper-triangular, single block (cuSolverDx).
//
// Factors the N*N Gram in shared memory and writes the upper triangle to R
// (strict lower zeroed). info carries the leading-minor index on
// non-positive-definiteness for breakdown detection.
//
template <typename T, int N>
__global__ __launch_bounds__(PotrfOp<T, N>::max_threads_per_block) void potrf_kernel(const T *__restrict__ G, T *__restrict__ R,
                                                                                     int *__restrict__ info) {
    using POTRF = PotrfOp<T, N>;
    constexpr unsigned lda = POTRF::lda; // smem leading dim (may be padded)

    extern __shared__ __align__(16) char smem[]; // NOLINT
    T *s = reinterpret_cast<T *>(smem);

    for (unsigned int e = threadIdx.x; e < N * N; e += blockDim.x) {
        const int i = e % N;
        const int j = e / N;
        s[i + (j * lda)] = G[i + (j * N)];
    }
    __syncthreads();

    POTRF().execute(s, lda, info);
    __syncthreads();

    // Upper triangle holds R (G = R^T R); zero the strict lower part.
    for (unsigned int e = threadIdx.x; e < N * N; e += blockDim.x) {
        const int i = e % N; // row
        const int j = e / N; // col
        R[i + (j * N)] = (i <= j) ? s[i + (j * lda)] : T{0};
    }
}

// Triangular inverse: Rinv = R^-1 for an upper-triangular N*N R, single block.
//
// One thread per column (columns are independent): back-substitute from the
// diagonal upward. Negligible cost (N <= 64) done once per CholQR pass.
//
template <typename T, int N>
__global__ void trinv_kernel(const T *__restrict__ R, T *__restrict__ Rinv) {
    const unsigned int j = threadIdx.x;
    if (j >= N) {
        return;
    }
    for (int i = N - 1; i >= 0; --i) {
        if (i > j) {
            Rinv[i + (j * N)] = T{0};
        } else if (i == j) {
            Rinv[i + (j * N)] = T{1} / R[i + (i * N)];
        } else {
            T s = T{0};
            for (int k = i + 1; k <= j; ++k) {
                s += R[i + (k * N)] * Rinv[k + (j * N)];
            }
            Rinv[i + (j * N)] = -s / R[i + (i * N)];
        }
    }
}

// Apply: Q = A * Rinv, per row-tile (cuBLASDx block GEMM).
//
// C[TILE,N] = A_panel[TILE,N] * Rinv[N,N]. Q may alias A (in-place second pass)
// since each panel is staged through shared memory before being overwritten.
//
template <class T, int N, int TILE>
__global__ void apply_kernel(const T *A, T *Q, int m, int ldA,
                             const T *__restrict__ Rinv) {
    using GEMM = GemmOp<T, TILE, N, N>;

    extern __shared__ __align__(16) char smem[]; // NOLINT
    auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(smem);
    auto a = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());  // TILE*N
    auto ri = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b()); // N*N
    auto c = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());  // TILE*N

    const int row0 = blockIdx.x * TILE;
    for (unsigned int e = threadIdx.x; e < TILE * N; e += blockDim.x) {
        const int r = e % TILE;
        const int col = e / TILE;
        const int gr = row0 + r;
        a(r, col) = (gr < m) ? A[gr + (col * ldA)] : T{0};
    }
    for (unsigned int e = threadIdx.x; e < N * N; e += blockDim.x) {
        ri(e % N, e / N) = Rinv[e];
    }
    __syncthreads();

    GEMM().execute(T{1}, a, ri, T{0}, c);
    __syncthreads();

    for (unsigned int e = threadIdx.x; e < TILE * N; e += blockDim.x) {
        const int r = e % TILE;
        const int col = e / TILE;
        const int gr = row0 + r;
        if (gr < m) {
            Q[gr + (col * ldA)] = c(r, col);
        }
    }
}

// Combine the two CholQR factors: R = R2 * R1 (both upper-triangular N*N), so
// that A = Q * R. Single block, small N*N cuBLASDx GEMM.
//
template <typename T, int N>
__global__ void rmul_kernel(const T *__restrict__ R2, const T *__restrict__ R1,
                            T *__restrict__ R) {
    using GEMM = GemmOp<T, N, N, N>;

    extern __shared__ __align__(16) char smem[]; // NOLINT
    auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(smem);
    auto a = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());
    auto b = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());
    auto c = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());

    for (unsigned int e = threadIdx.x; e < N * N; e += blockDim.x) {
        a(e % N, e / N) = R2[e];
        b(e % N, e / N) = R1[e];
    }
    __syncthreads();

    GEMM().execute(T{1}, a, b, T{0}, c);
    __syncthreads();

    for (unsigned int e = threadIdx.x; e < N * N; e += blockDim.x) {
        R[e] = c(e % N, e / N);
    }
}

// One full CholQR2 for a fixed compile-time block width N.
// d_A: m*N input (column-major, ld = m). d_Q: m*N output. d_R: N*N output.
//
template <typename T, int N>
void launch_cholqr2(T *d_Q, T *d_R, const T *d_A, int m, int ldA, T *d_G,
                    T *d_R1, T *d_R2, T *d_Rinv, int *d_info,
                    cudaStream_t stream) {
    constexpr int TILE = MATHDX_GEMM_TILE_ROWS;

    using GramGEMM = GemmOp<T, N, N, TILE>;
    using ApplyGEMM = GemmOp<T, TILE, N, N>;
    using RmulGEMM = GemmOp<T, N, N, N>;
    using POTRF = PotrfOp<T, N>;

    static_assert(cusolverdx::is_supported<POTRF, MATHDX_CHOLQR2_SM>(),
                  "POTRF unsupported on target SM");

    const int grid = (m + TILE - 1) / TILE;

    const unsigned gram_smem = cublasdx::get_shared_storage_size<GramGEMM>();
    const unsigned apply_smem = cublasdx::get_shared_storage_size<ApplyGEMM>();
    const unsigned rmul_smem = cublasdx::get_shared_storage_size<RmulGEMM>();
    const unsigned potrf_smem = POTRF::shared_memory_size;

    enable_dynamic_smem(gram_kernel<T, N, TILE>, gram_smem);
    enable_dynamic_smem(apply_kernel<T, N, TILE>, apply_smem);
    enable_dynamic_smem(rmul_kernel<T, N>, rmul_smem);
    // cuSolverDx examples always raise the cap; harmless when <= 48 KB.
    CUDA_CHECK(cudaFuncSetAttribute(
        reinterpret_cast<const void *>(potrf_kernel<T, N>),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(potrf_smem)));

    // cuSolverDx kernels MUST launch with exactly POTRF::block_dim.
    const dim3 gram_bd = GramGEMM::block_dim;
    const dim3 apply_bd = ApplyGEMM::block_dim;
    const dim3 rmul_bd = RmulGEMM::block_dim;
    const dim3 potrf_bd = POTRF::block_dim;

    // ---- Pass 1: Q1 = A * R1^-1, R1 = chol(A^T A) ----
    CUDA_CHECK(cudaMemsetAsync(d_G, 0, sizeof(T) * N * N, stream));
    gram_kernel<T, N, TILE>
        <<<grid, gram_bd, gram_smem, stream>>>(d_A, m, ldA, d_G);
    CUDA_CHECK_LAST();
    potrf_kernel<T, N><<<1, potrf_bd, potrf_smem, stream>>>(d_G, d_R1, d_info);
    CUDA_CHECK_LAST();
    trinv_kernel<T, N><<<1, N, 0, stream>>>(d_R1, d_Rinv);
    CUDA_CHECK_LAST();
    apply_kernel<T, N, TILE>
        <<<grid, apply_bd, apply_smem, stream>>>(d_A, d_Q, m, ldA, d_Rinv);
    CUDA_CHECK_LAST();

    // ---- Pass 2: Q = Q1 * R2^-1, R2 = chol(Q1^T Q1) ----
    CUDA_CHECK(cudaMemsetAsync(d_G, 0, sizeof(T) * N * N, stream));
    gram_kernel<T, N, TILE>
        <<<grid, gram_bd, gram_smem, stream>>>(d_Q, m, ldA, d_G);
    CUDA_CHECK_LAST();
    potrf_kernel<T, N><<<1, potrf_bd, potrf_smem, stream>>>(d_G, d_R2, d_info);
    CUDA_CHECK_LAST();
    trinv_kernel<T, N><<<1, N, 0, stream>>>(d_R2, d_Rinv);
    CUDA_CHECK_LAST();
    apply_kernel<T, N, TILE>
        <<<grid, apply_bd, apply_smem, stream>>>(d_Q, d_Q, m, ldA, d_Rinv);
    CUDA_CHECK_LAST();

    // ---- Combine: R = R2 * R1 ----
    rmul_kernel<T, N><<<1, rmul_bd, rmul_smem, stream>>>(d_R2, d_R1, d_R);
    CUDA_CHECK_LAST();
}

} // namespace dr_bcg::mathdx_detail

// Fused Cholesky QR policy for use in DR-BCG configuration
//
template <cils::SupportedType T>
class MathDxCholeskyQr2 {
  public:
    // m: rows (problem size n), n: cols (block size s).
    MathDxCholeskyQr2(cusolverDnHandle_t & /*cusolverH*/,
                      cusolverDnParams_t & /*params*/, QrDimensions dims)
        : block_size(dims.n) {
        CUDA_CHECK(cudaMalloc(&d_G, sizeof(T) * dims.n * dims.n));
        CUDA_CHECK(cudaMalloc(&d_R1, sizeof(T) * dims.n * dims.n));
        CUDA_CHECK(cudaMalloc(&d_R2, sizeof(T) * dims.n * dims.n));
        CUDA_CHECK(cudaMalloc(&d_Rinv, sizeof(T) * dims.n * dims.n));
        CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&h_info),
                                  sizeof(int)));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&h_diag),
                                  sizeof(T) * dims.n));
    }

    MathDxCholeskyQr2(const MathDxCholeskyQr2 &) = delete;
    MathDxCholeskyQr2 &operator=(const MathDxCholeskyQr2 &) = delete;
    MathDxCholeskyQr2(MathDxCholeskyQr2 &&) = delete;
    MathDxCholeskyQr2 &operator=(MathDxCholeskyQr2 &&) = delete;

    ~MathDxCholeskyQr2() {
        if (d_G != nullptr) {
            CUDA_CHECK(cudaFree(d_G));
        }
        if (d_R1 != nullptr) {
            CUDA_CHECK(cudaFree(d_R1));
        }
        if (d_R2 != nullptr) {
            CUDA_CHECK(cudaFree(d_R2));
        }
        if (d_Rinv != nullptr) {
            CUDA_CHECK(cudaFree(d_Rinv));
        }
        if (d_info != nullptr) {
            CUDA_CHECK(cudaFree(d_info));
        }
        if (h_info != nullptr) {
            CUDA_CHECK(cudaFreeHost(h_info));
        }
        if (h_diag != nullptr) {
            CUDA_CHECK(cudaFreeHost(h_diag));
        }
    }

    void solve(T *&d_Q, T *&d_R, const T *d_A, int m, int n,
               cublasHandle_t & /*cublasH*/, cusolverDnHandle_t & /*cusolverH*/,
               cusolverDnParams_t & /*params*/, cudaStream_t &stream) {
        assert(n < m && "Expect cols to be less than rows for DR-BCG");
        CudaTimerRange rng{g_event_timer, "QR:func", stream};

        dispatch(d_Q, d_R, d_A, {.m = m, .n = n}, stream);

        // Stage R2's diagonal (final-pass factor) for the breakdown check.
        CUDA_CHECK(cudaMemcpy2DAsync(h_diag, sizeof(T), d_R2,
                                     sizeof(T) * (n + 1), sizeof(T), n,
                                     cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_info, d_info, sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
    }

    void check(int n, const char *stage, cudaStream_t stream) {
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (*h_info > 0) {
            throw std::runtime_error(
                std::string(stage) +
                ": MathDx CholeskyQR2 failed, Gram lost positive definiteness "
                "at leading minor " +
                std::to_string(*h_info) +
                " (input may be too ill-conditioned for CholQR2)");
        }
        if (*h_info < 0) {
            throw std::runtime_error(std::string(stage) + ": " +
                                     std::to_string(-*h_info) +
                                     "-th parameter is wrong in MathDx POTRF");
        }
        for (int i = 0; i < n; ++i) {
            if (h_diag[i] == T{0}) {
                throw std::runtime_error(
                    std::string(stage) +
                    ": MathDx CholeskyQR2 produced a zero diagonal in R");
            }
        }
    }

  private:
    void dispatch(T *d_Q, T *d_R, const T *d_A, QrDimensions dims,
                  cudaStream_t stream) {
        using dr_bcg::mathdx_detail::launch_cholqr2;
        switch (dims.n) {
            // NOLINTBEGIN
        case 1:
            launch_cholqr2<T, 1>(d_Q, d_R, d_A, dims.m, dims.m, d_G, d_R1, d_R2, d_Rinv, d_info, stream);
            break;
        case 2:
            launch_cholqr2<T, 2>(d_Q, d_R, d_A, dims.m, dims.m, d_G, d_R1, d_R2, d_Rinv, d_info, stream);
            break;
        case 4:
            launch_cholqr2<T, 4>(d_Q, d_R, d_A, dims.m, dims.m, d_G, d_R1, d_R2, d_Rinv, d_info, stream);
            break;
        case 8:
            launch_cholqr2<T, 8>(d_Q, d_R, d_A, dims.m, dims.m, d_G, d_R1, d_R2, d_Rinv, d_info, stream);
            break;
        case 16:
            launch_cholqr2<T, 16>(d_Q, d_R, d_A, dims.m, dims.m, d_G, d_R1, d_R2, d_Rinv, d_info, stream);
            break;
        case 32:
            launch_cholqr2<T, 32>(d_Q, d_R, d_A, dims.m, dims.m, d_G, d_R1, d_R2, d_Rinv, d_info, stream);
            break;
        case 64:
            launch_cholqr2<T, 64>(d_Q, d_R, d_A, dims.m, dims.m, d_G, d_R1, d_R2, d_Rinv, d_info, stream);
            break;
            // NOLINTEND
        default:
            throw std::runtime_error(
                "unsupported block size " + std::to_string(dims.n) +
                " for MathDx CholQR2 (supported: 1, 2, 4, 8, 16, 32, 64)");
        }
    }

    int block_size = 0;
    T *d_G = nullptr;
    T *d_R1 = nullptr;
    T *d_R2 = nullptr;
    T *d_Rinv = nullptr;
    int *d_info = nullptr;
    int *h_info = nullptr;
    T *h_diag = nullptr;
};

#endif // SOLVERS_BUILD_MATHDX
