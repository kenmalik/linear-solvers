#pragma once

// MathDx-fused reduced-system (xi) chain for DR-BCG.
//
// This fuses the following operations:
//
//   G = s^T * AS                       (cuBLASDx tiled Gram)
//   G = F^T F, F = chol(G)             (cuSolverDx POTRF; info -> breakdown)
//
//   apply G^-1 twice:
//     C = G^-1 * sigma -> X += s*C     (X update)
//     U = AS * G^-1                    (feeds the w update; = A*s*xi)

#include "config.h"

#ifdef SOLVERS_BUILD_MATHDX

#include "mathdx_qr.cuh"

#include "common/cuda_checks.h"
#include "common/cuda_event_timer.h"

#include <cublasdx.hpp>
#include <cuda_runtime.h>
#include <cusolverdx.hpp>

#include <stdexcept>
#include <string>

namespace cils::dr_bcg::cuda::detail {

// Gram of two distinct panels: G += s^T * AS, accumulated across row-tiles.
//
// Two-input variant of gram_kernel (which forms A^T*A from a single input).
// Each block owns a TILE-row panel of the m*N column-major matrices s and AS,
// stages s^T (N*TILE) and AS (TILE*N) into shared memory, runs a cuBLASDx block
// GEMM to form the partial N*N Gram, then atomically reduces it into the global
// accumulator (pre-zeroed by the launcher).
//
template <typename T, int N, int TILE>
__global__ void gram2_kernel(const T *__restrict__ S, const T *__restrict__ AS,
                             int m, int ld, T *__restrict__ G) {
    using GEMM = GemmOp<T, N, N, TILE>;

    extern __shared__ __align__(16) char smem[]; // NOLINT
    auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(smem);
    auto st = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a()); // N*TILE
    auto as = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b()); // TILE*N
    auto g = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());  // N*N

    const int row0 = blockIdx.x * TILE;
    for (unsigned int e = threadIdx.x; e < TILE * N; e += blockDim.x) {
        const int r = e % TILE;
        const int col = e / TILE;
        const int gr = row0 + r;
        st(col, r) = (gr < m) ? S[gr + (col * ld)] : T{0};  // s^T : (N x TILE)
        as(r, col) = (gr < m) ? AS[gr + (col * ld)] : T{0}; // AS  : (TILE x N)
    }
    __syncthreads();

    GEMM().execute(T{1}, st, as, T{0}, g);
    __syncthreads();

    for (unsigned int e = threadIdx.x; e < N * N; e += blockDim.x) {
        const int i = e % N;
        const int j = e / N;
        atomicAdd(&G[i + (j * N)], g(i, j));
    }
}

// Explicit inverse from the Cholesky factor: Ginv = Finv * Finv^T = G^-1,
// where Finv is the upper-triangular inverse of F (G = F^T F). Single block,
// small N*N cuBLASDx GEMM.
//
template <typename T, int N>
__global__ void ginv_kernel(const T *__restrict__ Finv, T *__restrict__ Ginv) {
    using GEMM = GemmOp<T, N, N, N>;

    extern __shared__ __align__(16) char smem[]; // NOLINT
    auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(smem);
    auto a = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a()); // Finv
    auto b = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b()); // Finv^T
    auto c = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());

    for (unsigned int e = threadIdx.x; e < N * N; e += blockDim.x) {
        const int i = e % N;
        const int j = e / N;
        a(i, j) = Finv[i + (j * N)]; // Finv
        b(i, j) = Finv[j + (i * N)]; // Finv^T
    }
    __syncthreads();

    GEMM().execute(T{1}, a, b, T{0}, c);
    __syncthreads();

    for (unsigned int e = threadIdx.x; e < N * N; e += blockDim.x) {
        Ginv[e] = c(e % N, e / N);
    }
}

// Accumulating apply: C_out += A * Op, per row-tile (cuBLASDx block GEMM).
//
// apply_kernel (mathdx_qr.cuh) with beta=1: the TILE*N result is added into the
// existing C_out panel rather than overwriting it, so X += s*C lands in place.
//
template <typename T, int N, int TILE>
__global__ void apply_accum_kernel(const T *__restrict__ A,
                                   T *__restrict__ C_out, int m, int ld,
                                   const T *__restrict__ Op) {
    using GEMM = GemmOp<T, TILE, N, N>;

    extern __shared__ __align__(16) char smem[]; // NOLINT
    auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(smem);
    auto a = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());  // TILE*N
    auto op = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b()); // N*N
    auto c = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());  // TILE*N

    const int row0 = blockIdx.x * TILE;
    for (unsigned int e = threadIdx.x; e < TILE * N; e += blockDim.x) {
        const int r = e % TILE;
        const int col = e / TILE;
        const int gr = row0 + r;
        a(r, col) = (gr < m) ? A[gr + (col * ld)] : T{0};
    }
    for (unsigned int e = threadIdx.x; e < N * N; e += blockDim.x) {
        op(e % N, e / N) = Op[e];
    }
    __syncthreads();

    GEMM().execute(T{1}, a, op, T{0}, c);
    __syncthreads();

    for (unsigned int e = threadIdx.x; e < TILE * N; e += blockDim.x) {
        const int r = e % TILE;
        const int col = e / TILE;
        const int gr = row0 + r;
        if (gr < m) {
            C_out[gr + (col * ld)] += c(r, col);
        }
    }
}

// One full fused xi chain for a fixed compile-time block width N.
//   d_s, d_AS: m*N inputs (column-major, ld). d_sigma: N*N. d_X: m*N (updated
//   in place, X += s*G^-1*sigma). d_U: m*N output (= AS*G^-1, feeds w update).
// Workspaces (all N*N, single allocation each): d_G, d_F, d_Finv, d_Ginv, d_C.
//
template <typename T, int N>
void launch_xi(T *d_s, T *d_AS, T *d_sigma, T *d_X, T *d_U, int m, int ld,
               T *d_G, T *d_F, T *d_Finv, T *d_Ginv, T *d_C, int *d_info,
               cudaStream_t stream) {
    constexpr int TILE = MATHDX_GEMM_TILE_ROWS;

    using GramGEMM = GemmOp<T, N, N, TILE>;
    using ApplyGEMM = GemmOp<T, TILE, N, N>;
    using SmallGEMM = GemmOp<T, N, N, N>;
    using POTRF = PotrfOp<T, N>;

    static_assert(cusolverdx::is_supported<POTRF, MATHDX_CHOLQR2_SM>(),
                  "POTRF unsupported on target SM");

    const int grid = (m + TILE - 1) / TILE;

    const unsigned gram_smem = cublasdx::get_shared_storage_size<GramGEMM>();
    const unsigned apply_smem = cublasdx::get_shared_storage_size<ApplyGEMM>();
    const unsigned small_smem = cublasdx::get_shared_storage_size<SmallGEMM>();
    const unsigned potrf_smem = POTRF::shared_memory_size;

    enable_dynamic_smem(gram2_kernel<T, N, TILE>, gram_smem);
    enable_dynamic_smem(apply_kernel<T, N, TILE>, apply_smem);
    enable_dynamic_smem(apply_accum_kernel<T, N, TILE>, apply_smem);
    enable_dynamic_smem(ginv_kernel<T, N>, small_smem);
    enable_dynamic_smem(rmul_kernel<T, N>, small_smem);
    CUDA_CHECK(cudaFuncSetAttribute(
        reinterpret_cast<const void *>(potrf_kernel<T, N>),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(potrf_smem)));

    const dim3 gram_bd = GramGEMM::block_dim;
    const dim3 apply_bd = ApplyGEMM::block_dim;
    const dim3 small_bd = SmallGEMM::block_dim;
    const dim3 potrf_bd = POTRF::block_dim;

    // G = s^T * AS
    CUDA_CHECK(cudaMemsetAsync(d_G, 0, sizeof(T) * N * N, stream));
    gram2_kernel<T, N, TILE>
        <<<grid, gram_bd, gram_smem, stream>>>(d_s, d_AS, m, ld, d_G);
    CUDA_CHECK_LAST();

    // F = chol(G); Finv = F^-1; Ginv = Finv*Finv^T = G^-1
    potrf_kernel<T, N><<<1, potrf_bd, potrf_smem, stream>>>(d_G, d_F, d_info);
    CUDA_CHECK_LAST();
    trinv_kernel<T, N><<<1, N, 0, stream>>>(d_F, d_Finv);
    CUDA_CHECK_LAST();
    ginv_kernel<T, N><<<1, small_bd, small_smem, stream>>>(d_Finv, d_Ginv);
    CUDA_CHECK_LAST();

    // C = G^-1 * sigma; X += s * C
    rmul_kernel<T, N><<<1, small_bd, small_smem, stream>>>(d_Ginv, d_sigma, d_C);
    CUDA_CHECK_LAST();
    apply_accum_kernel<T, N, TILE>
        <<<grid, apply_bd, apply_smem, stream>>>(d_s, d_X, m, ld, d_C);
    CUDA_CHECK_LAST();

    // U = AS * G^-1
    apply_kernel<T, N, TILE>
        <<<grid, apply_bd, apply_smem, stream>>>(d_AS, d_U, m, ld, d_Ginv);
    CUDA_CHECK_LAST();
}

// Fused MathDx reduced-system (xi) chain helper for DR-BCG.
// Owns the small N*N workspaces and dispatches the compile-time-N kernel chain.
// Used by the fused solve loop (mathdx_fused.cuh) alongside MathDxCholeskyQr2.
template <cils::detail::SupportedType T>
class MathDxXiChain {
  public:
    struct ProblemSize {
        int m;
        int n;
    };

    // n: block size s (reduced-system side length).
    explicit MathDxXiChain(int n) : block_size(n) {
        CUDA_CHECK(cudaMalloc(&d_G, sizeof(T) * n * n));
        CUDA_CHECK(cudaMalloc(&d_F, sizeof(T) * n * n));
        CUDA_CHECK(cudaMalloc(&d_Finv, sizeof(T) * n * n));
        CUDA_CHECK(cudaMalloc(&d_Ginv, sizeof(T) * n * n));
        CUDA_CHECK(cudaMalloc(&d_C, sizeof(T) * n * n));
        CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&h_info),
                                  sizeof(int)));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&h_diag),
                                  sizeof(T) * n));
    }

    MathDxXiChain(const MathDxXiChain &) = delete;
    MathDxXiChain &operator=(const MathDxXiChain &) = delete;
    MathDxXiChain(MathDxXiChain &&) = delete;
    MathDxXiChain &operator=(MathDxXiChain &&) = delete;

    ~MathDxXiChain() {
        if (d_G != nullptr) {
            CUDA_CHECK(cudaFree(d_G));
        }
        if (d_F != nullptr) {
            CUDA_CHECK(cudaFree(d_F));
        }
        if (d_Finv != nullptr) {
            CUDA_CHECK(cudaFree(d_Finv));
        }
        if (d_Ginv != nullptr) {
            CUDA_CHECK(cudaFree(d_Ginv));
        }
        if (d_C != nullptr) {
            CUDA_CHECK(cudaFree(d_C));
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

    // X += s * G^-1 * sigma (in place); d_U <- AS * G^-1. m: rows (n), n: block.
    void apply(T *d_s, T *d_AS, T *d_sigma, T *d_X, T *d_U, int m, int n,
               cudaStream_t stream) {
        assert(n < m && "Expect cols to be less than rows for DR-BCG");
        cils::detail::CudaTimerRange rng{cils::detail::g_event_timer, "xi:func", stream};

        dispatch(d_s, d_AS, d_sigma, d_X, d_U, {.m = m, .n = n}, stream);

        // Stage F's diagonal + POTRF info for the breakdown check.
        CUDA_CHECK(cudaMemcpy2DAsync(h_diag, sizeof(T), d_F,
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
                ": MathDx xi chain failed, reduced-system Gram lost positive "
                "definiteness at leading minor " +
                std::to_string(*h_info) + " (breakdown / rank deficiency)");
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
                    ": MathDx xi chain produced a zero diagonal in the Gram "
                    "factor");
            }
        }
    }

  private:
    void dispatch(T *d_s, T *d_AS, T *d_sigma, T *d_X, T *d_U, ProblemSize size,
                  cudaStream_t stream) {
        using cils::dr_bcg::cuda::detail::launch_xi;
        switch (size.n) {
        // NOLINTBEGIN
        case 1:
            launch_xi<T, 1>(d_s, d_AS, d_sigma, d_X, d_U, size.m, size.m, d_G, d_F,
                            d_Finv, d_Ginv, d_C, d_info, stream);
            break;
        case 2:
            launch_xi<T, 2>(d_s, d_AS, d_sigma, d_X, d_U, size.m, size.m, d_G, d_F,
                            d_Finv, d_Ginv, d_C, d_info, stream);
            break;
        case 4:
            launch_xi<T, 4>(d_s, d_AS, d_sigma, d_X, d_U, size.m, size.m, d_G, d_F,
                            d_Finv, d_Ginv, d_C, d_info, stream);
            break;
        case 8:
            launch_xi<T, 8>(d_s, d_AS, d_sigma, d_X, d_U, size.m, size.m, d_G, d_F,
                            d_Finv, d_Ginv, d_C, d_info, stream);
            break;
        case 16:
            launch_xi<T, 16>(d_s, d_AS, d_sigma, d_X, d_U, size.m, size.m, d_G, d_F,
                             d_Finv, d_Ginv, d_C, d_info, stream);
            break;
        case 32:
            launch_xi<T, 32>(d_s, d_AS, d_sigma, d_X, d_U, size.m, size.m, d_G, d_F,
                             d_Finv, d_Ginv, d_C, d_info, stream);
            break;
        case 64:
            launch_xi<T, 64>(d_s, d_AS, d_sigma, d_X, d_U, size.m, size.m, d_G, d_F,
                             d_Finv, d_Ginv, d_C, d_info, stream);
            break;
        // NOLINTEND
        default:
            throw std::runtime_error(
                "unsupported block size " + std::to_string(size.n) +
                " for MathDx xi chain (supported: 1, 2, 4, 8, 16, 32, 64)");
        }
    }

    int block_size = 0;
    T *d_G = nullptr;
    T *d_F = nullptr;
    T *d_Finv = nullptr;
    T *d_Ginv = nullptr;
    T *d_C = nullptr;
    int *d_info = nullptr;
    int *h_info = nullptr;
    T *h_diag = nullptr;
};

} // namespace cils::dr_bcg::cuda::detail

#endif // SOLVERS_BUILD_MATHDX
