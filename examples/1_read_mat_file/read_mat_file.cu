#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include <mat_utils/mat_reader.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "dr_bcg/helper.h"
#include "dr_bcg/sparse.h"

__global__ void set_val(float *A_d, float val, size_t num_elements) {
    const int idx = blockIdx.x * blockDim.y + threadIdx.x;
    if (idx < num_elements) {
        A_d[idx] = val;
    }
}

class DeviceSuiteSparseMatrix {
  public:
    explicit DeviceSuiteSparseMatrix(mat_utils::SpMatReader &ssm_A) {
        CUDA_CHECK(cudaMalloc(&d_rowPtr, sizeof(int64_t) * (ssm_A.rows() + 1)));
        CUDA_CHECK(cudaMalloc(&d_colInd, sizeof(int64_t) * ssm_A.nnz()));
        CUDA_CHECK(cudaMalloc(&d_vals, sizeof(float) * ssm_A.nnz()));

        std::vector<size_t> rowCounts(ssm_A.rows(), 0);
        for (size_t j = 0; j < ssm_A.cols(); ++j) {
            for (size_t p = ssm_A.jc()[j]; p < ssm_A.jc()[j + 1]; ++p) {
                ++rowCounts[ssm_A.ir()[p]];
            }
        }

        std::vector<size_t> csrRowPtr_sz(ssm_A.rows() + 1, 0);
        for (size_t i = 0; i < ssm_A.rows(); ++i)
            csrRowPtr_sz[i + 1] = csrRowPtr_sz[i] + rowCounts[i];

        std::vector<size_t> next = csrRowPtr_sz;

        std::vector<size_t> csrColInd_sz(ssm_A.nnz());
        std::vector<float> csrVal(ssm_A.nnz());

        for (size_t j = 0; j < ssm_A.cols(); ++j) {
            for (size_t p = ssm_A.jc()[j]; p < ssm_A.jc()[j + 1]; ++p) {
                size_t row = ssm_A.ir()[p];
                size_t dst = next[row]++;
                csrColInd_sz[dst] = j;
                csrVal[dst] = static_cast<float>(ssm_A.data()[p]);
            }
        }

        // Convert host indices to int64_t
        std::vector<int64_t> csrRowPtr64(ssm_A.rows() + 1);
        std::vector<int64_t> csrColInd64(ssm_A.nnz());
        for (size_t i = 0; i < csrRowPtr_sz.size(); ++i)
            csrRowPtr64[i] = static_cast<int64_t>(csrRowPtr_sz[i]);
        for (size_t k = 0; k < csrColInd_sz.size(); ++k)
            csrColInd64[k] = static_cast<int64_t>(csrColInd_sz[k]);

        CUDA_CHECK(cudaMemcpy(d_rowPtr, csrRowPtr64.data(),
                              sizeof(int64_t) * csrRowPtr64.size(),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_colInd, csrColInd64.data(),
                              sizeof(int64_t) * csrColInd64.size(),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vals, csrVal.data(),
                              sizeof(float) * csrVal.size(),
                              cudaMemcpyHostToDevice));

        CUSPARSE_CHECK(cusparseCreateCsr(
            &A_, ssm_A.rows(), ssm_A.cols(), ssm_A.nnz(), d_rowPtr, d_colInd,
            d_vals, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    }

    ~DeviceSuiteSparseMatrix() {
        if (A_) {
            CUSPARSE_CHECK(cusparseDestroySpMat(A_));
        }
        if (d_rowPtr) {
            CUDA_CHECK(cudaFree(d_rowPtr));
            d_rowPtr = nullptr;
        }
        if (d_colInd) {
            CUDA_CHECK(cudaFree(d_colInd));
            d_colInd = nullptr;
        }
        if (d_vals) {
            CUDA_CHECK(cudaFree(d_vals));
            d_vals = nullptr;
        }
    }

    cusparseSpMatDescr_t &get() { return A_; }

  private:
    int64_t *d_rowPtr = nullptr;
    int64_t *d_colInd = nullptr;
    float *d_vals = nullptr;
    cusparseSpMatDescr_t A_{};
};

int main(int argc, char *argv[]) {
    int s;
    try {
        if (argc == 2) {
            s = 1;
        } else if (argc == 3) {
            s = std::atoi(argv[2]);
        } else {
            throw std::invalid_argument("Invalid arg count");
        }
    } catch (const std::exception &e) {
        std::cerr << "Usage: ./example_2 [.mat file] [block size]" << std::endl;
        return 1;
    }

    const std::string matrix_file = argv[1];
    mat_utils::SpMatReader ssm(matrix_file, {"Problem"}, "A");
    DeviceSuiteSparseMatrix A(ssm);

    const int n = ssm.rows();

    cusparseDnMatDescr_t X;
    thrust::device_vector<float> X_v(n * s, 0.0f);
    CUSPARSE_CHECK(cusparseCreateDnMat(&X, n, s, n,
                                       thrust::raw_pointer_cast(X_v.data()),
                                       CUDA_R_32F, CUSPARSE_ORDER_COL));

    cusparseDnMatDescr_t B;
    thrust::device_vector<float> B_v(n * s, 1.0f);
    CUSPARSE_CHECK(cusparseCreateDnMat(&B, n, s, n,
                                       thrust::raw_pointer_cast(B_v.data()),
                                       CUDA_R_32F, CUSPARSE_ORDER_COL));

    constexpr float tolerance = 1e-6;
    constexpr int max_iterations = 1;

    std::cout << "n: " << n << std::endl;
    std::cout << "s: " << s << std::endl;

    std::cerr << "Running..." << std::endl;
    int iterations = dr_bcg(A.get(), X, B, tolerance, max_iterations);
    std::cerr << "Finished!" << std::endl;

    // Verification
    // cusparseDnMatDescr_t AX;
    // thrust::device_vector<float> AX_v(n * s);
    // CUSPARSE_CHECK(cusparseCreateDnMat(&AX, n, s, n,
    //                                    thrust::raw_pointer_cast(AX_v.data()),
    //                                    CUDA_R_32F, CUSPARSE_ORDER_COL));

    // constexpr cusparseOperation_t transpose = CUSPARSE_OPERATION_NON_TRANSPOSE;
    // constexpr float alpha = 1;
    // constexpr float beta = 0;

    // cusparseHandle_t cusparseH = NULL;
    // CUSPARSE_CHECK(cusparseCreate(&cusparseH));

    // void *buffer = nullptr;
    // size_t buffer_size = 0;

    // CUSPARSE_CHECK(cusparseSpMM_bufferSize(
    //     cusparseH, transpose, transpose, &alpha, A.get(), X, &beta, AX,
    //     CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size));

    // if (buffer_size > 0) {
    //     CUDA_CHECK(cudaMalloc(&buffer, buffer_size));
    // }

    // CUSPARSE_CHECK(cusparseSpMM(cusparseH, transpose, transpose, &alpha,
    //                             A.get(), X, &beta, AX, CUDA_R_32F,
    //                             CUSPARSE_SPMM_ALG_DEFAULT, buffer));

    // if (buffer) {
    //     CUDA_CHECK(cudaFree(buffer));
    // }

    // CUSPARSE_CHECK(cusparseDestroy(cusparseH));

    // constexpr float check_tolerance = 0.001;
    // float min_error = std::numeric_limits<float>::max();
    // float max_error = 0;
    // float avg_error = 0;

    // int bad_count = 0;
    // int good_count = 0;

    // thrust::host_vector<float> expected = B_v;
    // thrust::host_vector<float> got = AX_v;

    // for (int i = 0; i < AX_v.size(); ++i) {
    //     const float error = std::abs(expected[i] - got[i]);
    //     if (error < min_error) {
    //         min_error = error;
    //     }
    //     if (error > max_error) {
    //         max_error = error;
    //     }
    //     avg_error += error;

    //     if (error > check_tolerance) {
    //         std::cerr << "Expected: " << expected[i] << ", Got: " << got[i]
    //                   << std::endl;
    //         ++bad_count;
    //     } else {
    //         ++good_count;
    //     }
    // }

    // std::cout << "Iterations: " << iterations << std::endl;

    // std::cout << "\nWith check_tolerance=" << check_tolerance << ':'
    //           << std::endl;
    // std::cout << "  Good values: " << good_count << std::endl;
    // std::cout << "  Bad values: " << bad_count << std::endl;

    // std::cout << "\nSummary:" << std::endl;
    // std::cout << "  min_error=" << min_error << std::endl;
    // std::cout << "  max_error=" << max_error << std::endl;
    // std::cout << "  avg_error=" << avg_error / expected.size() << std::endl;

    CUSPARSE_CHECK(cusparseDestroyDnMat(X));
    CUSPARSE_CHECK(cusparseDestroyDnMat(B));
    // CUSPARSE_CHECK(cusparseDestroyDnMat(AX));

    return 0;
}
