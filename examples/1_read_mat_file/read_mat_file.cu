#include <cassert>
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
        // Add debug prints for input matrix
        std::cout << "Input matrix (CSC format):\n";
        std::cout << "  rows: " << ssm_A.rows() << "\n";
        std::cout << "  cols: " << ssm_A.cols() << "\n";
        std::cout << "  nnz: " << ssm_A.nnz() << "\n";

        // Print first few entries of input CSC format
        std::cout << "First 5 column pointers (jc): ";
        for (size_t i = 0; i < std::min(size_t(5), ssm_A.cols() + 1); ++i) {
            std::cout << ssm_A.jc()[i] << " ";
        }
        std::cout << "\n";

        std::cout << "First 5 row indices (ir): ";
        for (size_t i = 0; i < std::min(size_t(5), ssm_A.nnz()); ++i) {
            std::cout << ssm_A.ir()[i] << " ";
        }
        std::cout << "\n";

        size_t min_row =
            *std::min_element(ssm_A.ir(), ssm_A.ir() + ssm_A.nnz());
        size_t min_col = ssm_A.jc()[0];
        bool is_one_based = (min_row == 1 || min_col == 1);
        assert(!is_one_based && "Matrix is 0 based");
        {
            // For SPD, verify diagonal entries exist and are positive
            std::vector<bool> has_diag(ssm_A.rows(), false);
            std::vector<double> diag_vals;
            for (size_t j = 0; j < ssm_A.cols(); ++j) {
                for (size_t p = ssm_A.jc()[j]; p < ssm_A.jc()[j + 1]; ++p) {
                    size_t i = ssm_A.ir()[p];
                    if (i == j) {
                        has_diag[i] = true;
                        diag_vals.push_back(ssm_A.data()[p]);
                    }
                }
            }

            int missing_diags =
                std::count(has_diag.begin(), has_diag.end(), false);
            int negative_diags =
                std::count_if(diag_vals.begin(), diag_vals.end(),
                              [](double v) { return v <= 0.0; });

            assert(missing_diags == 0 && "SPD check: no missing diagonals");
            assert(negative_diags == 0 && "SPD check: all diagonals positive");

            if (!diag_vals.empty()) {
                std::cout << "  First few diagonal values:";
                for (size_t i = 0; i < std::min(size_t(5), diag_vals.size());
                     ++i) {
                    std::cout << " " << diag_vals[i];
                }
                std::cout << "\n";
            }
        }

        // Adjust indices if 1-based
        std::vector<size_t> ir_adj(ssm_A.ir(), ssm_A.ir() + ssm_A.nnz());
        std::vector<size_t> jc_adj(ssm_A.jc(), ssm_A.jc() + ssm_A.cols() + 1);

        CUDA_CHECK(cudaMalloc(&d_rowPtr, sizeof(int64_t) * (ssm_A.rows() + 1)));
        CUDA_CHECK(cudaMalloc(&d_colInd, sizeof(int64_t) * ssm_A.nnz()));
        CUDA_CHECK(cudaMalloc(&d_vals, sizeof(float) * ssm_A.nnz()));

        // Step 1: Count entries per row to build CSR row pointers
        std::vector<size_t> rowCounts(ssm_A.rows(), 0);
        for (size_t j = 0; j < ssm_A.cols(); ++j) {
            for (size_t p = jc_adj[j]; p < jc_adj[j + 1]; ++p) {
                size_t row = ir_adj[p];
                ++rowCounts[row];
            }
        }

        // Step 2: Compute row pointer array
        std::vector<size_t> csrRowPtr(ssm_A.rows() + 1, 0);
        for (size_t i = 0; i < ssm_A.rows(); ++i) {
            csrRowPtr[i + 1] = csrRowPtr[i] + rowCounts[i];
        }

        // Step 3: Fill CSR arrays using another pass
        std::vector<size_t> rowInsertPos =
            csrRowPtr; // Current insert position for each row
        std::vector<size_t> csrColInd(ssm_A.nnz());
        std::vector<float> csrVal(ssm_A.nnz());

        for (size_t j = 0; j < ssm_A.cols(); ++j) {
            for (size_t p = jc_adj[j]; p < jc_adj[j + 1]; ++p) {
                size_t row = ir_adj[p];
                size_t insertPos = rowInsertPos[row]++;
                csrColInd[insertPos] = j;
                csrVal[insertPos] = static_cast<float>(ssm_A.data()[p]);
            }
        }

        // Debug print actual matrix structure
        std::cout << "\nVerifying CSR conversion for first few rows:\n";
        for (size_t i = 0; i < std::min(size_t(5), ssm_A.rows()); ++i) {
            std::cout << "Row " << i << " entries: ";
            for (size_t j = csrRowPtr[i]; j < csrRowPtr[i + 1]; ++j) {
                std::cout << "(" << csrColInd[j] << "," << csrVal[j] << ") ";
            }
            std::cout << "\n";
        }

        // Convert host indices to int64_t
        auto to_int64 = [](std::size_t x) { return static_cast<std::int64_t>(x); };
        std::vector<std::int64_t> csrRowPtr64(csrRowPtr.size());
        std::transform(csrRowPtr.cbegin(), csrRowPtr.cend(),
                       csrRowPtr64.begin(), to_int64);
        std::vector<std::int64_t> csrColInd64(csrColInd.size());
        std::transform(csrColInd.cbegin(), csrColInd.cend(),
                       csrColInd64.begin(), to_int64);

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
    cusparseDnMatDescr_t AX;
    thrust::device_vector<float> AX_v(n * s);
    CUSPARSE_CHECK(cusparseCreateDnMat(&AX, n, s, n,
                                       thrust::raw_pointer_cast(AX_v.data()),
                                       CUDA_R_32F, CUSPARSE_ORDER_COL));

    constexpr cusparseOperation_t transpose = CUSPARSE_OPERATION_NON_TRANSPOSE;
    constexpr float alpha = 1;
    constexpr float beta = 0;

    cusparseHandle_t cusparseH = NULL;
    CUSPARSE_CHECK(cusparseCreate(&cusparseH));

    void *buffer = nullptr;
    std::size_t buffer_size = 0;

    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        cusparseH, transpose, transpose, &alpha, A.get(), X, &beta, AX,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size));

    if (buffer_size > 0) {
        CUDA_CHECK(cudaMalloc(&buffer, buffer_size));
    }

    CUSPARSE_CHECK(cusparseSpMM(cusparseH, transpose, transpose, &alpha,
                                A.get(), X, &beta, AX, CUDA_R_32F,
                                CUSPARSE_SPMM_ALG_DEFAULT, buffer));

    if (buffer) {
        CUDA_CHECK(cudaFree(buffer));
    }

    CUSPARSE_CHECK(cusparseDestroy(cusparseH));

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

    // CUSPARSE_CHECK(cusparseDestroyDnMat(X));
    // CUSPARSE_CHECK(cusparseDestroyDnMat(B));
    // CUSPARSE_CHECK(cusparseDestroyDnMat(AX));

    return 0;
}
