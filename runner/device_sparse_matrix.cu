#include "device_sparse_matrix.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstddef>
#include <vector>

namespace {

void validate_matrix(const mat_utils::SpMatReader &ssm_A) {
    std::size_t min_row =
        *std::min_element(ssm_A.ir(), ssm_A.ir() + ssm_A.nnz());
    std::size_t min_col = ssm_A.jc()[0];
    assert((min_row != 1 && min_col != 1) &&
           "Matrix is expected to be 0 based");

    std::vector<bool> has_diag(ssm_A.rows(), false);
    std::vector<double> diag_vals;
    for (std::size_t j = 0; j < ssm_A.cols(); ++j) {
        for (std::size_t p = ssm_A.jc()[j]; p < ssm_A.jc()[j + 1]; ++p) {
            std::size_t i = ssm_A.ir()[p];
            if (i == j) {
                has_diag[i] = true;
                diag_vals.push_back(ssm_A.data()[p]);
            }
        }
    }

    int missing_diags = std::count(has_diag.begin(), has_diag.end(), false);
    int negative_diags = std::count_if(diag_vals.begin(), diag_vals.end(),
                                       [](double v) { return v <= 0.0; });

    assert(missing_diags == 0 && "SPD check: no missing diagonals");
    assert(negative_diags == 0 && "SPD check: all diagonals positive");
}

int to_int(std::size_t value) {
    assert(value <= static_cast<std::size_t>(INT_MAX) &&
           "cuSPARSE csr2cscEx2 requires 32-bit indices");
    return static_cast<int>(value);
}

} // namespace

DeviceSparseMatrix::DeviceSparseMatrix(const mat_utils::SpMatReader &ssm_A) {
    validate_matrix(ssm_A);

    const int rows = to_int(ssm_A.rows());
    const int cols = to_int(ssm_A.cols());
    const int nnz = to_int(ssm_A.nnz());

    std::vector<int> csc_col_ptr(cols + 1);
    std::transform(ssm_A.jc(), ssm_A.jc() + cols + 1, csc_col_ptr.begin(),
                   to_int);

    std::vector<int> csc_row_ind(nnz);
    std::transform(ssm_A.ir(), ssm_A.ir() + nnz, csc_row_ind.begin(), to_int);

    int *d_csc_col_ptr = nullptr;
    int *d_csc_row_ind = nullptr;
    double *d_csc_vals = nullptr;
    void *d_buffer = nullptr;

    CUDA_CHECK(cudaMalloc(&d_csc_col_ptr, sizeof(int) * (cols + 1)));
    CUDA_CHECK(cudaMalloc(&d_csc_row_ind, sizeof(int) * nnz));
    CUDA_CHECK(cudaMalloc(&d_csc_vals, sizeof(double) * nnz));

    CUDA_CHECK(cudaMemcpy(d_csc_col_ptr, csc_col_ptr.data(),
                          sizeof(int) * csc_col_ptr.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csc_row_ind, csc_row_ind.data(),
                          sizeof(int) * csc_row_ind.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csc_vals, ssm_A.data(), sizeof(double) * nnz,
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_rowPtr, sizeof(int) * (rows + 1)));
    CUDA_CHECK(cudaMalloc(&d_colInd, sizeof(int) * nnz));
    CUDA_CHECK(cudaMalloc(&d_vals, sizeof(double) * nnz));

    cusparseHandle_t cusparse = nullptr;
    CUSPARSE_CHECK(cusparseCreate(&cusparse));

    size_t buffer_size = 0;
    CUSPARSE_CHECK(cusparseCsr2cscEx2_bufferSize(cusparse, cols, rows, nnz,
                                                 d_csc_vals, d_csc_col_ptr, d_csc_row_ind,
                                                 d_vals, d_rowPtr, d_colInd,
                                                 CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
                                                 CUSPARSE_CSR2CSC_ALG1, &buffer_size));

    CUDA_CHECK(cudaMalloc(&d_buffer, buffer_size));

    CUSPARSE_CHECK(cusparseCsr2cscEx2(cusparse, cols, rows, nnz,
                                      d_csc_vals, d_csc_col_ptr, d_csc_row_ind,
                                      d_vals, d_rowPtr, d_colInd,
                                      CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
                                      CUSPARSE_CSR2CSC_ALG1, d_buffer));

    CUSPARSE_CHECK(cusparseCreateCsr(&A, rows, cols, nnz,
                                     d_rowPtr, d_colInd, d_vals,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    CUSPARSE_CHECK(cusparseDestroy(cusparse));
    CUDA_CHECK(cudaFree(d_buffer));
    CUDA_CHECK(cudaFree(d_csc_vals));
    CUDA_CHECK(cudaFree(d_csc_row_ind));
    CUDA_CHECK(cudaFree(d_csc_col_ptr));
}

DeviceSparseMatrix::~DeviceSparseMatrix() {
    if (A) {
        CUSPARSE_CHECK(cusparseDestroySpMat(A));
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
