#pragma once

#include "common/cuda_checks.h"
#include "common/cuda_type.cuh"

#include <mat_utils/mat_reader.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace cils::detail {

template <cils::detail::SupportedType T>
class DeviceSparseMatrix {
  public:
    explicit DeviceSparseMatrix(const mat_utils::MatReader<mat_utils::Sparsity::Sparse> &ssm_A) {
        std::span<std::size_t> jc = ssm_A.column_pointers();
        std::span<std::size_t> ir = ssm_A.row_indices();
        std::span<T> values = ssm_A.values<T>();

        std::size_t min_row =
            *std::ranges::min_element(ssm_A.row_indices());
        std::size_t min_col = jc[0];
        assert(min_row != 1 && min_col != 1 && "Matrix is expected to be 0 based");

        {
            // For SPD, verify diagonal entries exist and are positive
            std::vector<bool> has_diag(ssm_A.rows(), false);
            std::vector<double> diag_vals;
            for (std::size_t j = 0; j < ssm_A.cols(); ++j) {
                for (std::size_t p = jc[j]; p < jc[j + 1];
                     ++p) {
                    std::size_t i = ir[p];
                    if (i == j) {
                        has_diag[i] = true;
                        diag_vals.push_back(values[p]);
                    }
                }
            }

            auto missing_diags =
                std::count(has_diag.begin(), has_diag.end(), false);
            int negative_diags =
                std::count_if(diag_vals.begin(), diag_vals.end(),
                              [](double v) { return v <= 0.0; });

            assert(missing_diags == 0 && "SPD check: no missing diagonals");
            assert(negative_diags == 0 && "SPD check: all diagonals positive");
        }

        CUDA_CHECK(
            cudaMalloc(&d_rowPtr, sizeof(std::int64_t) * (ssm_A.rows() + 1)));
        CUDA_CHECK(cudaMalloc(&d_colInd, sizeof(std::int64_t) * ssm_A.nonzero_count()));
        CUDA_CHECK(cudaMalloc(&d_vals, sizeof(T) * ssm_A.nonzero_count()));

        // Step 1: Count entries per row to build CSR row pointers
        std::vector<std::size_t> rowCounts(ssm_A.rows(), 0);
        for (std::size_t j = 0; j < ssm_A.cols(); ++j) {
            for (std::size_t p = jc[j]; p < jc[j + 1]; ++p) {
                std::size_t row = ir[p];
                ++rowCounts[row];
            }
        }

        // Step 2: Compute row pointer array
        std::vector<std::size_t> csrRowPtr(ssm_A.rows() + 1, 0);
        for (std::size_t i = 0; i < ssm_A.rows(); ++i) {
            csrRowPtr[i + 1] = csrRowPtr[i] + rowCounts[i];
        }

        // Step 3: Fill CSR arrays using another pass
        std::vector<std::size_t> rowInsertPos =
            csrRowPtr; // Current insert position for each row
        std::vector<std::size_t> csrColInd(ssm_A.nonzero_count());
        std::vector<T> csrVal(ssm_A.nonzero_count());

        for (std::size_t j = 0; j < ssm_A.cols(); ++j) {
            for (std::size_t p = jc[j]; p < jc[j + 1]; ++p) {
                std::size_t row = ir[p];
                std::size_t insertPos = rowInsertPos[row]++;
                csrColInd[insertPos] = j;
                csrVal[insertPos] = static_cast<T>(values[p]);
            }
        }

        // Convert host indices to int64_t
        auto to_int64 = [](std::size_t x) {
            return static_cast<std::int64_t>(x);
        };
        std::vector<std::int64_t> csrRowPtr64(csrRowPtr.size());
        std::transform(csrRowPtr.cbegin(), csrRowPtr.cend(),
                       csrRowPtr64.begin(), to_int64);
        std::vector<std::int64_t> csrColInd64(csrColInd.size());
        std::transform(csrColInd.cbegin(), csrColInd.cend(),
                       csrColInd64.begin(), to_int64);

        CUDA_CHECK(cudaMemcpy(d_rowPtr, csrRowPtr64.data(),
                              sizeof(std::int64_t) * csrRowPtr64.size(),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_colInd, csrColInd64.data(),
                              sizeof(std::int64_t) * csrColInd64.size(),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vals, csrVal.data(), sizeof(T) * csrVal.size(),
                              cudaMemcpyHostToDevice));

        constexpr cusparseIndexType_t idxType = CUSPARSE_INDEX_64I;
        CUSPARSE_CHECK(cusparseCreateCsr(
            &A, ssm_A.rows(), ssm_A.cols(), ssm_A.nonzero_count(), d_rowPtr, d_colInd,
            d_vals, idxType, idxType, CUSPARSE_INDEX_BASE_ZERO, cils::detail::cuda_type<T>));
    }

    DeviceSparseMatrix(const DeviceSparseMatrix &) = delete;
    DeviceSparseMatrix &operator=(const DeviceSparseMatrix &) = delete;
    DeviceSparseMatrix(DeviceSparseMatrix &&) = delete;
    DeviceSparseMatrix &operator=(DeviceSparseMatrix &&) = delete;

    ~DeviceSparseMatrix() {
        if (A != nullptr) {
            CUSPARSE_CHECK(cusparseDestroySpMat(A));
        }
        if (d_rowPtr != nullptr) {
            CUDA_CHECK(cudaFree(d_rowPtr));
            d_rowPtr = nullptr;
        }
        if (d_colInd != nullptr) {
            CUDA_CHECK(cudaFree(d_colInd));
            d_colInd = nullptr;
        }
        if (d_vals != nullptr) {
            CUDA_CHECK(cudaFree(d_vals));
            d_vals = nullptr;
        }
    }

    cusparseSpMatDescr_t &get() { return A; }

  private:
    std::int64_t *d_rowPtr = nullptr;
    std::int64_t *d_colInd = nullptr;
    T *d_vals = nullptr;
    cusparseSpMatDescr_t A{};
};

using DeviceSparseMatrixFloat = DeviceSparseMatrix<float>;
using DeviceSparseMatrixDouble = DeviceSparseMatrix<double>;

} // namespace cils::detail
