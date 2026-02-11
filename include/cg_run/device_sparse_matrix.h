#pragma once

#include <cg_run/checks.h>

#include <mat_utils/mat_reader.h>

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cusparse_v2.h>
#include <type_traits>
#include <vector>

template <typename T>
concept FloatOrDouble = std::same_as<float, T> || std::same_as<double, T>;

namespace cg_run {
template <FloatOrDouble T> class DeviceSparseMatrix {
  public:
    constexpr static cusparseIndexType_t idxType = CUSPARSE_INDEX_64I;
    constexpr static cudaDataType valueType =
        std::is_same<T, float>::value ? CUDA_R_32F : CUDA_R_64F;

    DeviceSparseMatrix() noexcept {}

    explicit DeviceSparseMatrix(mat_utils::SpMatReader &ssm_A) noexcept {
        std::size_t min_row =
            *std::min_element(ssm_A.ir(), ssm_A.ir() + ssm_A.nnz());
        std::size_t min_col = ssm_A.jc()[0];

        {
            // For SPD, verify diagonal entries exist and are positive
            std::vector<bool> has_diag(ssm_A.rows(), false);
            std::vector<double> diag_vals;
            for (std::size_t j = 0; j < ssm_A.cols(); ++j) {
                for (std::size_t p = ssm_A.jc()[j]; p < ssm_A.jc()[j + 1];
                     ++p) {
                    std::size_t i = ssm_A.ir()[p];
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
        }

        // Use index arrays from the reader directly (no copies)
        const std::size_t *ir_ptr = ssm_A.ir();
        const std::size_t *jc_ptr = ssm_A.jc();

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_rowPtr),
                              sizeof(std::int64_t) * (ssm_A.rows() + 1)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_colInd),
                              sizeof(std::int64_t) * ssm_A.nnz()));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_vals),
                              sizeof(T) * ssm_A.nnz()));

        // Step 1: Count entries per row to build CSR row pointers
        std::vector<std::size_t> rowCounts(ssm_A.rows(), 0);
        for (std::size_t j = 0; j < ssm_A.cols(); ++j) {
            for (std::size_t p = jc_ptr[j]; p < jc_ptr[j + 1]; ++p) {
                std::size_t row = ir_ptr[p];
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
        std::vector<std::size_t> csrColInd(ssm_A.nnz());
        std::vector<T> csrVal(ssm_A.nnz());

        for (std::size_t j = 0; j < ssm_A.cols(); ++j) {
            for (std::size_t p = jc_ptr[j]; p < jc_ptr[j + 1]; ++p) {
                std::size_t row = ir_ptr[p];
                std::size_t insertPos = rowInsertPos[row]++;
                csrColInd[insertPos] = j;
                csrVal[insertPos] = static_cast<T>(ssm_A.data()[p]);
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

        CUSPARSE_CHECK(cusparseCreateCsr(
            &A, ssm_A.rows(), ssm_A.cols(), ssm_A.nnz(), d_rowPtr, d_colInd,
            d_vals, idxType, idxType, CUSPARSE_INDEX_BASE_ZERO, valueType));
    }

    DeviceSparseMatrix(std::int64_t rows, std::int64_t cols,
                       const std::vector<std::int64_t> &rowPtr,
                       const std::vector<std::int64_t> &colInd,
                       const std::vector<T> &vals) noexcept {
        CUDA_CHECK(cudaMalloc(&d_rowPtr, sizeof(std::int64_t) * rowPtr.size()));
        CUDA_CHECK(cudaMalloc(&d_colInd, sizeof(std::int64_t) * colInd.size()));
        CUDA_CHECK(cudaMalloc(&d_vals, sizeof(T) * vals.size()));

        CUDA_CHECK(cudaMemcpy(d_rowPtr, rowPtr.data(),
                              sizeof(std::int64_t) * rowPtr.size(),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_colInd, colInd.data(),
                              sizeof(std::int64_t) * colInd.size(),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vals, vals.data(), sizeof(T) * vals.size(),
                              cudaMemcpyHostToDevice));

        CUSPARSE_CHECK(cusparseCreateCsr(&A, rows, cols, vals.size(), d_rowPtr,
                                         d_colInd, d_vals, idxType, idxType,
                                         CUSPARSE_INDEX_BASE_ZERO, valueType));
    }

    DeviceSparseMatrix(const DeviceSparseMatrix &other) = delete;
    DeviceSparseMatrix &operator=(const DeviceSparseMatrix &other) = delete;

    DeviceSparseMatrix(DeviceSparseMatrix &&other)
        : d_rowPtr(other.d_rowPtr), d_colInd(other.d_colInd),
          d_vals(other.d_vals), A(other.A) {
        other.d_rowPtr = nullptr;
        other.d_colInd = nullptr;
        other.d_vals = nullptr;
        other.A = nullptr;
    }

    DeviceSparseMatrix &operator=(DeviceSparseMatrix &&other) {
        if (this != &other) {
            reset();

            d_rowPtr = other.d_rowPtr;
            d_colInd = other.d_colInd;
            d_vals = other.d_vals;
            A = other.A;

            other.d_rowPtr = nullptr;
            other.d_colInd = nullptr;
            other.d_vals = nullptr;
            other.A = nullptr;
        }
        return *this;
    }

    void reset() noexcept {
        if (A) {
            CUSPARSE_CHECK(cusparseDestroySpMat(A));
            A = nullptr;
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

    ~DeviceSparseMatrix() noexcept { reset(); }

    cusparseSpMatDescr_t &get() { return A; }

  private:
    std::int64_t *d_rowPtr = nullptr;
    std::int64_t *d_colInd = nullptr;
    T *d_vals = nullptr;
    cusparseSpMatDescr_t A = nullptr;
};
} // namespace cg_run