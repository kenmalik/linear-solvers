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
    explicit DeviceSparseMatrix(mat_utils::SpMatReader &ssm_A) {
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

        cudaMalloc(reinterpret_cast<void **>(&d_rowPtr),
                   sizeof(std::int64_t) * (ssm_A.rows() + 1));
        cudaMalloc(reinterpret_cast<void **>(&d_colInd),
                   sizeof(std::int64_t) * ssm_A.nnz());
        cudaMalloc(reinterpret_cast<void **>(&d_vals), sizeof(T) * ssm_A.nnz());

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

        cudaMemcpy(d_rowPtr, csrRowPtr64.data(),
                   sizeof(std::int64_t) * csrRowPtr64.size(),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_colInd, csrColInd64.data(),
                   sizeof(std::int64_t) * csrColInd64.size(),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_vals, csrVal.data(), sizeof(T) * csrVal.size(),
                   cudaMemcpyHostToDevice);

        cusparseIndexType_t idxType = CUSPARSE_INDEX_64I;
        cudaDataType valueType =
            std::is_same<T, float>::value ? CUDA_R_32F : CUDA_R_64F;

        cusparseCreateCsr(&A, ssm_A.rows(), ssm_A.cols(), ssm_A.nnz(), d_rowPtr,
                          d_colInd, d_vals, idxType, idxType,
                          CUSPARSE_INDEX_BASE_ZERO, valueType);
    }

    ~DeviceSparseMatrix() {
        if (A) {
            cusparseDestroySpMat(A);
        }
        if (d_rowPtr) {
            cudaFree(d_rowPtr);
            d_rowPtr = nullptr;
        }
        if (d_colInd) {
            cudaFree(d_colInd);
            d_colInd = nullptr;
        }
        if (d_vals) {
            cudaFree(d_vals);
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
} // namespace cg_run