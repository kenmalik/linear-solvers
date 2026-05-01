#pragma once

#include "common/cuda_checks.h"

#include <mat_utils/mat_reader.h>

class DeviceSparseMatrix {
  public:
    explicit DeviceSparseMatrix(const mat_utils::SpMatReader &ssm_A);
    ~DeviceSparseMatrix();

    DeviceSparseMatrix(const DeviceSparseMatrix &) = delete;
    DeviceSparseMatrix &operator=(const DeviceSparseMatrix &) = delete;

    DeviceSparseMatrix(DeviceSparseMatrix &&) = delete;
    DeviceSparseMatrix &operator=(DeviceSparseMatrix &&) = delete;

    cusparseSpMatDescr_t &get() { return A; }

  private:
    int *d_rowPtr = nullptr;
    int *d_colInd = nullptr;
    double *d_vals = nullptr;
    cusparseSpMatDescr_t A;
};
