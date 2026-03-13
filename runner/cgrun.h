#pragma once

#include <mat_utils/mat_reader.h>

#include <cg/mkl.h>

#include "parser.h"
#include "common/mkl_matrices.h"

int run_cg(const Args &args);
int run_dr_bcg(const Args &args);

// Convert a MATLAB CSC sparse matrix (as read by SpMatReader) to MKL CSR.
// MATLAB stores sparse matrices in CSC: jc[j]..jc[j+1]-1 are the entry
// indices for column j; ir holds the corresponding row indices.
// We scatter each CSC column into the appropriate CSR rows.
CSRMatrix read_mkl(const mat_utils::SpMatReader &reader);
