#pragma once

#include <mat_utils/mat_reader.h>

#include <cg/mkl.h>

#include "parser.h"
#include "common/mkl_matrices.h"

int run_cg(const Args &args);

CSRMatrix read_mkl(const mat_utils::SpMatReader &reader);
