#pragma once

#include <mat_utils/mat_reader.h>

#include <cg/mkl.h>

#include "parser.h"

int run_cg(const Args &args);

cg::mkl::MKLSparse read_mkl(const mat_utils::SpMatReader &reader);
