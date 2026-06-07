#include "dr_bcg/cuda.cuh"

#include "common/cuda_event_timer.h"
#include "common/log.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>

#include <nvtx3/nvtx3.hpp>

namespace dr_bcg::cuda {

Handles::Handles() {
    CUSPARSE_CHECK(cusparseCreate(&cusparse));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver));
    CUSOLVER_CHECK(cusolverDnCreateParams(&cusolver_params));
    CUBLAS_CHECK(cublasCreate_v2(&cublas));
}

Handles::~Handles() {
    CUSPARSE_CHECK(cusparseDestroy(cusparse));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver));
    CUSOLVER_CHECK(cusolverDnDestroyParams(cusolver_params));
    CUBLAS_CHECK(cublasDestroy_v2(cublas));
};

void Handles::set_stream(cudaStream_t stream) {
    CUSPARSE_CHECK(cusparseSetStream(cusparse, stream));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolver, stream));
    CUBLAS_CHECK(cublasSetStream_v2(cublas, stream));
}

} // namespace dr_bcg::cuda
