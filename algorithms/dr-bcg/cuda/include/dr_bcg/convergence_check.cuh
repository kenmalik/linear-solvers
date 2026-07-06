#pragma once

#include "dr_bcg/device_buffer.cuh"
#include "dr_bcg/handles.cuh"

#include "common/cuda_checks.h"
#include "common/cuda_event_timer.h"
#include "common/log.h"
#include "common/type_info.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <cstddef>

namespace dr_bcg::cuda {

// Check if matrix converged using the sigma convergence check
template <SupportedType T>
bool check_convergence(Handles &handles, DeviceBuffer<T> &d, T tolerance,
                       T sigma_norm0, T *&d_sigma_norm, std::int64_t s, cudaStream_t stream) {
    constexpr int incx = 1;

    nvtx3::scoped_range sigma_norm_range{"||sigma_1||"};
    CudaTimerRange er{g_event_timer, "||sigma_1||", stream};

    CUBLAS_CHECK(cublasDnrm2_v2(handles.cublas, s, d.sigma, incx, d_sigma_norm));

    T sigma_norm = 0;
    CUDA_CHECK(cudaMemcpyAsync(&sigma_norm, d_sigma_norm, sizeof(T),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cils::log(sigma_norm / sigma_norm0);
    return sigma_norm / sigma_norm0 < tolerance;
}

}