#pragma once

#include "supported_type.h"

#include <cuda_runtime.h>

template <cils::SupportedType T>
inline constexpr cudaDataType_t cuda_type = [] {
    if constexpr (std::is_same_v<T, float>) {
        return CUDA_R_32F;
    } else {
        return CUDA_R_64F;
    }
}();
