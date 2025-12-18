#pragma once

#include <type_traits>

template <typename T> struct Type_info {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

    static constexpr cudaDataType_t cuda = [] {
        if constexpr (std::is_same_v<T, float>) {
            return CUDA_R_32F;
        } else {
            return CUDA_R_64F;
        }
    }();
};