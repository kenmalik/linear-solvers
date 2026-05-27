#pragma once

#include <concepts>
#include <type_traits>

template <typename T>
concept SupportedType = std::same_as<T, float> || std::same_as<T, double>;

template <SupportedType T>
struct TypeInfo {
    static constexpr cudaDataType_t cuda = [] {
        if constexpr (std::is_same_v<T, float>) {
            return CUDA_R_32F;
        } else {
            return CUDA_R_64F;
        }
    }();
};