#pragma once

#include <concepts>

namespace cils {

template <typename T>
concept SupportedType = std::same_as<T, float> || std::same_as<T, double>;

}
