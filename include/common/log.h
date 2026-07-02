#pragma once

#include "config.h"

#include <iostream>

namespace cils {

template <typename T, bool Enabled = logging_enabled>
    requires Enabled
void log(T v) {
    std::cerr << v << '\n';
}

template <typename T, bool Enabled = logging_enabled>
    requires(!Enabled)
void log(T v) {}

}