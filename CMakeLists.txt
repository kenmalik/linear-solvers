cmake_minimum_required(VERSION 3.21)
project(DR_BCG VERSION 0.0.1 LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(CUDAToolkit 12.0 REQUIRED)

option(DR_BCG_BUILD_TESTS "Enable tests" OFF)
option(DR_BCG_BUILD_BENCHMARKS "Enable benchmarks" OFF)
option(DR_BCG_BUILD_EXAMPLES "Enable examples" OFF)

set(CMAKE_CUDA_ARCHITECTURES 80)

add_subdirectory("src")

if(DR_BCG_BUILD_TESTS)
    add_subdirectory("test")
endif()

if(DR_BCG_BUILD_BENCHMARKS)
    add_subdirectory("benchmark")
endif()

if(DR_BCG_BUILD_EXAMPLES)
    add_subdirectory("examples")
endif()
