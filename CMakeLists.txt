cmake_minimum_required(VERSION 3.10.0)
project(Solvers VERSION 0.1.0)

option(SOLVERS_BUILD_CUDA "Build CUDA implementations of solvers" OFF)
option(SOLVERS_BUILD_MKL "Build MKL implementations of solvers" OFF)

option(SOLVERS_BUILD_CG "Build CG implementations" OFF)
option(SOLVERS_BUILD_DR_BCG "Build DR-BCG implementations" OFF)

option(SOLVERS_BUILD_RUNNER "Build runner program" OFF)

find_package(MatUtils CONFIG REQUIRED)
find_package(cxxopts CONFIG REQUIRED)

if(SOLVERS_BUILD_CUDA)
    find_package(CUDAToolkit 12.0 REQUIRED)
endif()

if(SOLVERS_BUILD_MKL)
    find_package(MKL CONFIG)
    if(MKL_FOUND)
        message(STATUS "Found MKL: ${MKL_ROOT}")
    else()
        # Fallback: try to find MKL manually
        set(MKL_ROOT $ENV{MKLROOT})
        if(NOT MKL_ROOT)
            message(FATAL_ERROR "MKL not found. Set MKLROOT environment variable.")
        endif()
        set(MKL_INCLUDE_DIR ${MKL_ROOT}/include)
        set(MKL_LIB_DIR ${MKL_ROOT}/lib/intel64)
    endif()
endif()

if(SOLVERS_BUILD_CG)
    add_subdirectory(algorithms/cg)
endif()

if(SOLVERS_BUILD_DR_BCG)
    add_subdirectory(algorithms/dr-bcg)
endif()

if(SOLVERS_BUILD_RUNNER)
    add_subdirectory(runner)
endif()
