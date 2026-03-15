cmake_minimum_required(VERSION 3.10.0)
project(Solvers VERSION 0.1.0 LANGUAGES CXX CUDA)

option(SOLVERS_BUILD_CUDA "Build CUDA implementations of solvers" OFF)
option(SOLVERS_BUILD_MKL "Build MKL implementations of solvers" OFF)

option(SOLVERS_BUILD_CG "Build CG implementations" OFF)
option(SOLVERS_BUILD_DR_BCG "Build DR-BCG implementations" OFF)

option(SOLVERS_BUILD_TESTS "Build tests" OFF)
option(SOLVERS_BUILD_RUNNER "Build runner program" OFF)

option(SOLVERS_VERBOSE "Enable verbose output" OFF)
option(SOLVERS_ENABLE_TIMER "Enable timing output" OFF)

set(COMMON_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include")

if(SOLVERS_BUILD_CUDA)
    find_package(CUDAToolkit 12.0 REQUIRED)
    find_package(nvtx3 CONFIG REQUIRED)
endif()

if(SOLVERS_BUILD_MKL)
    find_package(MKL CONFIG REQUIRED)
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

if(SOLVERS_BUILD_TESTS)
    add_subdirectory(test)
endif()
