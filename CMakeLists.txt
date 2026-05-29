cmake_minimum_required(VERSION 3.18)
project(Solvers VERSION 0.1.0 LANGUAGES CXX CUDA)

include(GNUInstallDirs)

option(SOLVERS_BUILD_CUDA "Build CUDA implementations of solvers" ON)
option(SOLVERS_BUILD_MKL "Build MKL implementations of solvers" OFF)

option(SOLVERS_BUILD_DR_BCG "Build DR-BCG implementations" ON)
option(SOLVERS_BUILD_CG "Build CG implementations" OFF)

option(SOLVERS_BUILD_TESTS "Build tests" ON)
option(SOLVERS_BUILD_RUNNER "Build runner program" ON)
option(SOLVERS_BUILD_BENCHMARKS "Build benchmark programs" OFF)

option(SOLVERS_ENABLE_VERBOSE "Enable verbose output" OFF)
option(SOLVERS_ENABLE_TIMER "Enable timing output" OFF)

configure_file(
    ${CMAKE_SOURCE_DIR}/include/config.h.in
    ${CMAKE_BINARY_DIR}/include/config.h
)
add_library(solver_config INTERFACE)
target_include_directories(solver_config INTERFACE ${CMAKE_BINARY_DIR}/include)

add_library(solvers_internal INTERFACE)
target_include_directories(solvers_internal INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

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

if(SOLVERS_BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()
