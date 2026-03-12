cmake_minimum_required(VERSION 3.13)
project(mkl_cg CXX)

# Find MKL
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

option(MKL_CG_BUILD_TESTS "Build tests" OFF)
option(MKL_CG_BUILD_MAT_READERS
    "Build CG overloads which take MatUtils MatReaders. Requires a MATLAB installation" OFF)
option(MKL_CG_BUILD_CLI "Build CLI runner" OFF)
option(MKL_CG_PRINT_RELATIVE_RESIDUAL_NORMS
    "Print relative residual norms to cerr while solving" OFF)

add_library(cg src/cg.cpp)
target_include_directories(cg PUBLIC include/)
target_compile_features(cg PRIVATE cxx_std_17)

if(MKL_CG_BUILD_MAT_READERS)
    find_package(MatUtils CONFIG REQUIRED)
    target_link_libraries(cg PUBLIC MatUtils::mat_utils)
    target_compile_definitions(cg PUBLIC USE_MAT_UTILS)
endif()

if(MKL_CG_BUILD_CLI)
    if(NOT MKL_CG_BUILD_MAT_READERS)
        message(FATAL_ERROR "CLI requires .mat file readers. Use MKL_CG_BUILD_MAT_READERS to enable them.")
    endif()
    find_package(cxxopts CONFIG REQUIRED)
    add_executable(mkl-cgrun src/cli.cpp src/parser.cpp)
    target_link_libraries(mkl-cgrun PRIVATE
        cg
        cxxopts::cxxopts
    )
endif()

if(MKL_CG_PRINT_RELATIVE_RESIDUAL_NORMS)
    target_compile_definitions(cg PRIVATE PRINT_RRN)
endif()

if(MKL_FOUND)
    target_compile_options(cg PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
    target_include_directories(cg PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(cg PUBLIC MKL::MKL)
else()
    target_include_directories(cg PUBLIC ${MKL_INCLUDE_DIR})
    target_link_libraries(cg PUBLIC
        -L${MKL_LIB_DIR}
        -lmkl_intel_lp64
        -lmkl_sequential
        -lmkl_core
        -lpthread
        -lm
        -ldl
    )
endif()

if(MKL_CG_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
