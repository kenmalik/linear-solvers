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

add_library(cg src/cg.cpp)
target_include_directories(cg PUBLIC include/)
target_compile_features(cg PRIVATE cxx_std_17)
add_executable(cg_test src/main.cpp)
target_link_libraries(cg_test PRIVATE cg)

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
