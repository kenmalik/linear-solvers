option(MKL_CG_BUILD_MAT_READERS
    "Build CG overloads which take MatUtils MatReaders. Requires a MATLAB installation" OFF)
option(MKL_CG_BUILD_CLI "Build CLI runner" OFF)
option(MKL_CG_PRINT_RELATIVE_RESIDUAL_NORMS
    "Print relative residual norms to cerr while solving" OFF)

add_library(mkl_cg mkl_cg.cpp)
add_library(cg::mkl ALIAS mkl_cg)

target_include_directories(mkl_cg PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(mkl_cg PRIVATE cxx_std_17)
target_link_libraries(mkl_cg PRIVATE solver_config solvers_internal)
target_link_libraries(mkl_cg PUBLIC MKL::MKL)
