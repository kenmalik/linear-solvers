option(MKL_CG_BUILD_TESTS "Build tests" OFF)
option(MKL_CG_BUILD_MAT_READERS
    "Build CG overloads which take MatUtils MatReaders. Requires a MATLAB installation" OFF)
option(MKL_CG_BUILD_CLI "Build CLI runner" OFF)
option(MKL_CG_PRINT_RELATIVE_RESIDUAL_NORMS
    "Print relative residual norms to cerr while solving" OFF)

add_library(mkl_cg mkl_cg.cpp)
add_library(cg::mkl ALIAS mkl_cg)

target_include_directories(mkl_cg PUBLIC include)
target_compile_features(mkl_cg PRIVATE cxx_std_17)

if(MKL_CG_PRINT_RELATIVE_RESIDUAL_NORMS)
    target_compile_definitions(mkl_cg PRIVATE PRINT_RRN)
endif()

target_compile_options(mkl_cg PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(mkl_cg PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(mkl_cg PUBLIC MKL::MKL MatUtils::mat_utils)

if(MKL_CG_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
