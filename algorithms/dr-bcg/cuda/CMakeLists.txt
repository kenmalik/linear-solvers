add_library(cuda_dr_bcg INTERFACE)
add_library(dr_bcg::cuda ALIAS cuda_dr_bcg)

target_compile_features(cuda_dr_bcg INTERFACE cxx_std_20 cuda_std_20)
target_include_directories(
    cuda_dr_bcg INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(
    cuda_dr_bcg INTERFACE
    CUDA::cublas
    CUDA::cusolver
    CUDA::cusparse
    nvtx3::nvtx3-cpp
    solvers_internal
)

# MathDx instantiation is encapsulated in this dedicated static library
# with CUDA_RESOLVE_DEVICE_SYMBOLS ON, performing the device link internally
# and exposing plain C++ entry points.
#
# Downstream targets link it like usual with no device-LTO plumbing necessary.
if(SOLVERS_BUILD_MATHDX)
    if(CMAKE_VERSION VERSION_LESS 3.25)
        message(FATAL_ERROR
            "SOLVERS_BUILD_MATHDX requires CMake >= 3.25 for CUDA device LTO")
    endif()

    add_library(cuda_dr_bcg_mathdx STATIC mathdx_solve.cu)
    add_library(dr_bcg::cuda_mathdx ALIAS cuda_dr_bcg_mathdx)

    target_compile_features(cuda_dr_bcg_mathdx PRIVATE cxx_std_20 cuda_std_20)
    target_include_directories(cuda_dr_bcg_mathdx PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
    target_link_libraries(cuda_dr_bcg_mathdx
        PUBLIC cuda_dr_bcg
        PRIVATE mathdx::cublasdx mathdx::cusolverdx_fatbin solver_config)

    # cuSolverDx needs std::tuple in device code and cuBLASDx uses extended
    # lambdas internally.
    target_compile_options(cuda_dr_bcg_mathdx PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

    set_target_properties(cuda_dr_bcg_mathdx PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        INTERPROCEDURAL_OPTIMIZATION ON  # CUDA device LTO (-dlto)
        CUDA_ARCHITECTURES "80-real")
endif()
