find_package(MatUtils CONFIG REQUIRED)

add_library(solver_adapters)

target_include_directories(solver_adapters PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_link_libraries(solver_adapters PUBLIC
    MatUtils::mat_utils solver_config solvers_internal)

if(SOLVERS_BUILD_CUDA)
    target_sources(solver_adapters PRIVATE cuda_adapter.cu)
    target_compile_features(solver_adapters PRIVATE cuda_std_20)
    if(SOLVERS_BUILD_CG)
        target_link_libraries(solver_adapters PRIVATE cg::cuda)
    endif()
    if(SOLVERS_BUILD_DR_BCG)
        target_link_libraries(solver_adapters PRIVATE dr_bcg::cuda)
        if(SOLVERS_BUILD_MATHDX)
            target_link_libraries(solver_adapters PRIVATE dr_bcg::cuda_mathdx)
        endif()
    endif()
endif()

if(SOLVERS_BUILD_MKL)
    target_sources(solver_adapters PRIVATE mkl_adapter.cpp)
    # mkl_adapter.h includes common/mkl_matrices.h -> <mkl.h>, so MKL
    # include dirs must propagate to consumers that include the header.
    target_link_libraries(solver_adapters PUBLIC MKL::MKL)
    if(SOLVERS_BUILD_CG)
        target_link_libraries(solver_adapters PRIVATE cg::mkl)
    endif()
    if(SOLVERS_BUILD_DR_BCG)
        target_link_libraries(solver_adapters PRIVATE dr_bcg::mkl)
    endif()
endif()

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

    solvers_mathdx_target_sm(_mathdx_sm)

    target_compile_definitions(cuda_dr_bcg_mathdx PRIVATE
        MATHDX_TARGET_SM=${_mathdx_sm}0)

    set_target_properties(cuda_dr_bcg_mathdx PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        INTERPROCEDURAL_OPTIMIZATION ON  # CUDA device LTO (-dlto)
        CUDA_ARCHITECTURES "${_mathdx_sm}-real")

    unset(_mathdx_sm)
endif()
