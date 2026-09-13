add_library(cuda_dr_bcg INTERFACE)
add_library(dr_bcg::cuda ALIAS cuda_dr_bcg)

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

# Dev-only anchor TU so clangd can resolve headers in this INTERFACE-only
# target (see clangd_anchor.cu for why). EXCLUDE_FROM_ALL keeps it out of
# `ninja`/`make` by default -- it still gets a compile_commands.json entry,
# which is all clangd needs, without ever being linked into a real binary.
# Skipped outside Debug so CI/release configures don't pay for it.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_library(cuda_dr_bcg_clangd_anchor OBJECT EXCLUDE_FROM_ALL clangd_anchor.cu)
    target_link_libraries(cuda_dr_bcg_clangd_anchor PRIVATE cuda_dr_bcg solver_config)

    if(SOLVERS_BUILD_MATHDX)
        solvers_mathdx_target_sm(_anchor_mathdx_sm)
        target_link_libraries(cuda_dr_bcg_clangd_anchor PRIVATE
            mathdx::cublasdx mathdx::cusolverdx_fatbin)
        target_compile_definitions(cuda_dr_bcg_clangd_anchor PRIVATE
            MATHDX_TARGET_SM=${_anchor_mathdx_sm}0)
        target_compile_options(cuda_dr_bcg_clangd_anchor PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
            $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
        unset(_anchor_mathdx_sm)
    endif()
endif()

