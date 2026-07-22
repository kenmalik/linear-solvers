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
