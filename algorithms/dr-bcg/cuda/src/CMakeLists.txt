add_library(
    cuda_dr_bcg
    "cuda_dr_bcg.cu"
    "helper.cu"
)
add_library(dr_bcg::cuda ALIAS cuda_dr_bcg)

target_compile_features(cuda_dr_bcg PRIVATE cxx_std_17)
target_include_directories(
    cuda_dr_bcg PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    ${COMMON_HEADERS}
)

target_link_libraries(
    cuda_dr_bcg PUBLIC
    CUDA::cublas
    CUDA::cusolver
    CUDA::cusparse
    nvtx3::nvtx3-cpp
)

target_compile_definitions(
    cuda_dr_bcg PRIVATE
    $<$<CONFIG:Debug>:CUDA_DR_BCG_DEBUG_BUILD>
    $<$<BOOL:SOLVERS_VERBOSE>:SOLVERS_VERBOSE>
)
