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
