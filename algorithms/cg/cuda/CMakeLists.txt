add_library(cuda_cg cuda_cg.cu)
add_library(cg::cuda ALIAS cuda_cg)

target_include_directories(cuda_cg PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(cuda_cg PRIVATE solver_config solvers_internal)
target_link_libraries(cuda_cg PUBLIC CUDA::cusparse CUDA::cublas)
target_link_libraries(cuda_cg PRIVATE nvtx3::nvtx3-cpp)
