add_library(cuda_cg cuda_cg.cu)
add_library(cg::cuda ALIAS cuda_cg)

target_include_directories(cuda_cg PUBLIC include ${COMMON_HEADERS})
target_compile_features(cuda_cg PRIVATE cxx_std_20 cuda_std_20)
target_link_libraries(cuda_cg PRIVATE solver_config)
target_link_libraries(cuda_cg PUBLIC CUDA::cusparse CUDA::cublas)
target_link_libraries(cuda_cg PRIVATE nvtx3::nvtx3-cpp)
