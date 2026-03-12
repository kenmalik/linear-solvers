add_library(cuda_cg cuda_cg.cu)
add_library(cg::cuda ALIAS cuda_cg)

target_include_directories(cuda_cg PUBLIC include)
target_compile_features(cuda_cg PRIVATE cxx_std_20 cuda_std_20)
target_link_libraries(cuda_cg PUBLIC CUDA::cusparse CUDA::cublas)
target_link_libraries(cuda_cg PRIVATE nvtx3::nvtx3-cpp)

# add_executable(cgrun src/main.cu)
# target_compile_features(cgrun PRIVATE cxx_std_20 cuda_std_20)
# target_link_libraries(cgrun cg MatUtils::mat_utils cxxopts::cxxopts nvtx3::nvtx3-cpp)

# if (CMAKE_BUILD_TYPE STREQUAL "Debug")
#     message(STATUS "Debug build detected. Adding extra compile options")
#     target_compile_options(cgrun PRIVATE "-Wall -Wextra")
# endif()

# set_target_properties(
#     cgrun PROPERTIES
#     INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}:${Matlab_ROOT_DIR}/bin/glnxa64"
#     BUILD_WITH_INSTALL_RPATH TRUE
# )

# include(GNUInstallDirs)
# install(TARGETS cgrun RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if (CG_RUNNER_BUILD_TESTS)
    add_subdirectory("test")
endif()
