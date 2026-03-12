find_package(MatUtils CONFIG REQUIRED)
find_package(CUDAToolkit 12.0 REQUIRED)

if(SOLVERS_BUILD_CUDA)
    add_subdirectory(cuda)
endif()

if(SOLVERS_BUILD_MKL)
    add_subdirectory(mkl)
endif()
