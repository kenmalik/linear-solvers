if(SOLVERS_BUILD_CUDA)
    add_subdirectory(cuda)
endif()

if(SOLVERS_BUILD_MKL)
    add_subdirectory(mkl)
endif()
