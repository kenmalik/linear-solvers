find_package(MatUtils CONFIG REQUIRED)
find_package(cxxopts CONFIG REQUIRED)

add_executable(cgrun cgrun.cpp)

if(SOLVERS_BUILD_CUDA)
    if(SOLVERS_BUILD_CG)
        target_link_libraries(cgrun PRIVATE cg::cuda)
        target_compile_definitions(cgrun PRIVATE CUDA_CG_ENABLED)
    endif()
endif()
