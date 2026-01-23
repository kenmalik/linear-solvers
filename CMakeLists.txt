cmake_minimum_required(VERSION 3.10.0)
project(CGRun VERSION 0.1.0 LANGUAGES CUDA CXX)

option(CG_RUNNER_DOWNLOAD_DEPENDENCIES "Download dependencies using FetchContent" OFF)

add_executable(cgrun src/main.cu src/cg.cu)
target_compile_features(cgrun PRIVATE cxx_std_20 cuda_std_20)

find_package(MatUtils CONFIG)
if(NOT MatUtils_FOUND)
    if(CG_RUNNER_DOWNLOAD_DEPENDENCIES)
        include(FetchContent)
        FetchContent_Declare(
            MatUtils
            GIT_REPOSITORY https://github.com/kenmalik/mat-utils.git
            GIT_TAG v0.2.3
        )
        FetchContent_MakeAvailable(MatUtils)
    else()
        message(FATAL_ERROR "MatUtils not found. Install it or enable CG_RUNNER_DOWNLOAD_DEPENDENCIES.")
    endif()
endif()
target_link_libraries(cgrun MatUtils::mat_utils)

find_package(CUDAToolkit 12.0 REQUIRED)
target_link_libraries(cgrun CUDA::cusparse CUDA::cublas)

include(FetchContent)
FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG v3.3.1
)
FetchContent_MakeAvailable(cxxopts)
target_link_libraries(cgrun cxxopts::cxxopts)

target_include_directories(cgrun PRIVATE include/)

set_target_properties(
    cgrun PROPERTIES
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}:${Matlab_ROOT_DIR}/bin/glnxa64"
    BUILD_WITH_INSTALL_RPATH TRUE
)

include(GNUInstallDirs)
install(TARGETS cgrun RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
