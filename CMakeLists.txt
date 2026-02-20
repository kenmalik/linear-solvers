cmake_minimum_required(VERSION 3.10.0)
project(CGRun VERSION 0.1.0 LANGUAGES CUDA CXX)

option(CG_RUNNER_DOWNLOAD_DEPENDENCIES "Download dependencies using FetchContent" OFF)
option(CG_RUNNER_BUILD_TESTS "Fetch testing dependencies and build tests" OFF)

find_package(CUDAToolkit 12.0 REQUIRED)

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

include(FetchContent)
FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG v3.3.1
)
FetchContent_MakeAvailable(cxxopts)

find_package(nvtx3 CONFIG)
if(NOT nvtx3_FOUND)
    if(CG_RUNNER_DOWNLOAD_DEPENDENCIES)
        FetchContent_Declare(
            nvtx3
            GIT_REPOSITORY https://github.com/NVIDIA/NVTX.git
            GIT_TAG v3.4.0
        )
        FetchContent_MakeAvailable(nvtx3)
    else()
        message(FATAL_ERROR "NVTX not found. Install it or enable CG_RUNNER_DOWNLOAD_DEPENDENCIES.")
    endif()
endif()

add_library(cg src/cg.cu)
target_compile_features(cg PRIVATE cxx_std_20 cuda_std_20)
target_include_directories(cg PUBLIC include/)
target_link_libraries(cg PUBLIC CUDA::cusparse CUDA::cublas)
target_link_libraries(cg PRIVATE nvtx3::nvtx3-cpp)

add_executable(cgrun src/main.cu)
target_compile_features(cgrun PRIVATE cxx_std_20 cuda_std_20)
target_link_libraries(cgrun cg MatUtils::mat_utils cxxopts::cxxopts nvtx3::nvtx3-cpp)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Debug build detected. Adding extra compile options")
    target_compile_options(cgrun PRIVATE "-Wall -Wextra")
endif()

set_target_properties(
    cgrun PROPERTIES
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}:${Matlab_ROOT_DIR}/bin/glnxa64"
    BUILD_WITH_INSTALL_RPATH TRUE
)

include(GNUInstallDirs)
install(TARGETS cgrun RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if (CG_RUNNER_BUILD_TESTS)
    add_subdirectory("test")
endif()
