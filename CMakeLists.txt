cmake_minimum_required(VERSION 3.10.0)
project(CGRun VERSION 0.1.0 LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 80)

add_executable(cgrun src/main.cu src/cg.cu)

find_package(MatUtils CONFIG REQUIRED)
target_link_libraries(cgrun MatUtils::mat_utils)

find_package(CUDAToolkit 12.0 REQUIRED)
target_link_libraries(cgrun CUDA::cusparse CUDA::cublas)

target_include_directories(cgrun PRIVATE include/)

include(GNUInstallDirs)
install(TARGETS cgrun RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})