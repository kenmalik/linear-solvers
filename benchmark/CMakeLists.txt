find_package(MatUtils VERSION 0.4.0 CONFIG REQUIRED)

include(FetchContent)

FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.5
    SYSTEM
)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)
set_target_properties(benchmark benchmark_main PROPERTIES
    CXX_CLANG_TIDY "" CXX_CPPCHECK "")

add_executable(solver_benchmarks solver_benchmark.cpp)
target_link_libraries(solver_benchmarks PRIVATE
    benchmark::benchmark_main MatUtils::mat_utils solver_config)
target_compile_definitions(solver_benchmarks PRIVATE
    BENCHMARK_DATA_DIR="${PROJECT_SOURCE_DIR}/data")

if(SOLVERS_BUILD_CUDA OR SOLVERS_BUILD_MKL)
    target_link_libraries(solver_benchmarks PRIVATE solver_adapters)
endif()
