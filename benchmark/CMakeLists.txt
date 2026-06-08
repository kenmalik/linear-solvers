find_package(MatUtils CONFIG REQUIRED)

include(FetchContent)

FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.5
)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

add_executable(solver_benchmarks solver_benchmark.cpp)
target_compile_features(solver_benchmarks PRIVATE cxx_std_20)
target_link_libraries(solver_benchmarks PRIVATE benchmark::benchmark_main MatUtils::mat_utils solver_config)
target_compile_definitions(solver_benchmarks PRIVATE
    BENCHMARK_DATA_DIR="${PROJECT_SOURCE_DIR}/data")

if(SOLVERS_BUILD_CUDA OR SOLVERS_BUILD_MKL)
    target_link_libraries(solver_benchmarks PRIVATE solver_adapters)
endif()
