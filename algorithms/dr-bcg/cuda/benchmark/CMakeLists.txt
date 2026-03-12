set(BENCHMARK_ENABLE_TESTING OFF)
set(CMAKE_BUILD_TYPE "Release")

include(FetchContent)
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.4
)
FetchContent_MakeAvailable(benchmark)

add_executable(
    dr_bcg_benchmark
    "dr_bcg_benchmark.cu"
)
target_link_libraries(
    dr_bcg_benchmark PRIVATE
    benchmark::benchmark
    DR_BCG::dr_bcg
)
