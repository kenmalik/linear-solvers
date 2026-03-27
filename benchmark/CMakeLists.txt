find_package(MatUtils CONFIG REQUIRED)

include(FetchContent)

FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.5
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

add_executable(solver_benchmarks solver_benchmark.cpp)
target_compile_features(solver_benchmarks PRIVATE cxx_std_20)
target_include_directories(solver_benchmarks PRIVATE
    ${PROJECT_SOURCE_DIR}/runner
    ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(solver_benchmarks PRIVATE benchmark::benchmark_main MatUtils::mat_utils)
target_compile_definitions(solver_benchmarks PRIVATE
    BENCHMARK_DATA_DIR="${PROJECT_SOURCE_DIR}/data")

if(SOLVERS_BUILD_MKL)
    target_sources(solver_benchmarks PRIVATE ${PROJECT_SOURCE_DIR}/runner/mkl_adapter.cpp)
    target_compile_definitions(solver_benchmarks PRIVATE MKL_ENABLED)
    if(SOLVERS_BUILD_CG)
        target_link_libraries(solver_benchmarks PRIVATE cg::mkl)
    endif()
    if(SOLVERS_BUILD_DR_BCG)
        target_link_libraries(solver_benchmarks PRIVATE dr_bcg::mkl)
    endif()
endif()

if(SOLVERS_BUILD_CUDA)
    target_sources(solver_benchmarks PRIVATE ${PROJECT_SOURCE_DIR}/runner/cuda_adapter.cu)
    target_compile_definitions(solver_benchmarks PRIVATE CUDA_ENABLED)
    set_target_properties(solver_benchmarks PROPERTIES CUDA_STANDARD 20)
    if(SOLVERS_BUILD_CG)
        target_link_libraries(solver_benchmarks PRIVATE cg::cuda)
    endif()
    if(SOLVERS_BUILD_DR_BCG)
        target_link_libraries(solver_benchmarks PRIVATE dr_bcg::cuda)
    endif()
endif()
