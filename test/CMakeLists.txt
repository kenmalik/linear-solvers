find_package(MatUtils CONFIG REQUIRED)
find_package(cxxopts CONFIG REQUIRED)

include(FetchContent)

FetchContent_declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(solver_tests solver_test.cpp
    ${CMAKE_SOURCE_DIR}/runner/parser.cpp
    ${CMAKE_SOURCE_DIR}/runner/rhs.cpp)
target_compile_features(solver_tests PRIVATE cxx_std_20)
target_include_directories(solver_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/runner
    ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(solver_tests PRIVATE GTest::gtest_main MatUtils::mat_utils
    cxxopts::cxxopts)
target_compile_definitions(solver_tests PRIVATE
    TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/data")
target_compile_definitions(solver_tests PRIVATE
    $<$<BOOL:${SOLVERS_BUILD_MKL}>:MKL_ENABLED>
    $<$<BOOL:${SOLVERS_BUILD_CUDA}>:CUDA_ENABLED>
    $<$<BOOL:${SOLVERS_BUILD_CG}>:CG_ENABLED>
    $<$<BOOL:${SOLVERS_BUILD_DR_BCG}>:DR_BCG_ENABLED>)

if(SOLVERS_BUILD_MKL)
    target_sources(solver_tests PRIVATE ${CMAKE_SOURCE_DIR}/runner/mkl_adapter.cpp)
    if(SOLVERS_BUILD_CG)
        target_link_libraries(solver_tests PRIVATE cg::mkl)
    endif()
    if(SOLVERS_BUILD_DR_BCG)
        target_link_libraries(solver_tests PRIVATE dr_bcg::mkl)
    endif()
endif()

if(SOLVERS_BUILD_CUDA)
    target_sources(solver_tests PRIVATE ${CMAKE_SOURCE_DIR}/runner/cuda_adapter.cu)
    set_target_properties(solver_tests PROPERTIES CUDA_STANDARD 20)
    if(SOLVERS_BUILD_CG)
        target_link_libraries(solver_tests PRIVATE cg::cuda)
    endif()
    if(SOLVERS_BUILD_DR_BCG)
        target_link_libraries(solver_tests PRIVATE dr_bcg::cuda)
    endif()
endif()

include(GoogleTest)
gtest_discover_tests(solver_tests)
