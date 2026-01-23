include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
    dr_bcg_test
    "dr_bcg_test.cu"
)
target_link_libraries(
    dr_bcg_test
    GTest::gtest_main
    DR_BCG::dr_bcg
)

include(GoogleTest)
gtest_discover_tests(dr_bcg_test)
