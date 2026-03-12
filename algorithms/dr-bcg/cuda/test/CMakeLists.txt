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
