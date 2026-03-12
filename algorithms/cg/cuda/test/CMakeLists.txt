add_executable(test_cg test_cg.cu)
target_compile_features(test_cg PRIVATE cxx_std_20 cuda_std_20)
target_link_libraries(test_cg cg MatUtils::mat_utils GTest::gtest_main)
target_compile_definitions(test_cg PRIVATE
    TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)

include(GoogleTest)
gtest_discover_tests(test_cg)
