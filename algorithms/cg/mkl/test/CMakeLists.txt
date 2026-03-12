add_executable(cg_unit_tests cg_test.cpp)
target_link_libraries(cg_unit_tests PRIVATE cg GTest::gtest_main)
target_compile_features(cg_unit_tests PRIVATE cxx_std_17)

if (MKL_CG_BUILD_MAT_READERS)
    target_link_libraries(cg_unit_tests PRIVATE MatUtils::mat_utils)
    target_compile_definitions(cg_unit_tests PRIVATE
        USE_MAT_UTILS
        DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
endif()

include(GoogleTest)
gtest_discover_tests(cg_unit_tests)
