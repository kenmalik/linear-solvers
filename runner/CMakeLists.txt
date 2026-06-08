find_package(MatUtils CONFIG REQUIRED)
find_package(cxxopts CONFIG REQUIRED)

add_executable(cgrun cgrun.cpp parser.cpp rhs.cpp)
target_compile_features(cgrun PRIVATE cxx_std_20)
target_link_libraries(cgrun PRIVATE MatUtils::mat_utils solver_config solvers_internal)

if(SOLVERS_BUILD_CUDA OR SOLVERS_BUILD_MKL)
    target_link_libraries(cgrun PRIVATE solver_adapters)
endif()

set_target_properties(cgrun PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
install(TARGETS cgrun RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})