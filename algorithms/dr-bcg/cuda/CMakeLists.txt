option(DR_BCG_BUILD_TESTS "Enable tests" OFF)
option(DR_BCG_BUILD_BENCHMARKS "Enable benchmarks" OFF)
option(DR_BCG_BUILD_EXAMPLES "Enable examples" OFF)

# include(GNUInstallDirs)

# include(CMakePackageConfigHelpers)
# configure_package_config_file(
#   cmake/DR_BCGConfig.cmake.in
#   ${CMAKE_CURRENT_BINARY_DIR}/DR_BCGConfig.cmake
#   INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DR_BCG
# )

# install(FILES
#   ${CMAKE_CURRENT_BINARY_DIR}/DR_BCGConfig.cmake
#   DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DR_BCG
# )
# install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_subdirectory("src")

if(DR_BCG_BUILD_TESTS)
    add_subdirectory("test")
endif()

if(DR_BCG_BUILD_BENCHMARKS)
    add_subdirectory("benchmark")
endif()

if(DR_BCG_BUILD_EXAMPLES)
    add_subdirectory("examples")
endif()
