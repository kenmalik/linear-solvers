add_library(
    dr_bcg
    "helper.cu"
    "sparse.cu"
    "dense.cu"
)
add_library(DR_BCG::dr_bcg ALIAS dr_bcg)

target_compile_features(dr_bcg PRIVATE cxx_std_17)
target_include_directories(
    dr_bcg PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(
    dr_bcg PUBLIC
    CUDA::cublas
    CUDA::cusolver
    CUDA::cusparse
    nvtx3::nvtx3-cpp
)

target_compile_definitions(
    dr_bcg PRIVATE
    $<$<CONFIG:Debug>:CUDA_DR_BCG_DEBUG_BUILD>
)

# install(
#     TARGETS dr_bcg nvtx3-c nvtx3-cpp
#     EXPORT DR_BCGTargets
#     ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
# )
# install(
#     EXPORT DR_BCGTargets
#     NAMESPACE DR_BCG::
#     DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DR_BCG
# )
