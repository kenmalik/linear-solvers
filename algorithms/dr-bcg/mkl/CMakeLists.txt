add_library(mkl_dr_bcg mkl_dr_bcg.cpp)
add_library(dr_bcg::mkl ALIAS mkl_dr_bcg)

target_include_directories(mkl_dr_bcg PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(mkl_dr_bcg PRIVATE solver_config solvers_internal)
target_link_libraries(mkl_dr_bcg PUBLIC MKL::MKL)
