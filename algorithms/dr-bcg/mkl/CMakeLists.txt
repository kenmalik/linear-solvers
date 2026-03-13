add_library(mkl_dr_bcg mkl_dr_bcg.cpp)
add_library(dr_bcg::mkl ALIAS mkl_dr_bcg)

target_include_directories(mkl_dr_bcg PUBLIC include)

target_compile_options(mkl_dr_bcg PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(mkl_dr_bcg PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(mkl_dr_bcg PUBLIC MKL::MKL)
