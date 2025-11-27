option(DR_BCG_USE_THIN_QR "Use thin QR procedure instead of standard QR factorization" OFF)

include(FetchContent)
FetchContent_Declare(
    nvtx
    GIT_REPOSITORY https://github.com/NVIDIA/NVTX.git
    GIT_TAG v3.2.1
)
FetchContent_MakeAvailable(nvtx)

add_library(dr_bcg
    "dr_bcg.cu"
    "helper.cu"
    "sparse.cu"
    "dense.cu"
)
target_include_directories(dr_bcg PUBLIC "${CMAKE_SOURCE_DIR}/include")

target_link_libraries(
    dr_bcg PUBLIC
    CUDA::cublas
    CUDA::cusolver
    CUDA::cusparse
    nvtx3-cpp
)

if (DR_BCG_USE_THIN_QR)
    target_compile_definitions(dr_bcg PUBLIC "USE_THIN_QR")
endif()

target_compile_options(dr_bcg PUBLIC
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

target_compile_definitions(dr_bcg PRIVATE $<$<CONFIG:Debug>:CUDA_DR_BCG_DEBUG_BUILD>)