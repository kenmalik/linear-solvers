include(FetchContent)
FetchContent_Declare(
    mat_utils
    GIT_REPOSITORY https://github.com/kenmalik/mat-utils.git
    GIT_TAG v0.2.1
)
FetchContent_MakeAvailable(mat_utils)

add_executable(
    read_mat_file
    "read_mat_file.cu"
)
target_link_libraries(read_mat_file PRIVATE dr_bcg mat_utils)

# Set RPATH to prioritize CUDA over MATLAB in conflicts
set_target_properties(
    read_mat_file PROPERTIES
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}:${Matlab_ROOT_DIR}/bin/glnxa64"
    BUILD_WITH_INSTALL_RPATH TRUE
)
