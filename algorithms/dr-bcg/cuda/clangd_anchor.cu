// This translation unit exists purely so clang tooling (compile_commands.json)
// has a real, correctly-flagged entry inside this directory. dr_bcg::cuda is
// an INTERFACE-only target with no .cu source of its own, so headers here get
// no compile command; clangd then falls back to whatever unrelated TU is
// "closest" by directory (e.g. the plain-C++ MKL sibling target), which has
// none of the CUDA/cuBLAS/MathDx flags these headers need, and editing them
// shows bogus "file not found" errors.
//
// Never linked into anything -- see algorithms/dr-bcg/cuda/CMakeLists.txt.

#include "config.h"

#include "dr_bcg/cuda.cuh"
#include "dr_bcg/mathdx_fused.cuh"
