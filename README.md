# Linear Solvers

> [!NOTE]
> **This repo is currently under development**. The various solvers were initially
> implemented in separate repos, and we are currently in the process of transferring
> files and updating their documentation.

## Introduction

This repo contains a collection of linear solver implementations. The primary
solver of interest is [CUDA DR-BCG](algorithms/dr-bcg/cuda); the other
solvers serve as baselines to evaluate DR-BCG's performance.

The project implements the following algorithms:

- (Preconditioned) CG 
- (Preconditioned) DR-BCG

Each algorithm has implementations using the following technologies:

- Nvidia's CUDA APIs (cuBLAS, cuSOLVER, cuSPARSE)
- Intel's Math Kernel Library (MKL)

## Building

### Dependencies

Building the project requires **CMake**.

Depending on which parts of the project you would like to build, you would need the following dependencies:

| Feature | Dependencies |
| - | - |
| CUDA solvers | [CUDA Toolkit](https://developer.nvidia.com/cuda/toolkit) 12.0 |
| CUDA solvers (with fused kernels) | [NVIDIA MathDx](https://docs.nvidia.com/cuda/mathdx/) 25.12 |
| MKL solvers | [Intel oneMKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) |
| Solver Runner CLI | [MatUtils](https://github.com/kenmalik/mat-utils), [cxxopts](https://github.com/jarro2783/cxxopts) |

If building tests or benchmarks, CMake will automatically fetch [Google Test](https://github.com/google/googletest)
and [Google Benchmark](https://github.com/google/benchmark), respectively.

### Options

The following build options are provided:

| Build Option | Description |
| - | - |
| `SOLVERS_BUILD_CUDA` | Build CUDA implementations of solvers |
| `SOLVERS_BUILD_MKL` | Build MKL implementations of solvers |
| `SOLVERS_BUILD_CG` | Build CG implementations |
| `SOLVERS_BUILD_DR_BCG` | Build DR-BCG implementations |
| `SOLVERS_BUILD_TESTS` | Build tests |
| `SOLVERS_BUILD_RUNNER` | Build runner program |

As an example, you can build the runner program with CUDA and MKL DR-BCG like so:

```shell
cmake -B build -S . -DSOLVERS_BUILD_CUDA=ON -DSOLVERS_BUILD_MKL=ON -DSOLVERS_BUILD_DR_BCG=ON -DSOLVERS_BUILD_RUNNER=ON
cmake --build build
```

> [!IMPORTANT]
> If building the CUDA implementations, you must also define the
> `CMAKE_CUDA_ARCHITECTURES` option. For more information, see
> [Nvidia's compute capability chart](https://developer.nvidia.com/cuda/gpus).

e.g. If building for a GeForce RTX 2070

```shell
cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES=75 -DSOLVERS_BUILD_CUDA=ON -DSOLVERS_BUILD_DR_BCG=ON -DSOLVERS_BUILD_RUNNER=ON
cmake --build build
```

## Developing

This project was developed on an HPC cluster using Lmod.

Lmod's module system tends to break include paths and CUDA tends to break CMake's `compile_commands.json`.
Hence, we have a script to set up VSCode's IntelliSense include paths. Use it like so:

```shell
module load intel-oneapi-mkl  # Ensure the MKL module is loaded
make vscode-config
```

The `vscode-config` Make target calls [this script](scripts/vscode_config.sh). 
