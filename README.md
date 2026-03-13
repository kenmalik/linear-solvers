# Linear Solvers

## Introduction

This repo contains a collection of linear solver implementations. The primary
solver of interest is the [CUDA DR-BCG](algorithms/dr-bcg/cuda); the other
solvers serve as points of comparison to evaluate DR-BCG's performance.

Currently, there are two algorithms implemented:

- CG 
- DR-BCG

Each algorithm has implementations using the following technologies:

- Nvidia's CUDA APIs (cuBLAS, cuSOLVER, cuSPARSE)
- Intel's Math Kernel Library (MKL)

## Current Status

**This repo is currently under development**. The various solvers were initially
implemented in separate repos, and we are currently in the process of transferring
files and updating them to work together.

## Building

Building the project requires CMake. The following build options are provided:

| Build Option | Description |
| - | - |
| `SOLVERS_BUILD_CUDA` | Build CUDA implementations of solvers |
| `SOLVERS_BUILD_MKL` | Build MKL implementations of solvers |
| `SOLVERS_BUILD_CG` | Build CG implementations |
| `SOLVERS_BUILD_DR_BCG` | Build DR-BCG implementations |
| `SOLVERS_BUILD_TESTS` | Build tests |
| `SOLVERS_BUILD_RUNNER` | Build runner program |

All options have the value `OFF` by default.

As an example, you can build the runner program with CUDA and MKL DR-BCG like so:

```shell
cmake -B build -S . -DSOLVERS_BUILD_CUDA=ON -DSOLVERS_BUILD_MKL=ON -DSOLVERS_BUILD_DR_BCG=ON -DSOLVERS_BUILD_RUNNER=ON
cmake --build build
```

> **Note:** If building the CUDA implementations, you must also define the
> `CMAKE_CUDA_ARCHITECTURES` option. For more information, see
> [Nvidia's compute capability chart](https://developer.nvidia.com/cuda/gpus).

e.g. If building for a GeForce RTX 2070

```shell
cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES=75 -DSOLVERS_BUILD_CUDA=ON -DSOLVERS_BUILD_DR_BCG=ON -DSOLVERS_BUILD_RUNNER=ON
cmake --build build
```
