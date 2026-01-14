# CUDA Conjugate Gradient Solver

A GPU-accelerated preconditioned conjugate gradient (PCG) solver using CUDA.
The solver uses cuSPARSE and cuBLAS for sparse matrix operations and supports incomplete Cholesky preconditioning.

## Dependencies

- CUDA Toolkit 12.0+
- [MatUtils](https://github.com/kenmalik/mat-utils) - for reading matrix files
- [cxxopts](https://github.com/jarro2783/cxxopts) - for argument parsing (automatically fetched)
- CMake 3.10+
- C++20 compiler

## Building

```bash
mkdir build && cd build
cmake ..
make
```

### Build Options

| Option | Description | Default |
|--------|-------------|---------|
| `CMAKE_CUDA_ARCHITECTURES` | Target GPU architecture | `80` |
| `CG_RUNNER_DOWNLOAD_DEPENDENCIES` | Download dependencies (MatUtils) via FetchContent if not found | `OFF` |

To install:

```bash
make install
```

## Usage

```
cgrun <A> <R> [options]
```

### Positional Arguments

| Argument | Description |
|----------|-------------|
| `A` | Path to sparse matrix file (MATLAB .mat format, reads `Problem.A`) |
| `R` | Path to preconditioner file (MATLAB .mat format, reads `L` as lower triangular) |

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-h, --help` | Print help menu | |
| `--B <path>` | Path to right-hand side vector file | Vector of ones |
| `--tolerance <value>` | Convergence tolerance | `1e-6` |
| `--max-iterations <n>` | Maximum number of iterations | `n` (matrix dimension) |
| `--real-residual` | Recalculate residual as `r = b - A * x` each iteration instead of using the update formula | Off |

### Example

```bash
cgrun matrix.mat preconditioner.mat --tolerance 1e-8 --max-iterations 1000
```

## Output

The program prints the number of iterations to convergence on stdout.
