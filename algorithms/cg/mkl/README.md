# mkl-cg

Conjugate Gradient (CG) and Preconditioned Conjugate Gradient (PCG) solvers
for symmetric positive-definite systems Ax = b, built on Intel MKL.

## API

All overloads return the number of iterations performed.

```cpp
// Dense A stored row-major
int cg(const std::vector<double> &A, const std::vector<double> &b,
       std::vector<double> &x, double tolerance = 1e-6,
       int max_iterations = 100);

// Sparse A in CSR format
int cg(CsrMatrix &A, const std::vector<double> &b, std::vector<double> &x,
       double tolerance = 1e-6, int max_iterations = 100);

// Sparse A loaded from a .mat file
int cg(const mat_utils::SpMatReader &A, const std::vector<double> &b,
       std::vector<double> &x, double tolerance = 1e-6,
       int max_iterations = 100);

// Preconditioned CG with incomplete Cholesky factor L
// Preconditioner M = L * L^T. Convergence criterion: ||r|| <= tolerance * ||b||
int cg(const mat_utils::SpMatReader &A, const std::vector<double> &b,
       std::vector<double> &x, const mat_utils::SpMatReader &L,
       double tolerance = 1e-6, int max_iterations = 100,
       bool real_residual = false);
```

The `real_residual` flag on the PCG overload controls how the residual is
updated each iteration. When `false` (default), `r` is updated cheaply as
`r = r - alpha * q`. When `true`, it is recomputed exactly as `r = b - A*x`,
which prevents floating-point drift at the cost of an extra sparse
matrix-vector product per iteration.

The dense and CSR overloads use an absolute convergence criterion
(`||r|| <= tolerance`). The `SpMatReader` PCG overload uses a relative
criterion (`||r|| <= tolerance * ||b||`).

## Building

```sh
cmake -B build [options]
cmake --build build
```

### CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `MKL_CG_BUILD_TESTS` | `OFF` | Build the GoogleTest test suite |
| `MKL_CG_BUILD_MAT_READERS` | `OFF` | Enable `SpMatReader`-based overloads (requires MatUtils and a MATLAB installation) |
| `MKL_CG_BUILD_CLI` | `OFF` | Build the `mkl-cgrun` CLI runner (requires `MKL_CG_BUILD_MAT_READERS`) |
| `MKL_CG_PRINT_RELATIVE_RESIDUAL_NORMS` | `OFF` | Print `\|\|r\|\|/\|\|b\|\|` to `stderr` each iteration |

### Dependencies

- **Intel MKL** (required) — discovered via CMake's `find_package(MKL)` or the
  `MKLROOT` environment variable as a fallback.
- **MatUtils** (optional) — required when `MKL_CG_BUILD_MAT_READERS=ON`.
  Provides `SpMatReader` for loading sparse matrices from MATLAB `.mat` files.
- **cxxopts** (optional) — required when `MKL_CG_BUILD_CLI=ON`.
- **GoogleTest** — fetched automatically via CMake `FetchContent` when
  `MKL_CG_BUILD_TESTS=ON`.

### Example: build everything

```sh
cmake -B build \
  -DMKL_CG_BUILD_TESTS=ON \
  -DMKL_CG_BUILD_MAT_READERS=ON \
  -DMKL_CG_BUILD_CLI=ON
cmake --build build
```

## CLI (`mkl-cgrun`)

Reads A (and optionally L) from `.mat` files, solves the system with b = ones,
and prints the iteration count.

```
mkl-cgrun <A.mat> [L.mat]
```

- `A.mat` must contain a SuiteSparse-style `Problem` struct with field `A`.
- `L.mat` (optional) must contain a variable `L` — the lower triangular
  Cholesky factor of the preconditioner. When provided, PCG is used instead of CG.
