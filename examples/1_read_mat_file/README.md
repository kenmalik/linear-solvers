# Example 1: Reading MAT Files

This example runs DR-BCG using the sparse or dense matrix interface. The `n` by `n` matrix `A` is read from a `.mat` file containing a sparse CSC matrix. The initial `X` guess is set to an `n` by `s` matrix of zeros. `B` is set to an `n` by `s` matrix of ones.

## Running

Assuming you are in the `build/` directory, run the example binary with the sparse `.mat` file path and optional block size `s`.

By default, the solver runs in single-precision. This can be changed with the `--double` option.

### Basic usage

```bash
examples/1_read_mat_file/read_mat_file [path_to_mat_file] [block_size]
```

### Common options:

- `-X <file.mat>` : Load initial `X` from the given `.mat` file (variable name `X`).
- `-B <file.mat>` : Load right-hand side `B` from the given `.mat` file (variable name `B`).
- `-i <max_iters>`: Maximum iterations (default: `n`).
- `-t <tol>`      : Convergence tolerance (default `1e-6`).
- `--dense`      : Use the dense solver path (converts sparse `A` to dense).
- `--double`     : Run in double precision.
- `-s`           : Print a summary of AX vs B errors after solve.
- `-o <out.mat>` : Write final `X` to the given `.mat` (currently only available for single-precision).

## Examples

### Run float solver with default s=1
```bash
examples/1_read_mat_file/read_mat_file my_sparse.mat
```

### Run float solver with block size 2 and custom tolerance
```bash
examples/1_read_mat_file/read_mat_file my_sparse.mat 2 -t 1e-2
```

### Load initial X and B from .mat files and run double-precision solver
```bash
examples/1_read_mat_file/read_mat_file my_sparse.mat 2 -X X_init.mat -B B.mat --double
```

### Limit iterations and write output (float branch)
```bash
examples/1_read_mat_file/read_mat_file my_sparse.mat 2 -i 100 -o X_out.mat
```

## Notes

- The `-X` and `-B` files must contain variables named `X` and `B` respectively and must be of size `n x s`.
- The `--double` path currently prints a notice if `-o` is used because writing double output is not yet implemented.