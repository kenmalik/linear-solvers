# CUDA DR-BCG

## Introduction

This is a CUDA implementation of the Dubrulle-R Block Conjugate Gradient (DR-BCG) algorithm for solving linear systems.

The implementation was originally derived from the following MATLAB code:

```matlab
function [X_final, iterations] = DR_BCG(A, B, X, tol, maxit)
    iterations = 0;
    R = B - A * X;
    [w, sigma] = qr(R,'econ');
    s = w;

    for k = 1:maxit
        iterations = iterations + 1;
        xi = (s' * A * s)^-1;
        X = X + s * xi * sigma;
        if (norm(B(:,1) - A * X(:,1)) / norm(B(:,1))) < tol
            break
        else
            [w, zeta] = qr(w - A * s * xi,'econ');
            s = w + s * zeta';
            sigma = zeta * sigma;
        end
    end
    X_final = X;
end
```

## Building

To simply build the library, run the following commands from the root directory:

```bash
cmake -B build -S .
cmake --build build
```

### Options

You can pass options when building the project for additional/altered functionality.

The following options adjust the behavior of the DR-BCG algorithm:

The following options build additional portions of the project. These are off by default:

- `DR_BCG_BUILD_BENCHMARKS`
- `DR_BCG_BUILD_EXAMPLES`
- `DR_BCG_BUILD_TESTS`

You can pass these when building the project. For example:

```bash
cmake -B build -S . -DDR_BCG_BUILD_EXAMPLES=ON -DDR_BCG_BUILD_TESTS=ON
cmake --build build
```

## Running Examples

See the [examples directory](examples/) for directions on building and running examples.

## Building Tests and Benchmarks

By default, unit tests and benchmarks are not built alongside the main library.

To build them, build the project with the following flags:

```bash
cmake -B build -S . -DDR_BCG_BUILD_TESTS=ON -DDR_BCG_BUILD_BENCHMARKS=ON
```
