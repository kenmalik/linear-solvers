# Test Data

This directory contains test data to run `cgrun` and unit tests.

Some files have the following name and corresponding meaning (where `n` is the size of `<matrix>`'s first dimension):

- `<matrix>_ichol.mat`: incomplete Cholesky decomposition of `<matrix>`; used as preconditioner `L`
- `<matrix>_x.mat`: vector of zeros of size `n`; used as input to the `-x` option
- `<matrix>_X.mat`: matrix of zeros of size `n * (n - 1)`; used as input to the `-X` option
- `<matrix>_b.mat`: vector of normally-distributed random numbers of size `n`; used as input to the `-b` option
- `<matrix>_B.mat`: vector of normally-distributed random numbers of size `n * (n - 1)`; used as input to the `-B` option

The `x`, `X`, `b`, and `B` values can be conceptualized as follows:

```
A[x, X] = [b, B]
```

Meaning:

- `x` contains the first column of the initial guess matrix
- `X` contains the rest of the columns of the initial guess matrix
- `b` contains the first column of the RHS matrix
- `B` contains the rest of the columns of the RHS matrix
