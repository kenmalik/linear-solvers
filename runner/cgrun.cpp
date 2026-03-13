#include "common/mkl_matrices.h"
#include <iostream>
#include <vector>

#ifdef MKL_CG_ENABLED
#include <cg/mkl.h>
#endif

#ifdef MKL_DR_BCG_ENABLED
#include <dr_bcg/mkl.h>
#endif

#ifdef CUDA_CG_ENABLED
#include "cuda_adapter.h"
#include <cg/cuda.h>
#endif

#ifdef CUDA_DR_BCG_ENABLED
#include "cuda_adapter.h"
#include <dr_bcg/cuda.h>
#endif

#include "cgrun.h"
#include "parser.h"

int main(int argc, char *argv[]) {
    auto args = parse_args(argc, argv);

    if (!args) {
        return -1;
    }

    int iters;

    switch (args->algorithm) {
    case Algorithm::CG:
        iters = run_cg(*args);
        break;
    case Algorithm::DR_BCG:
        iters = run_dr_bcg(*args);
        break;
    default:
        std::cerr << "Unknown algorithm" << std::endl;
        return -1;
    }

    if (iters < 0) {
        return -1;
    }

    std::cout << iters << std::endl;
    return 0;
}

int run_cg(const Args &args) {
    int n = args.A.rows();
    std::vector<double> b(n, 1);
    std::vector<double> x(n, 0);

    int max_iters = args.max_iterations.value_or(n);

    std::cerr << "Running solver..." << std::endl;

    switch (args.implementation) {
#ifdef MKL_CG_ENABLED
    case Implementation::MKL: {
        auto A = read_mkl(args.A);
        A.descr.type = SPARSE_MATRIX_TYPE_GENERAL;

        if (args.L.has_value()) {
            auto L = read_mkl(args.L.value());
            L.descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
            L.descr.mode = SPARSE_FILL_MODE_LOWER;
            L.descr.diag = SPARSE_DIAG_NON_UNIT;
            return cg::mkl::solve(A, b, x, L, args.tolerance, max_iters);
        } else {
            std::cerr << "Not implemented" << std::endl;
            return -1;
        }
    }
#endif
#ifdef CUDA_CG_ENABLED
    case Implementation::CUDA: {
        return run_cuda_cg(args.A, b, x, args.L.value(), args.tolerance,
                           max_iters);
    }
#endif
    default:
        std::cerr << "Selected implementation not available in this build"
                  << std::endl;
        return -1;
    }
}

int run_dr_bcg(const Args &args) {
    int n = args.A.rows();
    int s = args.block_size;
    std::vector<double> b(n * s, 1);
    std::vector<double> x(n * s, 0);

    int max_iters = args.max_iterations.value_or(n);

    std::cerr << "Running solver..." << std::endl;

    switch (args.implementation) {
#ifdef MKL_DR_BCG_ENABLED
    case Implementation::MKL: {
        auto A = read_mkl(args.A);
        A.descr.type = SPARSE_MATRIX_TYPE_GENERAL;

        DenseMatrix b_dm{n, s, b};
        DenseMatrix x_dm{n, s, x};

        if (args.L.has_value()) {
            auto L = read_mkl(args.L.value());
            L.descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
            L.descr.mode = SPARSE_FILL_MODE_LOWER;
            L.descr.diag = SPARSE_DIAG_NON_UNIT;
            return dr_bcg::mkl::solve(A, L, b_dm, x_dm, args.tolerance,
                                      max_iters);
        } else {
            std::cerr << "Not implemented" << std::endl;
            return -1;
        }
    }
#endif
#ifdef CUDA_DR_BCG_ENABLED
    case Implementation::CUDA: {
        return run_cuda_dr_bcg(args.A, b, x, args.L.value(), args.tolerance,
                               max_iters, args.block_size);
    }
#endif
    default:
        std::cerr << "Selected implementation not available in this build"
                  << std::endl;
        return -1;
    }
}

CSRMatrix read_mkl(const mat_utils::SpMatReader &reader) {
    const MKL_INT n_rows = static_cast<MKL_INT>(reader.rows());
    const MKL_INT n_cols = static_cast<MKL_INT>(reader.cols());
    const MKL_INT nnz = static_cast<MKL_INT>(reader.nnz());

    const size_t *jc = reader.jc();
    const size_t *ir = reader.ir();
    const double *values = reader.data();

    CSRMatrix csr;
    csr.rows = n_rows;
    csr.cols = n_cols;
    csr.row_ptr.assign(n_rows + 1, 0);
    csr.col_idx.resize(nnz);
    csr.values.resize(nnz);

    // Count nnz per row
    for (MKL_INT k = 0; k < nnz; ++k)
        ++csr.row_ptr[ir[k] + 1];

    // Exclusive prefix sum → row_ptr
    for (MKL_INT i = 0; i < n_rows; ++i)
        csr.row_ptr[i + 1] += csr.row_ptr[i];

    // Scatter CSC columns into CSR rows
    std::vector<MKL_INT> cursor(csr.row_ptr.begin(),
                                csr.row_ptr.begin() + n_rows);
    for (MKL_INT j = 0; j < n_cols; ++j) {
        for (size_t k = jc[j]; k < jc[j + 1]; ++k) {
            MKL_INT row = static_cast<MKL_INT>(ir[k]);
            MKL_INT pos = cursor[row]++;
            csr.col_idx[pos] = j;
            csr.values[pos] = values[k];
        }
    }

    mkl_sparse_d_create_csr(&csr.mat, SPARSE_INDEX_BASE_ZERO, reader.rows(),
                            reader.cols(), csr.row_ptr.data(),
                            csr.row_ptr.data() + 1, csr.col_idx.data(),
                            csr.values.data());

    return csr;
}
