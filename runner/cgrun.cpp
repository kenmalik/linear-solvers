#include "common/mkl_matrices.h"
#include <iostream>
#include <vector>

#ifdef MKL_CG_ENABLED
#include <cg/mkl.h>
#endif

#ifdef CUDA_CG_ENABLED
#include "cuda_adapter.h"
#include <cg/cuda.h>
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

    std::cerr << "Running solver..." << std::endl;

    switch (args.implementation) {
#ifdef MKL_CG_ENABLED
    case Implementation::MKL: {
        const auto A = read_mkl(args.A);
        if (args.L.has_value()) {
            auto L = read_mkl(args.L.value());
            L.descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
            L.descr.mode = SPARSE_FILL_MODE_UPPER;
            L.descr.diag = SPARSE_DIAG_NON_UNIT;
            return cg::mkl::solve(A, b, x, L, 1e-6, n);
        } else {
            std::cerr << "Not implemented" << std::endl;
            return -1;
        }
    }
#endif
#ifdef CUDA_CG_ENABLED
    case Implementation::CUDA: {
        return run_cuda(args.A, b, x, args.L.value(), 1e-6, n);
    }
#endif
    default:
        std::cerr << "Selected implementation not available in this build"
                  << std::endl;
        return -1;
    }
}

CSRMatrix read_mkl(const mat_utils::SpMatReader &reader) {
    // SpMatReader stores the matrix in MATLAB's CSC format:
    //   jc: column pointers (size cols+1)
    //   ir: row indices    (size nnz)
    //
    // Since A is symmetric (SPD), treating the CSC arrays as CSR gives Aᵀ =
    // A, so we can use mkl_sparse_d_create_csr directly with jc/ir.
    //
    // The index arrays are size_t; copy to MKL_INT64 as required by the
    // API.
    const double *vals = reader.data();
    const size_t *jc = reader.jc();
    const size_t *ir = reader.ir();

    CSRMatrix sparse;

    sparse.values.assign(vals, vals + reader.nnz());
    sparse.row_ptr.assign(jc, jc + reader.jc_size());
    sparse.col_idx.assign(ir, ir + reader.ir_size());

    mkl_sparse_d_create_csr(&sparse.mat, SPARSE_INDEX_BASE_ZERO, reader.rows(),
                            reader.cols(), sparse.row_ptr.data(),
                            sparse.row_ptr.data() + 1, sparse.col_idx.data(),
                            sparse.values.data());

    sparse.descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    return sparse;
}
