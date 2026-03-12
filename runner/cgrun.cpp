#include <iostream>
#include <vector>

#ifdef CUDA_CG_ENABLED
#include <cg/cg.h>
#endif

#include "parser.h"

int main(int argc, char *argv[]) {
    auto args = parse_args(argc, argv);

    if (!args) {
        return -1;
    }

    int n = args->A.rows();
    std::vector<double> b(n, 1);
    std::vector<double> x(n, 0);

    std::cerr << "Running solver..." << std::endl;

#ifdef MKL_CG_ENABLED
    int iters;

    const auto A = cg::mkl::MKLSparse{args->A};
    if (args->L.has_value()) {
        auto L = cg::mkl::MKLSparse{args->L.value()};
        L.descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
        L.descr.mode = SPARSE_FILL_MODE_UPPER;
        L.descr.diag = SPARSE_DIAG_NON_UNIT;

        iters = cg::mkl::solve(A, b, x, L, 1e-6, n);
    } else {
        std::cerr << "Not implemented" << std::endl;
        // iters = cg::mkl::solve(args->A, b, x, 1e-6, n);
    }

    std::cout << iters << std::endl;
#endif

    return 0;
}
