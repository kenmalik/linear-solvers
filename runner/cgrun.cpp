#include <iostream>
#include <vector>

#ifdef MKL_CG_ENABLED
#include <cg/cg.h>
#endif

#ifdef CUDA_CG_ENABLED
#include <cg/cg.h>
#endif

#include "parser.h"

static int run_cg(const Args &args) {
    int n = args.A.rows();
    std::vector<double> b(n, 1);
    std::vector<double> x(n, 0);

    std::cerr << "Running solver..." << std::endl;

    switch (args.implementation) {
#ifdef MKL_CG_ENABLED
    case Implementation::MKL: {
        const auto A = cg::mkl::MKLSparse{args.A};
        if (args.L.has_value()) {
            auto L = cg::mkl::MKLSparse{args.L.value()};
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
        std::cerr << "CUDA CG not yet wired up in runner" << std::endl;
        return -1;
    }
#endif
    default:
        std::cerr << "Selected implementation not available in this build"
                  << std::endl;
        return -1;
    }
}

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
