#include <iostream>
#include <vector>

#ifdef MKL_CG_ENABLED
#include "mkl_adapter.h"
#endif

#ifdef MKL_DR_BCG_ENABLED
#include "mkl_adapter.h"
#endif

#ifdef CUDA_CG_ENABLED
#include "cuda_adapter.h"
#endif

#ifdef CUDA_DR_BCG_ENABLED
#include "cuda_adapter.h"
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

    switch (args.implementation) {
#ifdef MKL_CG_ENABLED
    case Implementation::MKL: {
        if (args.L.has_value()) {
            return run_mkl_cg(args.A, b, x, args.L.value(), args.tolerance,
                              max_iters);
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

    switch (args.implementation) {
#ifdef MKL_DR_BCG_ENABLED
    case Implementation::MKL: {
        if (args.L.has_value()) {
            return run_mkl_dr_bcg(args.A, b, x, args.L.value(), args.tolerance,
                                  max_iters, s);
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

