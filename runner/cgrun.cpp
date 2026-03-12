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
    if (args->L.has_value()) {
        iters = cg::mkl::solve(args->A, b, x, args->L.value(), 1e-6, n);
    } else {
        iters = cg::mkl::solve(args->A, b, x, 1e-6, n);
    }

    std::cout << iters << std::endl;
#endif

    return 0;
}
