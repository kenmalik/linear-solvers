#include <iostream>

#include "cg/cg.h"
#include "cli/parser.h"

int main(int argc, char *argv[]) {
    auto args = parse_args(argc, argv);

    if (!args) {
        return -1;
    }

    int n = args->A.rows();
    std::vector<double> b(n, 1);
    std::vector<double> x(n, 0);

    int iters;
    if (args->L.has_value()) {
        iters = cg(args->A, b, x, args->L.value(), 1e-6, n);
    } else {
        iters = cg(args->A, b, x, 1e-6, n);
    }

    std::cout << iters << std::endl;

    return 0;
}
