#include "config.h"

#include <iostream>
#include <vector>

#ifdef SOLVERS_BUILD_MKL
#include "mkl_adapter.h"
#endif

#ifdef SOLVERS_BUILD_CUDA
#include "cuda_adapter.h"
#endif

#include <mat_utils/mat_writer.h>

#include "common/cuda_event_timer.h"
#include "common/timer.h"

#include "cgrun.h"
#include "parser.h"
#include "rhs.h"

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

    if constexpr (timer_enabled) {
        if (args->implementation == Implementation::CUDA) {
            g_event_timer.report(args->timer_out);
        } else {
            g_timer.report(args->timer_out);
        }
    }

    return 0;
}

int run_cg(const Args &args) {
    int n = args.A.rows();
    std::vector<double> b;
    std::vector<double> x;

    try {
        b = prepare_rhs(args.b, args.B, n, 1);
        x = prepare_initial_guess(args.x, args.X, n, 1);
    } catch (const std::exception &e) {
        std::cerr << "Failed to prepare dense inputs: " << e.what() << std::endl;
        return -1;
    }

    int max_iters = args.max_iterations.value_or(n);

    int iters;

    switch (args.implementation) {
#ifdef SOLVERS_BUILD_CG
#ifdef SOLVERS_BUILD_MKL
    case Implementation::MKL: {
        if (args.L.has_value()) {
            iters = run_mkl_cg(args.A, b, x, args.L.value(), args.tolerance,
                               max_iters);
        } else {
            std::cerr << "Not implemented" << std::endl;
            return -1;
        }
        break;
    }
#endif // SOLVERS_BUILD_MKL
#ifdef SOLVERS_BUILD_CUDA
    case Implementation::CUDA: {
        iters = run_cuda_cg(args.A, b, x, args.L.value(), args.tolerance,
                            max_iters, args.disable_tensor_cores);
        break;
    }
#endif // SOLVERS_BUILD_CUDA
#endif // SOLVERS_BUILD_CG
    default:
        std::cerr << "Selected implementation not available in this build"
                  << std::endl;
        return -1;
    }

    if (iters >= 0 && args.output) {
        mat_utils::MatWriter writer(*args.output);
        writer.write_dense("X", x, n, 1);
        writer.close();
    }

    return iters;
}

int run_dr_bcg(const Args &args) {
    int n = args.A.rows();
    int s = args.block_size;
    std::vector<double> b;
    std::vector<double> x;

    try {
        b = prepare_rhs(args.b, args.B, n, s);
        x = prepare_initial_guess(args.x, args.X, n, s);
    } catch (const std::exception &e) {
        std::cerr << "Failed to prepare dense inputs: " << e.what() << std::endl;
        return -1;
    }

    int max_iters = args.max_iterations.value_or(n);

    int iters;

    switch (args.implementation) {
#ifdef SOLVERS_BUILD_DR_BCG
#ifdef SOLVERS_BUILD_MKL
    case Implementation::MKL: {
        if (args.L.has_value()) {
            iters = run_mkl_dr_bcg(args.A, b, x, args.L.value(), args.tolerance,
                                   max_iters, s);
        } else {
            std::cerr << "Not implemented" << std::endl;
            return -1;
        }
        break;
    }
#endif // SOLVERS_BUILD_MKL
#ifdef SOLVERS_BUILD_CUDA
    case Implementation::CUDA: {
        if (args.L.has_value()) {
            iters = run_cuda_dr_bcg(args.A, b, x, args.L.value(), args.tolerance,
                                    max_iters, args.block_size,
                                    args.disable_tensor_cores,
                                    args.qr_backend);
        } else {
            iters = run_cuda_dr_bcg(args.A, b, x, args.tolerance, max_iters,
                                    args.block_size,
                                    args.disable_tensor_cores,
                                    args.qr_backend);
        }
        break;
    }
#endif // SOLVERS_BUILD_CUDA
#endif // SOLVERS_BUILD_DR_BCG
    default:
        std::cerr << "Selected implementation not available in this build"
                  << std::endl;
        return -1;
    }

    if (iters >= 0 && args.output) {
        mat_utils::MatWriter writer(*args.output);
        writer.write_dense("X", x, n, s);
        writer.close();
    }

    return iters;
}
