#include "config.h"

#include <iostream>
#include <vector>

#ifdef SOLVERS_BUILD_MKL
#include "mkl_adapter.h"
#endif

#ifdef SOLVERS_BUILD_CUDA
#include "cuda_adapter.h"
#endif

#include "common/cuda_event_timer.h"
#include "common/timer.h"

#include "cgrun.h"
#include "parser.h"
#include "rhs.h"

namespace {

#if defined(SOLVERS_BUILD_CUDA) && defined(SOLVERS_BUILD_DR_BCG)
dr_bcg::cuda::QrBackend to_cuda_qr_backend(QrBackend backend) {
    switch (backend) {
    case QrBackend::Householder:
        return dr_bcg::cuda::QrBackend::Householder;
    case QrBackend::CholQR:
        return dr_bcg::cuda::QrBackend::CholQR;
    default:
        throw std::runtime_error("Unknown QR backend");
    }
}
#endif

} // namespace

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

    switch (args.implementation) {
#ifdef SOLVERS_BUILD_CG
#ifdef SOLVERS_BUILD_MKL
    case Implementation::MKL: {
        if (args.L.has_value()) {
            return run_mkl_cg(args.A, b, x, args.L.value(), args.tolerance,
                              max_iters);
        } else {
            std::cerr << "Not implemented" << std::endl;
            return -1;
        }
    }
#endif // SOLVERS_BUILD_MKL
#ifdef SOLVERS_BUILD_CUDA
    case Implementation::CUDA: {
        return run_cuda_cg(args.A, b, x, args.L.value(), args.tolerance,
                           max_iters, args.disable_tensor_cores);
    }
#endif // SOLVERS_BUILD_CUDA
#endif // SOLVERS_BUILD_CG
    default:
        std::cerr << "Selected implementation not available in this build"
                  << std::endl;
        return -1;
    }
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

    switch (args.implementation) {
#ifdef SOLVERS_BUILD_DR_BCG
#ifdef SOLVERS_BUILD_MKL
    case Implementation::MKL: {
        if (args.L.has_value()) {
            return run_mkl_dr_bcg(args.A, b, x, args.L.value(), args.tolerance,
                                  max_iters, s);
        } else {
            std::cerr << "Not implemented" << std::endl;
            return -1;
        }
    }
#endif // SOLVERS_BUILD_MKL
#ifdef SOLVERS_BUILD_CUDA
    case Implementation::CUDA: {
        if (args.L.has_value()) {
            return run_cuda_dr_bcg(args.A, b, x, args.L.value(), args.tolerance,
                                   max_iters, args.block_size,
                                   args.disable_tensor_cores,
                                   to_cuda_qr_backend(args.qr_backend));
        } else {
            return run_cuda_dr_bcg(args.A, b, x, args.tolerance, max_iters,
                                   args.block_size,
                                   args.disable_tensor_cores,
                                   to_cuda_qr_backend(args.qr_backend));
        }
    }
#endif // SOLVERS_BUILD_CUDA
#endif // SOLVERS_BUILD_DR_BCG
    default:
        std::cerr << "Selected implementation not available in this build"
                  << std::endl;
        return -1;
    }
}
