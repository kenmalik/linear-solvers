#include "config.h"

#include <iostream>
#include <type_traits>
#include <vector>

#ifdef SOLVERS_BUILD_MKL
#include "mkl_adapter.h"
#endif

#ifdef SOLVERS_BUILD_CUDA
#include "cuda_adapter.h"
#endif

#include <mat_utils/mat_writer.h>
#include <mat_utils/supported_type.h>

#include "common/cuda_event_timer.h"
#include "common/supported_type.h"
#include "common/timer.h"

#include "cgrun.h"
#include "parser.h"
#include "rhs.h"

int main(int argc, char *argv[]) { // NOLINT(*avoid-c-arrays)
    auto args = parse_args(argc, argv);

    if (!args) {
        return -1;
    }

    int iters = 0;

    switch (args->algorithm) {
    case Algorithm::CG:
        iters = run_cg(*args);
        break;
    case Algorithm::DR_BCG:
        iters = run_dr_bcg(*args);
        break;
    default:
        std::cerr << "Unknown algorithm\n";
        return -1;
    }

    if (iters < 0) {
        return -1;
    }

    std::cout << iters << '\n';

    if constexpr (timer_enabled) {
        if (args->implementation == Implementation::CUDA) {
            cils::detail::g_event_timer.report(args->timer_out);
        } else {
            g_timer.report(args->timer_out);
        }
    }

    return 0;
}

int run_cg(const Args &args) {
    std::size_t n = args.A.rows();
    std::vector<double> b;
    std::vector<double> x;

    try {
        b = prepare_rhs<double>(args.b, args.B, n, 1);
        x = prepare_initial_guess<double>(args.x, args.X, n, 1);
    } catch (const std::exception &e) {
        std::cerr << "Failed to prepare dense inputs: " << e.what() << '\n';
        return -1;
    }

    int max_iters = args.max_iterations.value_or(n);

    int iters = 0;

    switch (args.implementation) {
#ifdef SOLVERS_BUILD_CG
#ifdef SOLVERS_BUILD_MKL
    case Implementation::MKL: {
        if (args.L.has_value()) {
            iters = run_mkl_cg(args.A, b, x, args.L.value(), args.tolerance,
                               max_iters);
        } else {
            std::cerr << "Not implemented\n";
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
        std::cerr << "Selected implementation not available in this build\n";
        return -1;
    }

    if (iters >= 0 && args.output) {
        mat_utils::MatWriter writer(*args.output);
        writer.write_dense("X", x, n, 1);
        if (args.output_b) {
            writer.write_dense("B", b, n, 1);
        }
        writer.close();
    }

    return iters;
}

template <cils::SupportedType T>
int run_dr_bcg_impl(const Args &args) {
    std::size_t n = args.A.rows();
    int s = args.block_size;
    std::vector<T> b;
    std::vector<T> x;

    try {
        b = prepare_rhs<T>(args.b, args.B, n, s);
        x = prepare_initial_guess<T>(args.x, args.X, n, s);
    } catch (const std::exception &e) {
        std::cerr << "Failed to prepare dense inputs: " << e.what() << '\n';
        return -1;
    }

    int max_iterations = args.max_iterations.value_or(n);

    int iters = 0;

    switch (args.implementation) {
#ifdef SOLVERS_BUILD_DR_BCG
#ifdef SOLVERS_BUILD_MKL
    case Implementation::MKL: {
        if constexpr (std::is_same_v<T, double>) {
            MklDrBcgConfig config{
                .tolerance = args.tolerance,
                .max_iterations = max_iterations,
                .block_size = s};

            if (args.L.has_value()) {
                iters = run_mkl_dr_bcg(args.A, b, x, args.L.value(), config);
            } else {
                std::cerr << "Not implemented\n";
                return -1;
            }
        } else {
            std::cerr << "MKL implementation does not support single precision\n";
            return -1;
        }
        break;
    }
#endif // SOLVERS_BUILD_MKL
#ifdef SOLVERS_BUILD_CUDA
    case Implementation::CUDA: {
        CudaDrBcgConfig<T> config{
            .tolerance = static_cast<T>(args.tolerance),
            .max_iterations = max_iterations,
            .block_size = args.block_size,
            .disable_tensor_cores = args.disable_tensor_cores,
            .qr_backend = args.qr_backend,
            .fused_xi = args.fused_xi};

        try {
            if (args.L.has_value()) {
                iters = run_cuda_dr_bcg<T>(args.A, b, x, args.L.value(), config);
            } else {
                iters = run_cuda_dr_bcg<T>(args.A, b, x, config);
            }
        } catch (const std::exception &e) {
            std::cerr << "CUDA DR-BCG failed: " << e.what() << '\n';
            return -1;
        }

        break;
    }
#endif // SOLVERS_BUILD_CUDA
#endif // SOLVERS_BUILD_DR_BCG
    default:
        std::cerr << "Selected implementation not available in this build\n";
        return -1;
    }

    if (iters >= 0 && args.output) {
        mat_utils::MatWriter writer(*args.output);
        writer.write_dense("X", x, n, s);
        if (args.output_b) {
            writer.write_dense("B", b, n, s);
        }
        writer.close();
    }

    return iters;
}

int run_dr_bcg(const Args &args) {
    if (args.A.is_double()) {
        return run_dr_bcg_impl<double>(args);
    }
    return run_dr_bcg_impl<float>(args);
}
