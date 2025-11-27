#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <mat_utils/mat_reader.h>
#include <mat_utils/mat_writer.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "dr_bcg/dense.h"
#include "dr_bcg/device_sparse_matrix.h"
#include "dr_bcg/helper.h"
#include "dr_bcg/sparse.h"

template <typename T>
void verify(cusparseSpMatDescr_t A, cusparseDnMatDescr_t X, int n, int s,
            const thrust::device_vector<T> &B_v, bool print_summary = false) {
    static_assert(std::is_same<T, float>::value ||
                      std::is_same<T, double>::value,
                  "verify<T> only supports float or double");

    thrust::device_vector<T> AX_v(n * s);
    T *d_AX = thrust::raw_pointer_cast(AX_v.data());
    cusparseDnMatDescr_t AX;
    cudaDataType_t valueType =
        std::is_same<T, float>::value ? CUDA_R_32F : CUDA_R_64F;
    CUSPARSE_CHECK(
        cusparseCreateDnMat(&AX, n, s, n, d_AX, valueType, CUSPARSE_ORDER_COL));

    cusparseHandle_t cusparseH;
    CUSPARSE_CHECK(cusparseCreate(&cusparseH));

    std::size_t buffer_size;
    void *buffer = nullptr;

    constexpr cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    T alpha = static_cast<T>(1);
    T beta = static_cast<T>(0);
    cudaDataType_t compute_type = valueType;
    constexpr cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT;

    CUSPARSE_CHECK(cusparseSpMM_bufferSize(cusparseH, op, op, &alpha, A, X,
                                           &beta, AX, compute_type, alg,
                                           &buffer_size));

    if (buffer_size > 0) {
        CUDA_CHECK(cudaMalloc(&buffer, buffer_size));
    }

    CUSPARSE_CHECK(cusparseSpMM(cusparseH, op, op, &alpha, A, X, &beta, AX,
                                compute_type, alg, buffer));
    CUDA_CHECK(cudaDeviceSynchronize());

    if (buffer) {
        CUDA_CHECK(cudaFree(buffer));
    }

    thrust::host_vector<T> got = AX_v;
    thrust::host_vector<T> expected = B_v;

    if (got.size() != expected.size()) {
        std::cerr << "Size mismatch" << std::endl;
        return;
    }

    T avg_error = static_cast<T>(0);
    T max_error = static_cast<T>(0);
    T min_error = std::numeric_limits<T>::max();

    for (int i = 0; i < expected.size(); ++i) {
        T error = std::abs(got[i] - expected[i]);
        if (error > max_error) {
            max_error = error;
        }
        if (error < min_error) {
            min_error = error;
        }
        avg_error += error;
    }
    avg_error /= static_cast<T>(expected.size());

    if (print_summary) {
        std::cerr << "Summary:" << std::endl;
        std::cerr << "Max Error: " << max_error << std::endl;
        std::cerr << "Min Error: " << min_error << std::endl;
        std::cerr << "Avg Error: " << avg_error << std::endl;
    }
}

struct Args {
    std::string matrix_file;
    int s = 1;
    std::optional<int> max_iters = std::nullopt;
    std::optional<std::filesystem::path> out_file = std::nullopt;
    std::optional<std::filesystem::path> X_file = std::nullopt;
    std::optional<std::filesystem::path> B_file = std::nullopt;
    double tolerance = 1e-6;
    bool print_summary = false;
    bool print_help = false;
    bool dense = false;
    bool use_double = false;
};

Args parse_args(int argc, char *argv[]) {
    Args args;

    int positional_number = 0;
    bool reading_max_iters = false;
    bool reading_out_file = false;
    bool reading_tolerance = false;

    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];

        if (reading_max_iters) {
            char *endptr;
            long max_iters = std::strtol(arg, &endptr, 10);
            if (*endptr != '\0' ||
                max_iters > std::numeric_limits<int>::max()) {
                throw std::invalid_argument("Invalid max iterations");
            }
            args.max_iters = max_iters;
            reading_max_iters = false;
        } else if (reading_out_file) {
            std::filesystem::path out_file{arg};
            if (std::filesystem::exists(out_file)) {
                throw std::invalid_argument(
                    "Output file already exists. Cannot overwrite.");
            }
            args.out_file = out_file;
            reading_out_file = false;
        } else if (reading_tolerance) {
            char *endptr;
            double tol = std::strtod(arg, &endptr);
            if (*endptr != '\0' || tol <= 0.0) {
                throw std::invalid_argument("Invalid tolerance");
            }
            args.tolerance = tol;
            reading_tolerance = false;
        } else if (std::strcmp(arg, "-h") == 0) {
            args.print_help = true;
        } else if (std::strcmp(arg, "-s") == 0) {
            args.print_summary = true;
        } else if (std::strcmp(arg, "--dense") == 0) {
            args.dense = true;
        } else if (std::strcmp(arg, "--double") == 0) {
            args.use_double = true;
        } else if (std::strcmp(arg, "-i") == 0) {
            reading_max_iters = true;
        } else if (std::strcmp(arg, "-t") == 0) {
            reading_tolerance = true;
        } else if (std::strcmp(arg, "-X") == 0) {
            // next argument is X file path
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing X file path");
            }
            std::filesystem::path Xp{argv[++i]};
            if (!std::filesystem::exists(Xp)) {
                throw std::invalid_argument("X file does not exist");
            }
            args.X_file = Xp;
        } else if (std::strcmp(arg, "-B") == 0) {
            // next argument is B file path
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing B file path");
            }
            std::filesystem::path Bp{argv[++i]};
            if (!std::filesystem::exists(Bp)) {
                throw std::invalid_argument("B file does not exist");
            }
            args.B_file = Bp;
        } else if (std::strcmp(arg, "-o") == 0) {
            reading_out_file = true;
        } else {
            if (positional_number == 0) {
                args.matrix_file = std::string(arg);
            } else if (positional_number == 1) {
                char *endptr;
                long s = std::strtol(arg, &endptr, 10);
                if (*endptr != '\0' || s > std::numeric_limits<int>::max()) {
                    throw std::invalid_argument("Invalid block size");
                }

                args.s = s;
            } else {
                throw std::invalid_argument("Invalid argument count");
            }
            ++positional_number;
        }
    }

    return args;
}

void print_help() {
    std::cerr << "Usage: ./example_2 [.mat file] [block_size]" << std::endl
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  -h print this help menu" << std::endl;
    std::cerr << "  -s (default 1) print summary of errors between AX and 1_nxn"
              << std::endl;
    std::cerr << "  -i [max_iterations] (default block_size) set the maximum "
                 "iterations solver will run"
              << std::endl;
    std::cerr << "  -t [tolerance] (float) set solver convergence tolerance"
              << std::endl;
    std::cerr << "  -o [output_file] set file to output final X to"
              << std::endl;
    std::cerr << "  -X [path to .mat] read initial X from given .mat file"
              << std::endl;
    std::cerr
        << "  -B [path to .mat] read right-hand side B from given .mat file"
        << std::endl;
    std::cerr << "  --dense use dense solver variant" << std::endl;
    std::cerr << "  --double use double-precision variant" << std::endl;
}

int main(int argc, char *argv[]) {
    Args args;
    try {
        args = parse_args(argc, argv);
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl << std::endl;
        print_help();
        return 1;
    }

    if (args.print_help) {
        print_help();
        return 0;
    }

    mat_utils::SpMatReader ssm(args.matrix_file, {"Problem"}, "A");

    if (args.use_double) {
        DeviceSparseMatrixDouble A(ssm);
        const int n = ssm.rows();

        cusparseDnMatDescr_t X;
        thrust::device_vector<double> X_v(n * args.s, 0.0);
        double *d_X = thrust::raw_pointer_cast(X_v.data());
        CUSPARSE_CHECK(cusparseCreateDnMat(&X, n, args.s, n, d_X, CUDA_R_64F,
                                           CUSPARSE_ORDER_COL));

        if (args.X_file.has_value()) {
            mat_utils::DnMatReader Xr{*args.X_file, {}, "X"};
            std::vector<double> X_double(Xr.size());
            for (int i = 0; i < Xr.size(); ++i) {
                X_double[i] = static_cast<double>(Xr.data()[i]);
            }
            assert(Xr.size() == X_v.size() &&
                   "X read from file must be n by s");
            CUDA_CHECK(cudaMemcpy(d_X, X_double.data(),
                                  sizeof(double) * X_double.size(),
                                  cudaMemcpyHostToDevice));
            std::cerr << "Read " << Xr.size() << " values from X" << std::endl;
        }

        cusparseDnMatDescr_t B;
        thrust::device_vector<double> B_v(n * args.s, 1.0);
        double *d_B = thrust::raw_pointer_cast(B_v.data());
        CUSPARSE_CHECK(cusparseCreateDnMat(&B, n, args.s, n, d_B, CUDA_R_64F,
                                           CUSPARSE_ORDER_COL));

        if (args.B_file.has_value()) {
            mat_utils::DnMatReader Br{*args.B_file, {}, "B"};
            std::vector<double> B_double(Br.size());
            for (int i = 0; i < Br.size(); ++i) {
                B_double[i] = static_cast<double>(Br.data()[i]);
            }
            assert(Br.size() == B_v.size() &&
                   "B read from file must be n by s");
            CUDA_CHECK(cudaMemcpy(d_B, B_double.data(),
                                  sizeof(double) * B_double.size(),
                                  cudaMemcpyHostToDevice));
            std::cerr << "Read " << Br.size() << " values from B" << std::endl;
        }

        double tolerance = args.tolerance;
        const int max_iterations =
            args.max_iters.has_value() ? *args.max_iters : n;

        std::cerr << args.matrix_file << ' ' << n << ' ' << args.s << std::endl;

        int iterations = 0;
        if (args.dense) {
            cusparseHandle_t cusparseH;
            CUSPARSE_CHECK(cusparseCreate(&cusparseH));

            cusparseDnMatDescr_t A_dense;
            thrust::device_vector<double> A_dense_v(n * n);
            double *d_A_dense = thrust::raw_pointer_cast(A_dense_v.data());
            CUSPARSE_CHECK(cusparseCreateDnMat(&A_dense, n, n, n, d_A_dense,
                                               CUDA_R_64F, CUSPARSE_ORDER_COL));

            void *buffer = nullptr;
            std::size_t buffer_size = 0;
            CUSPARSE_CHECK(cusparseSparseToDense_bufferSize(
                cusparseH, A.get(), A_dense, CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
                &buffer_size));
            CUDA_CHECK(cudaMalloc(&buffer, buffer_size));

            CUSPARSE_CHECK(cusparseSparseToDense(
                cusparseH, A.get(), A_dense, CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
                buffer));

            iterations = dr_bcg::dr_bcg(d_A_dense, d_X, d_B, n, args.s,
                                        tolerance, max_iterations);

            CUDA_CHECK(cudaFree(buffer));
            CUSPARSE_CHECK(cusparseDestroyDnMat(A_dense));
            CUSPARSE_CHECK(cusparseDestroy(cusparseH));
        } else {
            iterations =
                dr_bcg::dr_bcg(A.get(), X, B, tolerance, max_iterations);
            verify(A.get(), X, n, args.s, B_v, args.print_summary);
        }

        std::cout << "Iterations: " << iterations << std::endl;

        if (args.out_file) {
            std::cerr << "Double output currently not supported" << std::endl;
        }

        return 0;
    } else {
        DeviceSparseMatrixFloat A(ssm);

        const int n = ssm.rows();

        cusparseDnMatDescr_t X;
        thrust::device_vector<float> X_v(n * args.s, 0.0f);
        float *d_X = thrust::raw_pointer_cast(X_v.data());
        CUSPARSE_CHECK(cusparseCreateDnMat(&X, n, args.s, n, d_X, CUDA_R_32F,
                                           CUSPARSE_ORDER_COL));

        if (args.X_file.has_value()) {
            mat_utils::DnMatReader Xr{*args.X_file, {}, "X"};
            std::vector<float> X_float(Xr.size());
            for (int i = 0; i < Xr.size(); ++i) {
                X_float[i] = static_cast<float>(Xr.data()[i]);
            }
            assert(Xr.size() == X_v.size() &&
                   "X read from file must be n by s");
            CUDA_CHECK(cudaMemcpy(d_X, X_float.data(),
                                  sizeof(float) * X_float.size(),
                                  cudaMemcpyHostToDevice));
            std::cerr << "Read " << Xr.size() << " values from X" << std::endl;
        }

        cusparseDnMatDescr_t B;
        thrust::device_vector<float> B_v(n * args.s, 1.0f);
        float *d_B = thrust::raw_pointer_cast(B_v.data());
        CUSPARSE_CHECK(cusparseCreateDnMat(&B, n, args.s, n, d_B, CUDA_R_32F,
                                           CUSPARSE_ORDER_COL));

        if (args.B_file.has_value()) {
            mat_utils::DnMatReader Br{*args.B_file, {}, "B"};
            std::vector<float> B_float(Br.size());
            for (int i = 0; i < Br.size(); ++i) {
                B_float[i] = static_cast<float>(Br.data()[i]);
            }
            assert(Br.size() == B_v.size() &&
                   "B read from file must be n by s");
            CUDA_CHECK(cudaMemcpy(d_B, B_float.data(),
                                  sizeof(float) * B_float.size(),
                                  cudaMemcpyHostToDevice));
            std::cerr << "Read " << Br.size() << " values from B" << std::endl;
        }

        float tolerance = static_cast<float>(args.tolerance);
        const int max_iterations =
            args.max_iters.has_value() ? *args.max_iters : n;

        std::cerr << args.matrix_file << ' ' << n << ' ' << args.s << std::endl;

        int iterations = 0;
        if (args.dense) {
            cusparseHandle_t cusparseH;
            CUSPARSE_CHECK(cusparseCreate(&cusparseH));

            cusparseDnMatDescr_t A_dense;
            float *d_A_dense = nullptr;
            CUDA_CHECK(cudaMalloc(&d_A_dense, sizeof(float) * n * n));
            CUSPARSE_CHECK(cusparseCreateDnMat(&A_dense, n, n, n, d_A_dense,
                                               CUDA_R_32F, CUSPARSE_ORDER_COL));

            void *buffer = nullptr;
            std::size_t buffer_size = 0;
            CUSPARSE_CHECK(cusparseSparseToDense_bufferSize(
                cusparseH, A.get(), A_dense, CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
                &buffer_size));
            CUDA_CHECK(cudaMalloc(&buffer, buffer_size));

            CUSPARSE_CHECK(cusparseSparseToDense(
                cusparseH, A.get(), A_dense, CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
                buffer));

            iterations = dr_bcg::dr_bcg(d_A_dense, d_X, d_B, n, args.s,
                                        tolerance, max_iterations);

            CUDA_CHECK(cudaFree(buffer));
            CUSPARSE_CHECK(cusparseDestroyDnMat(A_dense));
            CUDA_CHECK(cudaFree(d_A_dense));
            CUSPARSE_CHECK(cusparseDestroy(cusparseH));
        } else {
            iterations =
                dr_bcg::dr_bcg(A.get(), X, B, tolerance, max_iterations);
            verify(A.get(), X, n, args.s, B_v, args.print_summary);
        }

        std::cout << "Iterations: " << iterations << std::endl;

        if (args.out_file) {
            std::vector<float> X_final(X_v.begin(), X_v.end());
            mat_utils::MatWriter w(*args.out_file);
            w.write_dense("X", X_final, n, args.s);
        }

        return 0;
    }
}