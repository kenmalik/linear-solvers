#include <array>
#include <cmath>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include <mat_utils/mat_reader.h>
#include <mat_utils/mat_writer.h>

#include <thrust/device_vector.h>

#include "dr_bcg/device_sparse_matrix.h"
#include "dr_bcg/dr_bcg.h"
#include "dr_bcg/helper.h"

struct Args {
    int s = 1;
    float tolerance = 0;
    std::filesystem::path A_file;
    std::filesystem::path L_file;
    std::filesystem::path X_file;
    std::filesystem::path B_file;
};

std::ostream &operator<<(std::ostream &os, const Args &a) {
    os << "Args(s=" << a.s << ", tolerance=" << a.tolerance
       << ", A_file=" << a.A_file << ", L_file=" << a.L_file
       << ", X_file=" << a.X_file << ", B_file=" << a.B_file << ")";
    return os;
}

void print_help(std::string_view binary_name) {
    std::cerr << "Usage: " << binary_name
              << " [spd matrix] [preconditioner] [OPTIONAL block size]"
              << std::endl
              << std::endl
              << "By default, block size is set to 1." << std::endl
              << std::endl
              << "Options:" << std::endl
              << "-X [path to X file] Path to initial X matrix file"
              << std::endl
              << "-B [path to B file] Path to B matrix file" << std::endl
              << "-t [tolerance] (float value) Conversion tolerance"
              << std::endl;
}

Args parse(int argc, char *argv[]) {
    Args args;

    namespace fs = std::filesystem;

    std::array<std::function<void(std::string_view)>, 3> positional_handlers = {
        [&A_file = args.A_file](std::string_view arg) {
            A_file = fs::path{arg};
            if (!fs::exists(A_file))
                throw std::runtime_error(A_file);
        },
        [&L_file = args.L_file](std::string_view arg) {
            L_file = fs::path{arg};
            if (!fs::exists(L_file))
                throw std::runtime_error(L_file);
        },
        [&s = args.s](std::string_view arg) {
            s = std::stoi(std::string(arg));
        }};

    std::unordered_map<std::string, std::function<void(std::string_view)>>
        option_handlers = {
            {"-X",
             [&X_file = args.X_file](std::string_view arg) {
                 X_file = fs::path{arg};
                 if (!fs::exists(X_file))
                     throw std::runtime_error(X_file);
             }},
            {"-B",
             [&B_file = args.B_file](std::string_view arg) {
                 B_file = fs::path{arg};
                 if (!fs::exists(B_file))
                     throw std::runtime_error(B_file);
             }},
            {"-t", [&tolerance = args.tolerance](std::string_view arg) {
                 tolerance = std::stof(std::string(arg));
             }}};

    int position = 0;
    std::function<void(std::string_view)> current_callback = nullptr;
    for (int i = 1; i < argc; ++i) {
        std::string arg{argv[i]};
        if (option_handlers.count(arg)) {
            current_callback = option_handlers.at(arg);
        } else if (current_callback) {
            current_callback(arg);
            current_callback = nullptr;
        } else if (position < positional_handlers.size()) {
            positional_handlers[position](arg);
            ++position;
        } else {
            throw std::runtime_error(arg);
        }
    }
    if (position < 2) {
        throw std::runtime_error("Invalid argument count");
    }

    return args;
}

int main(int argc, char *argv[]) {
    std::string binary_name = std::filesystem::path(argv[0]).filename();

    Args args;
    try {
        args = parse(argc, argv);
    } catch (const std::runtime_error &e) {
        std::cerr << "Bad argument: " << e.what() << std::endl << std::endl;
        print_help(binary_name);
        return 1;
    } catch (const std::exception &e) {
        print_help(binary_name);
        return 1;
    }

    std::cerr << "Running with " << args << std::endl;

    cusolverDnHandle_t cusolverH;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    cusolverDnParams_t cusolverP;
    CUSOLVER_CHECK(cusolverDnCreateParams(&cusolverP));

    cublasHandle_t cublasH;
    CUBLAS_CHECK(cublasCreate_v2(&cublasH));

    cusparseHandle_t cusparseH;
    CUSPARSE_CHECK(cusparseCreate(&cusparseH));

    // Read A
    const std::string A_file = args.A_file;
    mat_utils::SpMatReader ssm_A(A_file, {"Problem"}, "A");
    DeviceSparseMatrixFloat A{ssm_A};
    const int n = ssm_A.rows();
    std::cerr << "Read " << ssm_A.nnz() << " values from A" << std::endl;

    // Read L
    const std::string L_file = args.L_file;
    mat_utils::SpMatReader ssm_L(L_file, {}, "L");
    DeviceSparseMatrixFloat L{ssm_L};
    std::cerr << "Read " << ssm_L.nnz() << " values from L" << std::endl;

    cusparseFillMode_t fill = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t diag = CUSPARSE_DIAG_TYPE_NON_UNIT;
    cusparseSpMatSetAttribute(L.get(), CUSPARSE_SPMAT_FILL_MODE, &fill,
                              sizeof(fill));
    cusparseSpMatSetAttribute(L.get(), CUSPARSE_SPMAT_DIAG_TYPE, &diag,
                              sizeof(diag));

    // X = 0
    cusparseDnMatDescr_t X;
    thrust::device_vector<float> X_vec(n * args.s, 0);
    float *d_X = thrust::raw_pointer_cast(X_vec.data());
    CUSPARSE_CHECK(cusparseCreateDnMat(&X, n, args.s, n, d_X, CUDA_R_32F,
                                       CUSPARSE_ORDER_COL));

    if (!args.X_file.empty()) {
        mat_utils::DnMatReader X{args.X_file, {}, "X"};
        std::vector<float> X_float(X.size());
        for (int i = 0; i < X.size(); ++i) {
            X_float[i] = static_cast<float>(X.data()[i]);
        }
        assert(X.size() == X_vec.size() && "X read from file must be n by s");
        CUDA_CHECK(cudaMemcpy(d_X, X_float.data(),
                              sizeof(float) * X_float.size(),
                              cudaMemcpyHostToDevice));
        std::cerr << "Read " << X.size() << " values from X" << std::endl;
    }

    // B = 1
    cusparseDnMatDescr_t B;
    thrust::device_vector<float> B_vec(n * args.s, 1);
    float *d_B = thrust::raw_pointer_cast(B_vec.data());
    CUSPARSE_CHECK(cusparseCreateDnMat(&B, n, args.s, n, d_B, CUDA_R_32F,
                                       CUSPARSE_ORDER_COL));

    if (!args.B_file.empty()) {
        mat_utils::DnMatReader B{args.B_file, {}, "B"};
        std::vector<float> B_float(B.size());
        for (int i = 0; i < B.size(); ++i) {
            B_float[i] = static_cast<float>(B.data()[i]);
        }
        assert(B.size() == B_vec.size() && "B read from file must be n by s");
        CUDA_CHECK(cudaMemcpy(d_B, B_float.data(),
                              sizeof(float) * B_float.size(),
                              cudaMemcpyHostToDevice));
        std::cerr << "Read " << B.size() << " values from B" << std::endl;
    }

    float tolerance = std::numeric_limits<float>::epsilon();
    if (args.tolerance > 0) {
        tolerance = args.tolerance;
    }
    const int max_iterations = n;

    int iterations = 0;
    dr_bcg::dr_bcg(cusolverH, cusolverP, cublasH, cusparseH, A.get(), X, B,
                   L.get(), tolerance, max_iterations, &iterations);

    std::cout << iterations << std::endl;

    std::vector<float> h_X(n * args.s);
    CUDA_CHECK(cudaMemcpy(h_X.data(), d_X, sizeof(float) * h_X.size(),
                          cudaMemcpyDeviceToHost));
    mat_utils::MatWriter writer("X.mat");
    writer.write_dense("X", h_X, n, args.s);

    CUSPARSE_CHECK(cusparseDestroyDnMat(X));
    CUSPARSE_CHECK(cusparseDestroyDnMat(B));

    CUSPARSE_CHECK(cusparseDestroy(cusparseH));
    CUBLAS_CHECK(cublasDestroy_v2(cublasH));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUSOLVER_CHECK(cusolverDnDestroyParams(cusolverP));

    return 0;
}
