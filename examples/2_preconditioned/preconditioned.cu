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

#include "dr_bcg/dr_bcg.h"
#include "dr_bcg/helper.h"

class DeviceSuiteSparseMatrix {
  public:
    explicit DeviceSuiteSparseMatrix(mat_utils::MatReader &ssm_A) {
        CUDA_CHECK(cudaMalloc(&d_rowPtr, sizeof(int64_t) * (ssm_A.rows() + 1)));
        CUDA_CHECK(cudaMalloc(&d_colInd, sizeof(int64_t) * ssm_A.nnz()));
        CUDA_CHECK(cudaMalloc(&d_vals, sizeof(float) * ssm_A.nnz()));

        std::vector<size_t> rowCounts(ssm_A.rows(), 0);
        for (size_t j = 0; j < ssm_A.cols(); ++j) {
            for (size_t p = ssm_A.jc()[j]; p < ssm_A.jc()[j + 1]; ++p) {
                ++rowCounts[ssm_A.ir()[p]];
            }
        }

        std::vector<size_t> csrRowPtr_sz(ssm_A.rows() + 1, 0);
        for (size_t i = 0; i < ssm_A.rows(); ++i)
            csrRowPtr_sz[i + 1] = csrRowPtr_sz[i] + rowCounts[i];

        std::vector<size_t> next = csrRowPtr_sz;

        std::vector<size_t> csrColInd_sz(ssm_A.nnz());
        std::vector<float> csrVal(ssm_A.nnz());

        for (size_t j = 0; j < ssm_A.cols(); ++j) {
            for (size_t p = ssm_A.jc()[j]; p < ssm_A.jc()[j + 1]; ++p) {
                size_t row = ssm_A.ir()[p];
                size_t dst = next[row]++;
                csrColInd_sz[dst] = j;
                csrVal[dst] = static_cast<float>(ssm_A.data()[p]);
            }
        }

        // Convert host indices to int64_t
        std::vector<int64_t> csrRowPtr64(ssm_A.rows() + 1);
        std::vector<int64_t> csrColInd64(ssm_A.nnz());
        for (size_t i = 0; i < csrRowPtr_sz.size(); ++i)
            csrRowPtr64[i] = static_cast<int64_t>(csrRowPtr_sz[i]);
        for (size_t k = 0; k < csrColInd_sz.size(); ++k)
            csrColInd64[k] = static_cast<int64_t>(csrColInd_sz[k]);

        CUDA_CHECK(cudaMemcpy(d_rowPtr, csrRowPtr64.data(),
                              sizeof(int64_t) * csrRowPtr64.size(),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_colInd, csrColInd64.data(),
                              sizeof(int64_t) * csrColInd64.size(),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vals, csrVal.data(),
                              sizeof(float) * csrVal.size(),
                              cudaMemcpyHostToDevice));

        CUSPARSE_CHECK(cusparseCreateCsr(
            &A_, ssm_A.rows(), ssm_A.cols(), ssm_A.nnz(), d_rowPtr, d_colInd,
            d_vals, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    }

    ~DeviceSuiteSparseMatrix() {
        if (A_) {
            CUSPARSE_CHECK(cusparseDestroySpMat(A_));
        }
        if (d_rowPtr) {
            CUDA_CHECK(cudaFree(d_rowPtr));
            d_rowPtr = nullptr;
        }
        if (d_colInd) {
            CUDA_CHECK(cudaFree(d_colInd));
            d_colInd = nullptr;
        }
        if (d_vals) {
            CUDA_CHECK(cudaFree(d_vals));
            d_vals = nullptr;
        }
    }

    cusparseSpMatDescr_t &get() { return A_; }

  private:
    int64_t *d_rowPtr = nullptr;
    int64_t *d_colInd = nullptr;
    float *d_vals = nullptr;
    cusparseSpMatDescr_t A_{};
};

struct Args {
    int s;
    std::filesystem::path A_file;
    std::filesystem::path L_file;
    std::filesystem::path X_file;
    std::filesystem::path B_file;
};

std::ostream &operator<<(std::ostream &os, const Args &a) {
    os << "Args(s=" << a.s << ", A_file=" << a.A_file << ", L_file=" << a.L_file
       << ", X_file=" << a.X_file << ", B_file=" << a.B_file << ")";
    return os;
}

void print_help(std::string_view binary_name) {
    std::cerr << "Usage: " << binary_name
              << " [spd matrix] [preconditioner] [block size]" << std::endl
              << std::endl
              << "Options:" << std::endl
              << "-X [path to X file] Path to initial X matrix file"
              << std::endl
              << "-B [path to B file] Path to B matrix file" << std::endl;
}

Args parse(int argc, char *argv[]) {
    Args args;

    std::array<std::function<void(std::string_view)>, 3> positional_handlers = {
        [&A_file = args.A_file](std::string_view arg) {
            A_file = std::filesystem::path{arg};
        },
        [&L_file = args.L_file](std::string_view arg) {
            L_file = std::filesystem::path{arg};
        },
        [&s = args.s](std::string_view arg) {
            s = std::stoi(std::string(arg));
        }};

    std::unordered_map<std::string, std::function<void(std::string_view)>>
        option_handlers = {
            {"-X",
             [&X_file = args.X_file](std::string_view arg) {
                 X_file = std::filesystem::path{arg};
             }},
            {"-B", [&B_file = args.B_file](std::string_view arg) {
                 B_file = std::filesystem::path{arg};
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
            throw std::runtime_error("Invalid argument: " + arg);
        }
    }
    if (position < positional_handlers.size()) {
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
        std::cerr << e.what() << std::endl << std::endl;
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
    const std::string A_file = argv[1];
    mat_utils::MatReader ssm_A(A_file, {"Problem"}, "A");
    DeviceSuiteSparseMatrix A{ssm_A};
    const int n = ssm_A.rows();

    // Read L
    const std::string L_file = argv[2];
    mat_utils::MatReader ssm_L(L_file, {}, "L");
    DeviceSuiteSparseMatrix L{ssm_L};

    cusparseFillMode_t fill = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t diag = CUSPARSE_DIAG_TYPE_NON_UNIT;
    cusparseSpMatSetAttribute(L.get(), CUSPARSE_SPMAT_FILL_MODE, &fill,
                              sizeof(fill));
    cusparseSpMatSetAttribute(L.get(), CUSPARSE_SPMAT_DIAG_TYPE, &diag,
                              sizeof(diag));

    // X = 0
    thrust::device_vector<float> X_vec(n * args.s);
    thrust::fill(X_vec.begin(), X_vec.end(), 0);
    float *d_X = thrust::raw_pointer_cast(X_vec.data());
    cusparseDnMatDescr_t X;
    CUSPARSE_CHECK(cusparseCreateDnMat(&X, n, args.s, n, d_X, CUDA_R_32F,
                                       CUSPARSE_ORDER_COL));

    // B = 1
    thrust::device_vector<float> B_vec(n * args.s);
    thrust::fill(B_vec.begin(), B_vec.end(), 1);
    float *d_B = thrust::raw_pointer_cast(B_vec.data());
    cusparseDnMatDescr_t B;
    CUSPARSE_CHECK(cusparseCreateDnMat(&B, n, args.s, n, d_B, CUDA_R_32F,
                                       CUSPARSE_ORDER_COL));

    constexpr float tolerance = std::numeric_limits<float>::epsilon();
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
