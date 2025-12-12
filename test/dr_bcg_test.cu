#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "dr_bcg/dr_bcg.h"
#include "dr_bcg/helper.h"

#include <cuda/std/cmath>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/mismatch.h>

template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
bool match(const thrust::host_vector<T> &a, const thrust::host_vector<T> &b,
           T tolerance = 1e-6) {
    if (a.size() != b.size()) {
        return false;
    }

    auto float_compare = [tolerance](T x, T y) {
        return cuda::std::abs(x - y) <= tolerance;
    };
    auto [a_diff, b_diff] =
        thrust::mismatch(a.begin(), a.end(), b.begin(), float_compare);

    return a_diff == a.end() && b_diff == b.end();
}

TEST(InvertSquareMatrix, TwoByTwoMatrix) {
    constexpr int m = 2;

    cusolverDnHandle_t cusolverH;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    cusolverDnParams_t params;
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));

    std::vector<float> vals = {1, 3, 2, 4};
    thrust::device_vector<float> A(vals.begin(), vals.end());
    float *d_A = thrust::raw_pointer_cast(A.data());

    invert_square_matrix(cusolverH, params, d_A, m);

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUSOLVER_CHECK(cusolverDnDestroyParams(params));

    std::vector<float> expected_vals = {-2, 1.5f, 1, -0.5f};
    thrust::host_vector<float> expected(expected_vals.begin(),
                                        expected_vals.end());
    std::cerr << "expected" << std::endl;
    print_matrix(thrust::raw_pointer_cast(expected.data()), m, m);

    thrust::host_vector<float> got = A;
    print_device_matrix(d_A, 2, 2);
    ASSERT_TRUE(match(expected, got));
}

TEST(InvertSquareMatrix, DiagonalMatrix) {
    constexpr int m = 8;

    cusolverDnHandle_t cusolverH;
    cusolverDnParams_t cusolverParams;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUSOLVER_CHECK(cusolverDnCreateParams(&cusolverParams));

    struct diagonal_functor {
        const int N;
        const float X;
        diagonal_functor(int n, float x) : N(n), X(x) {}

        __host__ __device__ float operator()(int index) const {
            int row = index / N;
            int col = index % N;
            return (row == col) ? X : 0;
        }
    };

    thrust::counting_iterator<int> begin(0);
    thrust::counting_iterator<int> end = begin + m * m;

    constexpr float fill_value = 10;
    thrust::host_vector<float> I(m * m);
    thrust::transform(begin, end, I.begin(), diagonal_functor(m, fill_value));

    thrust::device_vector<float> A = I;
    float *d_A = thrust::raw_pointer_cast(A.data());

    // Operation
    invert_square_matrix(cusolverH, cusolverParams, d_A, m);

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUSOLVER_CHECK(cusolverDnDestroyParams(cusolverParams));

    // Test that result contains reciprocal of fill value on diagonal
    thrust::host_vector<float> expected(m * m);
    thrust::transform(begin, end, expected.begin(),
                      diagonal_functor(m, 1 / fill_value));
    thrust::host_vector<float> got = A;
    ASSERT_TRUE(match(expected, got));
}

#ifdef DR_BCG_USE_THIN_QR

TEST(ThinQR, OutputCorrect) {
    constexpr float test_tolerance = 1e-6;

    constexpr int m = 32;
    constexpr int n = 4;

    cublasHandle_t cublasH;
    CUBLAS_CHECK(cublasCreate_v2(&cublasH));

    cusolverDnHandle_t cusolverH;
    cusolverDnParams_t params;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0, 1);

    std::vector<float> h_A_in(m * n);
    std::vector<float> h_A_out(m * n);

    for (auto &val : h_A_in) {
        val = dist(gen);
    }

    float *d_A = nullptr;
    float *d_Q = nullptr;
    float *d_R = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, sizeof(float) * m * n));
    CUDA_CHECK(cudaMalloc(&d_Q, sizeof(float) * m * n));
    CUDA_CHECK(cudaMalloc(&d_R, sizeof(float) * m * m));

    CUDA_CHECK(cudaMemcpy(d_A, h_A_in.data(), sizeof(float) * m * n,
                          cudaMemcpyHostToDevice));

    dr_bcg::thin_qr(cusolverH, params, cublasH, d_Q, d_R, m, n, d_A);

    std::cerr << "Q:" << std::endl;
    print_device_matrix(d_Q, m, n);
    std::cerr << "R:" << std::endl;
    print_device_matrix(d_R, n, n);

    constexpr float alpha = 1;
    constexpr float beta = 0;
    CUBLAS_CHECK(cublasSgemm_v2(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n,
                                &alpha, d_Q, m, d_R, n, &beta, d_A, m));

    CUDA_CHECK(cudaMemcpy(h_A_out.data(), d_A, sizeof(float) * h_A_out.size(),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_R));

    CUBLAS_CHECK(cublasDestroy_v2(cublasH));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUSOLVER_CHECK(cusolverDnDestroyParams(params));

    std::cerr << "A in:" << std::endl;
    print_matrix(h_A_in.data(), m, n);
    std::cerr << "A out:" << std::endl;
    print_matrix(h_A_out.data(), m, n);

    for (int i = 0; i < h_A_in.size(); i++) {
        ASSERT_NEAR(h_A_in.at(i), h_A_out.at(i), test_tolerance);
    }
}

#else

TEST(QR_Factorization, ProductOfFactorsIsA) {
    constexpr int m = 8;
    constexpr int n = 4;

    cusolverDnHandle_t cusolverH;
    cusolverDnParams_t params;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));

    thrust::device_vector<float> A(m * n);
    thrust::fill(A.begin(), A.end(), 2);
    float *d_A = thrust::raw_pointer_cast(A.data());

    thrust::device_vector<float> Q(m * n);
    float *d_Q = thrust::raw_pointer_cast(Q.data());

    thrust::device_vector<float> R(n * n);
    float *d_R = thrust::raw_pointer_cast(R.data());

    qr_factorization(cusolverH, params, d_Q, d_R, m, n, d_A);

    std::cerr << "Q" << std::endl;
    print_device_matrix(d_Q, m, n);
    std::cerr << std::endl << "R" << std::endl;
    print_device_matrix(d_R, n, n);

    // Verification that A = Q * R
    cublasHandle_t cublasH;
    CUBLAS_CHECK(cublasCreate_v2(&cublasH));

    thrust::device_vector<float> res(m * n);
    float *d_res = thrust::raw_pointer_cast(res.data());

    constexpr float alpha = 1;
    constexpr float beta = 0;
    CUBLAS_CHECK(cublasSgemm_v2(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n,
                                &alpha, d_Q, m, d_R, n, &beta, d_res, m));

    std::cerr << std::endl << "Q * R" << std::endl;
    print_device_matrix(d_res, n, n);

    thrust::host_vector<float> expected = A;
    thrust::host_vector<float> got = res;
    ASSERT_TRUE(match(expected, got, 1e-5f));
}

#endif // DR_BCG_USE_THIN_QR

TEST(CopyUpperTriangular, CopyFromSquareMatrix) {
    constexpr int n = 3;

    thrust::device_vector<float> dv_A(n * n);
    thrust::fill(dv_A.begin(), dv_A.end(), 3.0f);
    float *d_A = thrust::raw_pointer_cast(dv_A.data());

    thrust::device_vector<float> dv_B(n * n, 1.0f);
    float *d_B = thrust::raw_pointer_cast(dv_B.data());

    copy_upper_triangular(d_B, d_A, n, n);

    std::vector<float> expected = {3.0f, 1.0f, 1.0f, 3.0f, 3.0f,
                                   1.0f, 3.0f, 3.0f, 3.0f};

    thrust::host_vector<float> hv_expected(expected.begin(), expected.end());
    thrust::host_vector<float> hv_got = dv_B;

    ASSERT_TRUE(match(hv_expected, hv_got));
}

TEST(CopyUpperTriangular, CopyFromTallMatrix) {
    constexpr int m = 6;
    constexpr int n = 3;

    thrust::device_vector<float> dv_A(m * n);
    thrust::fill(dv_A.begin(), dv_A.end(), 3.0f);
    float *d_A = thrust::raw_pointer_cast(dv_A.data());

    thrust::device_vector<float> dv_B(n * n, 1.0f);
    float *d_B = thrust::raw_pointer_cast(dv_B.data());

    copy_upper_triangular(d_B, d_A, m, n);

    std::vector<float> expected = {3.0f, 1.0f, 1.0f, 3.0f, 3.0f,
                                   1.0f, 3.0f, 3.0f, 3.0f};

    thrust::host_vector<float> hv_expected(expected.begin(), expected.end());
    thrust::host_vector<float> hv_got = dv_B;

    ASSERT_TRUE(match(hv_expected, hv_got));
}

TEST(SPTRI_LeftMultiply, IdentityStaysSame) {
    cusparseHandle_t cusparseH;
    CUSPARSE_CHECK(cusparseCreate(&cusparseH));

    constexpr int m = 8;
    constexpr int n = 4;

    constexpr cudaDataType_t data_type = CUDA_R_32F;
    constexpr cusparseOrder_t order = CUSPARSE_ORDER_COL;

    constexpr cusparseIndexType_t index_type = CUSPARSE_INDEX_64I;
    constexpr cusparseIndexBase_t base_type = CUSPARSE_INDEX_BASE_ZERO;

    // A = I
    thrust::device_vector<float> A_vals(m);
    thrust::fill(A_vals.begin(), A_vals.end(), 1);

    auto counter = thrust::make_counting_iterator<int64_t>(0);

    thrust::device_vector<int64_t> A_row_offsets(m + 1);
    thrust::copy_n(counter, A_row_offsets.size(), A_row_offsets.begin());
    int64_t *row_offsets = thrust::raw_pointer_cast(A_row_offsets.data());

    counter = thrust::make_counting_iterator<int64_t>(0);
    thrust::device_vector<int64_t> A_col_indices(m);
    thrust::copy_n(counter, A_col_indices.size(), A_col_indices.begin());
    int64_t *col_indices = thrust::raw_pointer_cast(A_col_indices.data());

    thrust::device_vector<float> values(m * n);
    thrust::fill(values.begin(), values.end(), 1);
    float *d_values = thrust::raw_pointer_cast(values.data());

    cusparseSpMatDescr_t A_desc;
    CUSPARSE_CHECK(cusparseCreateCsr(&A_desc, m, m, m, row_offsets, col_indices,
                                     d_values, index_type, index_type,
                                     base_type, data_type));

    // B = 1
    thrust::device_vector<float> B(m * n);
    thrust::fill(B.begin(), B.end(), 1);
    float *d_B = thrust::raw_pointer_cast(B.data());
    cusparseDnMatDescr_t B_desc;
    CUSPARSE_CHECK(
        cusparseCreateDnMat(&B_desc, m, n, m, d_B, data_type, order));

    // C initialized
    thrust::device_vector<float> C(m * n);
    float *d_C = thrust::raw_pointer_cast(C.data());
    cusparseDnMatDescr_t C_desc;
    CUSPARSE_CHECK(
        cusparseCreateDnMat(&C_desc, m, n, m, d_C, data_type, order));

    constexpr cusparseOperation_t op_type = CUSPARSE_OPERATION_NON_TRANSPOSE;
    sptri_left_multiply(cusparseH, C_desc, op_type, A_desc, B_desc);

    CUSPARSE_CHECK(cusparseDestroySpMat(A_desc));
    CUSPARSE_CHECK(cusparseDestroyDnMat(B_desc));
    CUSPARSE_CHECK(cusparseDestroyDnMat(C_desc));

    CUSPARSE_CHECK(cusparseDestroy(cusparseH));

    thrust::host_vector<float> expected = B;
    thrust::host_vector<float> got = C;
    ASSERT_TRUE(match(expected, got));
}
