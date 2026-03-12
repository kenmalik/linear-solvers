#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "dr_bcg/helper.h"
#include "dr_bcg/internal/math.h"
#include "dr_bcg/internal/type_info.h"

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

template <typename T> class Sptri_left_multiply_test : public testing::Test {
  protected:
    static constexpr cudaDataType_t data_type = Type_info<T>::cuda;

    static constexpr int m = 8;
    static constexpr int n = 4;

    static constexpr cusparseOrder_t order = CUSPARSE_ORDER_COL;

    static constexpr cusparseIndexType_t index_type = CUSPARSE_INDEX_64I;
    static constexpr cusparseIndexBase_t base_type = CUSPARSE_INDEX_BASE_ZERO;

    cusparseSpMatDescr_t A_desc;
    cusparseDnMatDescr_t B_desc;
    cusparseDnMatDescr_t C_desc;

    cusparseHandle_t cusparseH;

    thrust::device_vector<int64_t> A_row_offsets;
    int64_t *row_offsets = nullptr;
    thrust::device_vector<int64_t> A_col_indices;
    int64_t *col_indices = nullptr;
    thrust::device_vector<T> values;
    T *d_values = nullptr;

    thrust::device_vector<T> B;
    T *d_B = nullptr;

    thrust::device_vector<T> C;
    T *d_C = nullptr;

    Sptri_left_multiply_test()
        : A_row_offsets(m + 1), A_col_indices(m), values(m * n), B(m * n),
          C(m * n) {
        CUSPARSE_CHECK(cusparseCreate(&cusparseH));

        // A = I
        auto counter = thrust::make_counting_iterator<int64_t>(0);
        thrust::copy_n(counter, A_row_offsets.size(), A_row_offsets.begin());
        row_offsets = thrust::raw_pointer_cast(A_row_offsets.data());

        counter = thrust::make_counting_iterator<int64_t>(0);
        thrust::copy_n(counter, A_col_indices.size(), A_col_indices.begin());
        col_indices = thrust::raw_pointer_cast(A_col_indices.data());

        thrust::fill(values.begin(), values.end(), 1);
        d_values = thrust::raw_pointer_cast(values.data());

        CUSPARSE_CHECK(cusparseCreateCsr(&A_desc, m, m, m, row_offsets,
                                         col_indices, d_values, index_type,
                                         index_type, base_type, data_type));

        // B = 1
        thrust::fill(B.begin(), B.end(), 1);
        d_B = thrust::raw_pointer_cast(B.data());
        CUSPARSE_CHECK(
            cusparseCreateDnMat(&B_desc, m, n, m, d_B, data_type, order));

        // C initialized
        d_C = thrust::raw_pointer_cast(C.data());
        CUSPARSE_CHECK(
            cusparseCreateDnMat(&C_desc, m, n, m, d_C, data_type, order));
    }

    ~Sptri_left_multiply_test() {
        CUSPARSE_CHECK(cusparseDestroySpMat(A_desc));
        CUSPARSE_CHECK(cusparseDestroyDnMat(B_desc));
        CUSPARSE_CHECK(cusparseDestroyDnMat(C_desc));

        CUSPARSE_CHECK(cusparseDestroy(cusparseH));
    }
};

using ValidTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(Sptri_left_multiply_test, ValidTypes);

TYPED_TEST(Sptri_left_multiply_test, IdentityStaysSame) {
    constexpr cusparseOperation_t op_type = CUSPARSE_OPERATION_NON_TRANSPOSE;

    sptri_left_multiply<TypeParam>(this->cusparseH, this->C_desc, op_type,
                                   this->A_desc, this->B_desc);

    thrust::host_vector<TypeParam> expected = this->B;
    thrust::host_vector<TypeParam> got = this->C;

    std::cerr << "Expected: ";
    for (int i = 0; i < 8; ++i) {
        std::cerr << expected[i] << std::endl;
    }
    std::cerr << std::endl;

    std::cerr << "Got: ";
    for (int i = 0; i < 8; ++i) {
        std::cerr << got[i] << " ";
    }
    std::cerr << std::endl;

    ASSERT_TRUE(match(expected, got));
}