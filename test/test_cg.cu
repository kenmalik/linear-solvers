#include <gtest/gtest.h>

#include <cublas_v2.h>
#include <cusparse_v2.h>

#include <mat_utils/mat_reader.h>

#include <cg_run/cg.h>
#include <cg_run/checks.h>
#include <cg_run/device_sparse_matrix.h>
#include <cg_run/device_vector.h>

#include <vector>

#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "data"
#endif

class CgTest : public ::testing::Test {
  protected:
    cusparseHandle_t cusparse;
    cublasHandle_t cublas;

    void SetUp() override {
        CUSPARSE_CHECK(cusparseCreate(&cusparse));
        CUBLAS_CHECK(cublasCreate_v2(&cublas));
    }

    void TearDown() override {
        CUBLAS_CHECK(cublasDestroy_v2(cublas));
        CUSPARSE_CHECK(cusparseDestroy(cusparse));
    }
};

TEST_F(CgTest, ConvergesOn1138Bus) {
    mat_utils::SpMatReader A_reader{
        TEST_DATA_DIR "/1138_bus.mat", {"Problem"}, "A"};
    mat_utils::SpMatReader L_reader{
        TEST_DATA_DIR "/1138_bus_ichol.mat", {}, "L"};
    mat_utils::DnMatReader b_reader{TEST_DATA_DIR "/1138_bus_b.mat", {}, "B"};

    cg_run::DeviceSparseMatrix<double> A{A_reader};
    cg_run::DeviceSparseMatrix<double> L{L_reader};
    cg_run::DeviceVector b{std::vector<double>(
        b_reader.data(), b_reader.data() + b_reader.rows())};
    cg_run::DeviceVector x{std::vector<double>(A_reader.rows(), 0.0)};

    cusparseFillMode_t fill_mode = CUSPARSE_FILL_MODE_LOWER;
    CUSPARSE_CHECK(cusparseSpMatSetAttribute(L.get(), CUSPARSE_SPMAT_FILL_MODE,
                                             &fill_mode, sizeof(fill_mode)));

    const double tolerance = 1e-6;
    const int max_iterations = static_cast<int>(A_reader.rows());

    int iterations = cg_run::cg(cusparse, cublas, A.get(), b.get(), x.get(),
                                L.get(), tolerance, max_iterations);

    EXPECT_LT(iterations, max_iterations)
        << "CG did not converge within " << max_iterations << " iterations";
}
