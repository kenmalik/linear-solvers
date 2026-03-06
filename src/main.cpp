#include "cg/cg.h"
#include <iostream>
#include <cmath>

int main() {
    // Example: Solve a 3x3 symmetric positive-definite system
    // A = [4, 1, 0]
    //     [1, 3, 1]
    //     [0, 1, 2]
    std::vector<double> A = {
        4.0, 1.0, 0.0,
        1.0, 3.0, 1.0,
        0.0, 1.0, 2.0
    };

    // b = [1, 2, 3]
    std::vector<double> b = {1.0, 2.0, 3.0};

    // Initial guess x = [0, 0, 0]
    std::vector<double> x = {0.0, 0.0, 0.0};

    int iterations = cg(A, b, x, 1e-10, 100);

    std::cout << "Converged in " << iterations << " iterations\n";
    std::cout << "Solution: [";
    for (size_t i = 0; i < x.size(); ++i) {
        std::cout << x[i];
        if (i < x.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";

    // Verify: compute residual ||b - Ax||
    std::vector<double> residual = b;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            residual[i] -= A[i * 3 + j] * x[j];
        }
    }
    double norm = 0.0;
    for (int i = 0; i < 3; ++i) {
        norm += residual[i] * residual[i];
    }
    std::cout << "Residual norm: " << std::sqrt(norm) << "\n";

    return 0;
}
