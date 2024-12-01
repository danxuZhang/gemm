#include "gemm.h"
#include <chrono>
#include <iostream>
#include <sstream>
#include <random>
#include <vector>

void generate_random_matrix(std::vector<double> &matrix, int rows, int cols) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);

  for (int i = 0; i < rows * cols; ++i) {
    matrix[i] = dis(gen);
  }
}

bool validate_results(const std::vector<double>& reference, 
                     const std::vector<double>& result, 
                     const std::string& implementation_name,
                     double tolerance = 1e-6) {
    if (reference.size() != result.size()) {
        std::cout << "Error: Size mismatch between reference and " 
                  << implementation_name << " results\n";
        return false;
    }

    double max_diff = 0.0;
    double max_rel_diff = 0.0;
    int error_count = 0;

    for (size_t i = 0; i < reference.size(); ++i) {
        double abs_diff = std::abs(reference[i] - result[i]);
        double rel_diff = abs_diff / (std::abs(reference[i]) + tolerance);
        
        max_diff = std::max(max_diff, abs_diff);
        max_rel_diff = std::max(max_rel_diff, rel_diff);
        
        if (rel_diff > tolerance) {
            error_count++;
        }
    }

    std::cout << implementation_name << " validation results:\n"
              << "  Maximum absolute difference: " << max_diff << "\n"
              << "  Maximum relative difference: " << max_rel_diff << "\n"
              << "  Elements exceeding tolerance: " << error_count << "/"
              << reference.size() << "\n\n";

    return error_count == 0;
}

template <typename Func> double measure_time(Func &&func) {
  auto start = std::chrono::high_resolution_clock::now();
  func();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double>(end - start).count();
}

int main() {
  const double alpha = 0.8;
  const double beta = 0.2;
  const int M = 1024;
  const int N = 1024;
  const int K = 1024;

  std::ostringstream oss;
  oss << "Benchmark on A(" << M << "," << K << ")xB(" << K << "," << N << ")+C(" << M << "," << N << ")\n";
  std::cout << oss.str() << std::endl;

  // Initialize matrices
  std::vector<double> A(M * K);
  std::vector<double> B(K * N);
  std::vector<double> C_seq(M * N);

  // Generate random data
  generate_random_matrix(A, M, K);
  generate_random_matrix(B, K, N);

  double time_seq = measure_time([&]() {
    gemm::dgemm_seq(M, N, K, alpha, A.data(), K, B.data(), N, beta, C_seq.data(),
                    N);
  });
  std::cout << "DGEMM (sequential) time: " << time_seq << " seconds\n";

#ifdef WITH_MKL
  std::vector<double> C_mkl(M * N);
  double time_mkl = measure_time([&]() {
    gemm::dgemm_mkl(M, N, K, alpha, A.data(), K, B.data(), N, beta, C_mkl.data(),
                    N);
  });
  std::cout << "DGEMM (sequential mkl) time: " << time_mkl << " seconds\n";
  validate_results(C_seq, C_mkl, "MKL");
#endif

#ifdef WITH_CUBLAS
  std::vector<double> C_cublas(M * N);
  double time_cublas = measure_time([&]() {
    gemm::dgemm_cublas(M, N, K, alpha, A.data(), K, B.data(), N, beta, C_cublas.data(), N);
  });
  std::cout << "DGEMM (cuBlas) time: " << time_cublas << " seconds\n";
  validate_results(C_seq, C_cublas, "cuBLAS");
#endif

#ifdef WITH_CUDA
    std::vector<double> C_cuda(M * N);
    double time_cuda = measure_time([&]() {
        gemm::dgemm_cuda(M, N, K, alpha, A.data(), K, B.data(), N, beta, C_cuda.data(), N);
    });
    std::cout << "DGEMM (myCUDA) time: " << time_cuda << " seconds\n";
    validate_results(C_seq, C_cuda, "MyCUDA");
#endif

  return 0;
}
