#include "gemm.h"
#include <chrono>
#include <iostream>
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

template <typename Func> double measure_time(Func &&func) {
  auto start = std::chrono::high_resolution_clock::now();
  func();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double>(end - start).count();
}

int main() {
  const int M = 1024;
  const int N = 1024;
  const int K = 1024;

  // Initialize matrices
  std::vector<double> A(M * K);
  std::vector<double> B(K * N);
  std::vector<double> C_seq(M * N);

  // Generate random data
  generate_random_matrix(A, M, K);
  generate_random_matrix(B, K, N);

  // Test sequential implementation
  double time_seq = measure_time([&]() {
    gemm::dgemm_seq(M, N, K, 1.0, A.data(), K, B.data(), N, 0.0, C_seq.data(),
                    N);
  });
  std::cout << "DGEMM (sequential) time: " << time_seq << " seconds\n";

#ifdef WITH_MKL
  std::vector<double> C_mkl(M * N);
  double time_mkl = measure_time([&]() {
    gemm::dgemm_mkl(M, N, K, 1.0, A.data(), K, B.data(), N, 0.0, C_mkl.data(),
                    N);
  });
  std::cout << "DGEMM (sequential mkl) time: " << time_mkl << " seconds\n";
#endif

#ifdef WITH_CUBLAS
  std::vector<double> C_cublas(M * N);
  double time_cuda = measure_time([&]() {
    gemm::dgemm_cublas(M, N, K, 1.0, A.data(), K, B.data(), N, 0.0, C_cublas.data(), N);
  });
  std::cout << "DGEMM (cuBlas) time: " << time_cuda << " seconds\n";
#endif

#ifdef WITH_CUDA
    std::vector<double> C_cuda(M * N);
    double time_cuda = measure_time([&]() {
        gemm::dgemm_cuda(M, N, K, 1.0, A.data(), K, B.data(), N, 0.0, C_cuda.data(), N);
    });
    std::cout << "DGEMM (myCUDA) time: " << time_cuda << " seconds\n";
#endif


  return 0;
}
