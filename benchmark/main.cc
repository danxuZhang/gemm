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
  std::vector<double> C_mkl(M * N);

  // Generate random data
  generate_random_matrix(A, M, K);
  generate_random_matrix(B, K, N);

  // Test sequential implementation
  double time_seq = measure_time([&]() {
    gemm::dgemm_seq(M, N, K, 1.0, A.data(), K, B.data(), N, 0.0, C_seq.data(),
                    N);
  });
  std::cout << "Sequential DGEMM time: " << time_seq << " seconds\n";

#ifdef WITH_MKL
  // Test MKL implementation
  double time_mkl = measure_time([&]() {
    gemm::dgemm_mkl(M, N, K, 1.0, A.data(), K, B.data(), N, 0.0, C_mkl.data(),
                    N);
  });
  std::cout << "MKL DGEMM time: " << time_mkl << " seconds\n";
#endif

  return 0;
}
