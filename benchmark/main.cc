#include "gemm.h"
#include <chrono>
#include <functional>
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>

struct BenchmarkResult {
    double min_time;
    double max_time;
    double avg_time;
    double stddev;
    double gflops;
    std::vector<double> times;
};

class GemmBenchmark {
private:
    // Test configurations
    const std::vector<std::tuple<int, int, int>> test_sizes = {
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
        {8192, 8192, 8192},
        // Non-square matrices
        {2048, 1024, 4096},
        {4096, 2048, 1024},
        // Odd sizes to test padding/alignment
        {1023, 2047, 4095},
    };
    
    const int num_warmup = 3;
    const int num_runs = 10;
    const double alpha = 1.0;
    const double beta = 0.0;
    const double validation_tolerance = 1e-6;

    // Helper functions
    void generate_random_matrix(std::vector<double>& matrix, int rows, int cols) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<> dis(-1.0, 1.0);
        
        #pragma omp parallel for
        for (int i = 0; i < rows * cols; ++i) {
            matrix[i] = dis(gen);
        }
    }

    BenchmarkResult run_single_benchmark(
        const std::string& impl_name,
        std::function<void(int, int, int, double, const double*, int, 
                          const double*, int, double, double*, int)> gemm_func,
        int M, int N, int K) {
        
        std::vector<double> A(M * K);
        std::vector<double> B(K * N);
        std::vector<double> C(M * N);
        
        generate_random_matrix(A, M, K);
        generate_random_matrix(B, K, N);
        
        // Warm-up runs
        for (int i = 0; i < num_warmup; ++i) {
            gemm_func(M, N, K, alpha, A.data(), K, B.data(), N, beta, C.data(), N);
        }
        
        // Timed runs
        std::vector<double> times;
        times.reserve(num_runs);
        
        for (int i = 0; i < num_runs; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            gemm_func(M, N, K, alpha, A.data(), K, B.data(), N, beta, C.data(), N);
            auto end = std::chrono::high_resolution_clock::now();
            
            double time = std::chrono::duration<double>(end - start).count();
            times.push_back(time);
        }
        
        // Calculate statistics
        double min_time = *std::min_element(times.begin(), times.end());
        double max_time = *std::max_element(times.begin(), times.end());
        double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / num_runs;
        
        double variance = 0.0;
        for (double time : times) {
            variance += (time - avg_time) * (time - avg_time);
        }
        double stddev = std::sqrt(variance / num_runs);
        
        // Calculate GFLOPS (2*M*N*K for matrix multiplication)
        double gflops = (2.0 * M * N * K) / (min_time * 1e9);
        
        return {min_time, max_time, avg_time, stddev, gflops, times};
    }

    void print_results(const std::string& impl_name, 
                      const BenchmarkResult& result,
                      int M, int N, int K) {
        std::cout << std::fixed << std::setprecision(3)
                  << impl_name << " Results (M=" << M << ", N=" << N << ", K=" << K << "):\n"
                  << "  Min time:  " << result.min_time * 1000 << " ms\n"
                  << "  Max time:  " << result.max_time * 1000 << " ms\n"
                  << "  Avg time:  " << result.avg_time * 1000 << " ms\n"
                  << "  Std dev:   " << result.stddev * 1000 << " ms\n"
                  << "  GFLOPS:    " << result.gflops << "\n\n";
    }

public:
    void run_benchmarks() {
        std::cout << "Running GEMM benchmarks with:\n"
                  << "  Warm-up runs: " << num_warmup << "\n"
                  << "  Timed runs:   " << num_runs << "\n\n";

        for (const auto& [M, N, K] : test_sizes) {
            std::cout << "\nTesting size M=" << M << ", N=" << N << ", K=" << K << "\n";
            std::cout << "========================================\n";

#ifdef WITH_CUDA
            auto cuda_result = run_single_benchmark(
                "Custom CUDA", gemm::dgemm_cuda, M, N, K);
            print_results("Custom CUDA", cuda_result, M, N, K);
#endif

#ifdef WITH_CUBLAS
            auto cublas_result = run_single_benchmark(
                "cuBLAS", gemm::dgemm_cublas, M, N, K);
            print_results("cuBLAS", cublas_result, M, N, K);
#endif


#ifdef WITH_MKL
            auto mkl_result = run_single_benchmark(
                "MKL", gemm::dgemm_mkl, M, N, K);
            print_results("MKL", mkl_result, M, N, K);
#endif
        }
    }

    void validate_implementations() {
        // Use a smaller size for validation
        const int M = 1024, N = 1024, K = 1024;
        std::vector<double> A(M * K);
        std::vector<double> B(K * N);
        std::vector<double> C_ref(M * N);
        std::vector<double> C_test(M * N);

        generate_random_matrix(A, M, K);
        generate_random_matrix(B, K, N);

        // Generate reference result using MKL or sequential implementation
#ifdef WITH_MKL
        gemm::dgemm_mkl(M, N, K, alpha, A.data(), K, B.data(), N, beta, C_ref.data(), N);
#else
        gemm::dgemm_seq(M, N, K, alpha, A.data(), K, B.data(), N, beta, C_ref.data(), N);
#endif

        //  each implementation
#ifdef WITH_CUBLAS
        gemm::dgemm_cublas(M, N, K, alpha, A.data(), K, B.data(), N, beta, C_test.data(), N);
#endif

#ifdef WITH_CUDA
        gemm::dgemm_cuda(M, N, K, alpha, A.data(), K, B.data(), N, beta, C_test.data(), N);
#endif
    }
};

int main() {
    GemmBenchmark benchmark;
    
    std::cout << "Validating implementations...\n";
    benchmark.validate_implementations();
    
    std::cout << "\nRunning performance benchmarks...\n";
    benchmark.run_benchmarks();
    
    return 0;
}
