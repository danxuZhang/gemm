#include "gemm.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

namespace gemm {

#define BLOCK_SIZE 32

__global__ void dgemm_kernel(int M, int N, int K, double alpha,
                            const double* __restrict__ A,
                            const double* __restrict__ B,
                            double beta,
                            double* __restrict__ C) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        double sum = 0.0;
        for (int k = 0; k < K; ++k) {
            sum += A[row*K+k] * B[k*N + col];
        }
        C[row*N+col] = alpha * sum + beta * C[row*N+col];
    }
}

void dgemm_cuda(int m, int n, int k, double alpha, const double* a, int lda,
                const double* b, int ldb, double beta, double* c, int ldc) {
    // Allocate device memory
    double *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, m * k * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, k * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_c, m * n * sizeof(double)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_a, a, m * k * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, k * n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, c, m * n * sizeof(double), cudaMemcpyHostToDevice));

    // Set up grid and block dimensions
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(
        (n + BLOCK_SIZE - 1) / BLOCK_SIZE,    // Grid width
        (m + BLOCK_SIZE - 1) / BLOCK_SIZE     // Grid height
    );

    dgemm_kernel<<<numBlocks, threadsPerBlock>>>(m, n, k, alpha, d_a, d_b, beta, d_c);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(c, d_c, m * n * sizeof(double), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

} // namespace gemm