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

__global__ void dgemm_kernel(int m, int n, int k, double alpha,
                            const double* __restrict__ a,
                            const double* __restrict__ b,
                            double beta,
                            double* __restrict__ c) {
    // Dummy kernel - just copy input to output
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m * n) {
        c[idx] = c[idx] * beta + alpha * a[idx % (m*k)] * b[idx % (k*n)];
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
    int blockSize = 256;
    int numBlocks = (m * n + blockSize - 1) / blockSize;

    // Launch dummy kernel
    dgemm_kernel<<<numBlocks, blockSize>>>(m, n, k, alpha, d_a, d_b, beta, d_c);
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