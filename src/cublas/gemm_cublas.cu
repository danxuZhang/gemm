#include "gemm.h"
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace gemm {

void handleCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s\n", message, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

void handleCublasError(cublasStatus_t status, const char* message) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error: %s\n", message);
        exit(EXIT_FAILURE);
    }
}

void dgemm_cublas(int m, int n, int k, double alpha, const double* a, int lda,
                const double* b, int ldb, double beta, double* c, int ldc) {
    // Create cuBLAS handle
    cublasHandle_t handle;
    handleCublasError(cublasCreate(&handle), "cublasCreate failed");

    cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    // Allocate device memory
    double *d_a, *d_b, *d_c;
    handleCudaError(cudaMalloc(&d_a, m * k * sizeof(double)), "cudaMalloc d_a failed");
    handleCudaError(cudaMalloc(&d_b, k * n * sizeof(double)), "cudaMalloc d_b failed");
    handleCudaError(cudaMalloc(&d_c, m * n * sizeof(double)), "cudaMalloc d_c failed");

    // Copy data to device
    handleCudaError(cudaMemcpy(d_a, a, m * k * sizeof(double), cudaMemcpyHostToDevice),
                   "cudaMemcpy a failed");
    handleCudaError(cudaMemcpy(d_b, b, k * n * sizeof(double), cudaMemcpyHostToDevice),
                   "cudaMemcpy b failed");

    // Perform DGEMM
    // Note: cuBLAS uses column-major order, while our input is row-major
    // So we compute B * A instead of A * B and adjust the parameters accordingly
    handleCublasError(
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                    n, m, k,
                    &alpha,
                    d_b, ldb,
                    d_a, lda,
                    &beta,
                    d_c, ldc),
        "cublasDgemm failed");

    // Copy result back to host
    handleCudaError(cudaMemcpy(c, d_c, m * n * sizeof(double), cudaMemcpyDeviceToHost),
                   "cudaMemcpy c failed");

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cublasDestroy(handle);
}

} // namespace gemm