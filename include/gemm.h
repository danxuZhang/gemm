#pragma once

namespace gemm {

void dgemm_seq(int m, int n, int k,
               double alpha,
               const double* a, int lda,
               const double* b, int ldb,
               double beta,
               double* c, int ldc);

#ifdef WITH_MKL
void dgemm_mkl(int m, int n, int k,
               double alpha,
               const double* a, int lda,
               const double* b, int ldb,
               double beta,
               double* c, int ldc);
#endif

#ifdef WITH_CUBLAS
void dgemm_cublas(int m, int n, int k,
               double alpha,
               const double* a, int lda,
               const double* b, int ldb,
               double beta,
               double* c, int ldc);
#endif

#ifdef WITH_CUDA
void dgemm_cuda(int m, int n, int k,
               double alpha,
               const double* a, int lda,
               const double* b, int ldb,
               double beta,
               double* c, int ldc);
#endif

} // namespace gemm
