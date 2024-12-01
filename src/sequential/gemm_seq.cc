#include "gemm.h"

namespace gemm {

void dgemm_seq(int m, int n, int k, double alpha, const double *a, int lda,
               const double *b, int ldb, double beta, double *c, int ldc) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      c[i * ldc + j] *= beta;
    }
  }

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0.0;
      for (int p = 0; p < k; p++) {
        sum += a[i * lda + p] * b[p * ldb + j];
      }
      c[i * ldc + j] += alpha * sum;
    }
  }
}

} // namespace gemm
