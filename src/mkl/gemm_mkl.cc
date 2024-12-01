#include "gemm.h"
#include <mkl.h>

namespace gemm {

void dgemm_mkl(int m, int n, int k, double alpha, const double *a, int lda,
               const double *b, int ldb, double beta, double *c, int ldc) {
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, lda,
              b, ldb, beta, c, ldc);
}

} // namespace gemm
