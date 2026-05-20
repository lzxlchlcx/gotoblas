#include <immintrin.h>

int dgemm_beta_avx2(int m, int n, double beta, double *C, int ldc)
{
    int i, j;
    int m4 = m & ~3;

    if (beta == 0.0) {
        __m256d vzero = _mm256_setzero_pd();
        for (j = 0; j < n; j++) {
            for (i = 0; i < m4; i += 4) {
                _mm256_storeu_pd(&C[i + j * ldc], vzero);
            }
            for (; i < m; i++) {
                C[i + j * ldc] = 0.0;
            }
        }
    } else if (beta != 1.0) {
        __m256d vbeta = _mm256_set1_pd(beta);
        for (j = 0; j < n; j++) {
            for (i = 0; i < m4; i += 4) {
                __m256d c = _mm256_loadu_pd(&C[i + j * ldc]);
                c = _mm256_mul_pd(c, vbeta);
                _mm256_storeu_pd(&C[i + j * ldc], c);
            }
            for (; i < m; i++) {
                C[i + j * ldc] *= beta;
            }
        }
    }
    return 0;
}
