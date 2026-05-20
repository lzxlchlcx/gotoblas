#include <immintrin.h>

int sgemm_beta_avx2(int m, int n, float beta, float *C, int ldc)
{
    int i, j;
    int m8 = m & ~7;

    if (beta == 0.0f) {
        __m256 vzero = _mm256_setzero_ps();
        for (j = 0; j < n; j++) {
            for (i = 0; i < m8; i += 8) {
                _mm256_storeu_ps(&C[i + j * ldc], vzero);
            }
            for (; i < m; i++) {
                C[i + j * ldc] = 0.0f;
            }
        }
    } else if (beta != 1.0f) {
        __m256 vbeta = _mm256_set1_ps(beta);
        for (j = 0; j < n; j++) {
            for (i = 0; i < m8; i += 8) {
                __m256 c = _mm256_loadu_ps(&C[i + j * ldc]);
                c = _mm256_mul_ps(c, vbeta);
                _mm256_storeu_ps(&C[i + j * ldc], c);
            }
            for (; i < m; i++) {
                C[i + j * ldc] *= beta;
            }
        }
    }
    return 0;
}
