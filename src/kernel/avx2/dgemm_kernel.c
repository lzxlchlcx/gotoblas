#include <immintrin.h>

int dgemm_kernel_avx2(int m, int n, int k,
                      double alpha,
                      const double *A,
                      const double *B,
                      double *C, int ldc)
{
    if (m < 4 || n < 4) {
        int i, j, p;
        for (j = 0; j < n; j++) {
            for (i = 0; i < m; i++) {
                double sum = 0.0;
                for (p = 0; p < k; p++) {
                    sum += A[i + p * m] * B[p + j * k];
                }
                C[i + j * ldc] += alpha * sum;
            }
        }
        return 0;
    }

    __m256d c0 = _mm256_setzero_pd();
    __m256d c1 = _mm256_setzero_pd();
    __m256d c2 = _mm256_setzero_pd();
    __m256d c3 = _mm256_setzero_pd();

    __m256d valpha = _mm256_set1_pd(alpha);

    int p;
    for (p = 0; p < k; p++) {
        __m256d a0 = _mm256_loadu_pd(&A[p * 4]);

        __m256d b0 = _mm256_set1_pd(B[p + 0 * k]);
        c0 = _mm256_fmadd_pd(a0, b0, c0);

        __m256d b1 = _mm256_set1_pd(B[p + 1 * k]);
        c1 = _mm256_fmadd_pd(a0, b1, c1);

        __m256d b2 = _mm256_set1_pd(B[p + 2 * k]);
        c2 = _mm256_fmadd_pd(a0, b2, c2);

        __m256d b3 = _mm256_set1_pd(B[p + 3 * k]);
        c3 = _mm256_fmadd_pd(a0, b3, c3);
    }

    c0 = _mm256_mul_pd(c0, valpha);
    c1 = _mm256_mul_pd(c1, valpha);
    c2 = _mm256_mul_pd(c2, valpha);
    c3 = _mm256_mul_pd(c3, valpha);

    __m256d c_old;
    c_old = _mm256_loadu_pd(&C[0 + 0 * ldc]);
    c0 = _mm256_add_pd(c_old, c0);
    _mm256_storeu_pd(&C[0 + 0 * ldc], c0);

    c_old = _mm256_loadu_pd(&C[0 + 1 * ldc]);
    c1 = _mm256_add_pd(c_old, c1);
    _mm256_storeu_pd(&C[0 + 1 * ldc], c1);

    c_old = _mm256_loadu_pd(&C[0 + 2 * ldc]);
    c2 = _mm256_add_pd(c_old, c2);
    _mm256_storeu_pd(&C[0 + 2 * ldc], c2);

    c_old = _mm256_loadu_pd(&C[0 + 3 * ldc]);
    c3 = _mm256_add_pd(c_old, c3);
    _mm256_storeu_pd(&C[0 + 3 * ldc], c3);

    return 0;
}
