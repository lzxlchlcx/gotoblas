#include <immintrin.h>

int sgemm_kernel_avx2(int m, int n, int k,
                      float alpha,
                      const float *A,
                      const float *B,
                      float *C, int ldc)
{
    if (m < 8 || n < 4) {
        int i, j, p;
        for (j = 0; j < n; j++) {
            for (i = 0; i < m; i++) {
                float sum = 0.0f;
                for (p = 0; p < k; p++) {
                    sum += A[i + p * m] * B[p + j * k];
                }
                C[i + j * ldc] += alpha * sum;
            }
        }
        return 0;
    }

    __m256 c0 = _mm256_setzero_ps();
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();

    __m256 valpha = _mm256_set1_ps(alpha);

    int p;
    for (p = 0; p < k; p++) {
        __m256 a0 = _mm256_loadu_ps(&A[p * 8]);

        __m256 b0 = _mm256_set1_ps(B[p + 0 * k]);
        c0 = _mm256_fmadd_ps(a0, b0, c0);

        __m256 b1 = _mm256_set1_ps(B[p + 1 * k]);
        c1 = _mm256_fmadd_ps(a0, b1, c1);

        __m256 b2 = _mm256_set1_ps(B[p + 2 * k]);
        c2 = _mm256_fmadd_ps(a0, b2, c2);

        __m256 b3 = _mm256_set1_ps(B[p + 3 * k]);
        c3 = _mm256_fmadd_ps(a0, b3, c3);
    }

    c0 = _mm256_mul_ps(c0, valpha);
    c1 = _mm256_mul_ps(c1, valpha);
    c2 = _mm256_mul_ps(c2, valpha);
    c3 = _mm256_mul_ps(c3, valpha);

    __m256 c_old;
    c_old = _mm256_loadu_ps(&C[0 + 0 * ldc]);
    c0 = _mm256_add_ps(c_old, c0);
    _mm256_storeu_ps(&C[0 + 0 * ldc], c0);

    c_old = _mm256_loadu_ps(&C[0 + 1 * ldc]);
    c1 = _mm256_add_ps(c_old, c1);
    _mm256_storeu_ps(&C[0 + 1 * ldc], c1);

    c_old = _mm256_loadu_ps(&C[0 + 2 * ldc]);
    c2 = _mm256_add_ps(c_old, c2);
    _mm256_storeu_ps(&C[0 + 2 * ldc], c2);

    c_old = _mm256_loadu_ps(&C[0 + 3 * ldc]);
    c3 = _mm256_add_ps(c_old, c3);
    _mm256_storeu_ps(&C[0 + 3 * ldc], c3);

    return 0;
}
