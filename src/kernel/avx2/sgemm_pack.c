#include <immintrin.h>

int sgemm_pack_a_nn_avx2(int m, int k, const float *A, int lda, float *A_packed)
{
    int i, p;
    int m8 = m & ~7;
    for (p = 0; p < k; p++) {
        for (i = 0; i < m8; i += 8) {
            __m256 a = _mm256_loadu_ps(&A[i + p * lda]);
            _mm256_storeu_ps(&A_packed[i + p * m], a);
        }
        for (; i < m; i++) {
            A_packed[i + p * m] = A[i + p * lda];
        }
    }
    return 0;
}

int sgemm_pack_a_tn_avx2(int m, int k, const float *A, int lda, float *A_packed)
{
    int i, p;
    int m8 = m & ~7;
    for (p = 0; p < k; p++) {
        for (i = 0; i < m8; i += 8) {
            float t0 = A[p + (i+0) * lda];
            float t1 = A[p + (i+1) * lda];
            float t2 = A[p + (i+2) * lda];
            float t3 = A[p + (i+3) * lda];
            float t4 = A[p + (i+4) * lda];
            float t5 = A[p + (i+5) * lda];
            float t6 = A[p + (i+6) * lda];
            float t7 = A[p + (i+7) * lda];
            __m256 a = _mm256_set_ps(t7, t6, t5, t4, t3, t2, t1, t0);
            _mm256_storeu_ps(&A_packed[i + p * m], a);
        }
        for (; i < m; i++) {
            A_packed[i + p * m] = A[p + i * lda];
        }
    }
    return 0;
}

int sgemm_pack_b_nn_avx2(int k, int n, const float *B, int ldb, float *B_packed)
{
    int p, j;
    for (j = 0; j < n; j++) {
        for (p = 0; p < k; p++) {
            B_packed[p + j * k] = B[p + j * ldb];
        }
    }
    return 0;
}

int sgemm_pack_b_tn_avx2(int k, int n, const float *B, int ldb, float *B_packed)
{
    int p, j;
    for (j = 0; j < n; j++) {
        for (p = 0; p < k; p++) {
            B_packed[p + j * k] = B[j + p * ldb];
        }
    }
    return 0;
}
