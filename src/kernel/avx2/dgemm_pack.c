#include <immintrin.h>

int dgemm_pack_a_nn_avx2(int m, int k, const double *A, int lda, double *A_packed)
{
    int i, p;
    int m4 = m & ~3;
    for (p = 0; p < k; p++) {
        for (i = 0; i < m4; i += 4) {
            __m256d a = _mm256_loadu_pd(&A[i + p * lda]);
            _mm256_storeu_pd(&A_packed[i + p * m], a);
        }
        for (; i < m; i++) {
            A_packed[i + p * m] = A[i + p * lda];
        }
    }
    return 0;
}

int dgemm_pack_a_tn_avx2(int m, int k, const double *A, int lda, double *A_packed)
{
    int i, p;
    int m4 = m & ~3;
    for (p = 0; p < k; p++) {
        for (i = 0; i < m4; i += 4) {
            double t0 = A[p + (i+0) * lda];
            double t1 = A[p + (i+1) * lda];
            double t2 = A[p + (i+2) * lda];
            double t3 = A[p + (i+3) * lda];
            __m256d a = _mm256_set_pd(t3, t2, t1, t0);
            _mm256_storeu_pd(&A_packed[i + p * m], a);
        }
        for (; i < m; i++) {
            A_packed[i + p * m] = A[p + i * lda];
        }
    }
    return 0;
}

int dgemm_pack_b_nn_avx2(int k, int n, const double *B, int ldb, double *B_packed)
{
    int p, j;
    int k4 = k & ~3;
    for (j = 0; j < n; j++) {
        const double *src = B + j * ldb;
        double *dst = B_packed + j * k;
        for (p = 0; p < k4; p += 4) {
            __m256d b = _mm256_loadu_pd(&src[p]);
            _mm256_storeu_pd(&dst[p], b);
        }
        for (; p < k; p++) {
            dst[p] = src[p];
        }
    }
    return 0;
}

int dgemm_pack_b_tn_avx2(int k, int n, const double *B, int ldb, double *B_packed)
{
    int p, j;
    int n4 = n & ~3;

    for (j = 0; j < n4; j += 4) {
        for (p = 0; p < k; p++) {
            __m256d v = _mm256_set_pd(
                B[(j+3) + p * ldb],
                B[(j+2) + p * ldb],
                B[(j+1) + p * ldb],
                B[(j+0) + p * ldb]);
            _mm256_storeu_pd(&B_packed[p + j * k], v);
        }
    }
    for (; j < n; j++) {
        for (p = 0; p < k; p++) {
            B_packed[p + j * k] = B[j + p * ldb];
        }
    }
    return 0;
}
