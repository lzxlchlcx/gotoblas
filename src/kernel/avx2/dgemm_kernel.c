#include <immintrin.h>
#include "config/haswell.h"

#define MR GEMM_HASWELL_D_MR
#define NR GEMM_HASWELL_D_NR

static int dgemm_kernel_scalar(int m, int n, int k,
                               double alpha,
                               const double *A,
                               const double *B,
                               double *C, int ldc)
{
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

#if MR == 8 && NR == 6

static int dgemm_kernel_fast(int m, int n, int k,
                             double alpha,
                             const double *A,
                              const double *B,
                             double *C, int ldc)
{
    __m256d c00 = _mm256_setzero_pd();
    __m256d c01 = _mm256_setzero_pd();
    __m256d c10 = _mm256_setzero_pd();
    __m256d c11 = _mm256_setzero_pd();
    __m256d c20 = _mm256_setzero_pd();
    __m256d c21 = _mm256_setzero_pd();
    __m256d c30 = _mm256_setzero_pd();
    __m256d c31 = _mm256_setzero_pd();
    __m256d c40 = _mm256_setzero_pd();
    __m256d c41 = _mm256_setzero_pd();
    __m256d c50 = _mm256_setzero_pd();
    __m256d c51 = _mm256_setzero_pd();

    int p;
    for (p = 0; p < k; p++) {
        __m256d a0 = _mm256_loadu_pd(&A[p * 8]);
        __m256d a1 = _mm256_loadu_pd(&A[p * 8 + 4]);

        __m256d b = _mm256_set1_pd(B[p + 0 * k]);
        c00 = _mm256_fmadd_pd(a0, b, c00);
        c01 = _mm256_fmadd_pd(a1, b, c01);

        b = _mm256_set1_pd(B[p + 1 * k]);
        c10 = _mm256_fmadd_pd(a0, b, c10);
        c11 = _mm256_fmadd_pd(a1, b, c11);

        b = _mm256_set1_pd(B[p + 2 * k]);
        c20 = _mm256_fmadd_pd(a0, b, c20);
        c21 = _mm256_fmadd_pd(a1, b, c21);

        b = _mm256_set1_pd(B[p + 3 * k]);
        c30 = _mm256_fmadd_pd(a0, b, c30);
        c31 = _mm256_fmadd_pd(a1, b, c31);

        b = _mm256_set1_pd(B[p + 4 * k]);
        c40 = _mm256_fmadd_pd(a0, b, c40);
        c41 = _mm256_fmadd_pd(a1, b, c41);

        b = _mm256_set1_pd(B[p + 5 * k]);
        c50 = _mm256_fmadd_pd(a0, b, c50);
        c51 = _mm256_fmadd_pd(a1, b, c51);
    }

    __m256d valpha = _mm256_set1_pd(alpha);

    c00 = _mm256_mul_pd(c00, valpha);
    c01 = _mm256_mul_pd(c01, valpha);
    c10 = _mm256_mul_pd(c10, valpha);
    c11 = _mm256_mul_pd(c11, valpha);
    c20 = _mm256_mul_pd(c20, valpha);
    c21 = _mm256_mul_pd(c21, valpha);
    c30 = _mm256_mul_pd(c30, valpha);
    c31 = _mm256_mul_pd(c31, valpha);
    c40 = _mm256_mul_pd(c40, valpha);
    c41 = _mm256_mul_pd(c41, valpha);
    c50 = _mm256_mul_pd(c50, valpha);
    c51 = _mm256_mul_pd(c51, valpha);

    __m256d t;
    t  = _mm256_loadu_pd(&C[0 + 0 * ldc]);
    c00 = _mm256_add_pd(t, c00);
    _mm256_storeu_pd(&C[0 + 0 * ldc], c00);

    t  = _mm256_loadu_pd(&C[4 + 0 * ldc]);
    c01 = _mm256_add_pd(t, c01);
    _mm256_storeu_pd(&C[4 + 0 * ldc], c01);

    t  = _mm256_loadu_pd(&C[0 + 1 * ldc]);
    c10 = _mm256_add_pd(t, c10);
    _mm256_storeu_pd(&C[0 + 1 * ldc], c10);

    t  = _mm256_loadu_pd(&C[4 + 1 * ldc]);
    c11 = _mm256_add_pd(t, c11);
    _mm256_storeu_pd(&C[4 + 1 * ldc], c11);

    t  = _mm256_loadu_pd(&C[0 + 2 * ldc]);
    c20 = _mm256_add_pd(t, c20);
    _mm256_storeu_pd(&C[0 + 2 * ldc], c20);

    t  = _mm256_loadu_pd(&C[4 + 2 * ldc]);
    c21 = _mm256_add_pd(t, c21);
    _mm256_storeu_pd(&C[4 + 2 * ldc], c21);

    t  = _mm256_loadu_pd(&C[0 + 3 * ldc]);
    c30 = _mm256_add_pd(t, c30);
    _mm256_storeu_pd(&C[0 + 3 * ldc], c30);

    t  = _mm256_loadu_pd(&C[4 + 3 * ldc]);
    c31 = _mm256_add_pd(t, c31);
    _mm256_storeu_pd(&C[4 + 3 * ldc], c31);

    t  = _mm256_loadu_pd(&C[0 + 4 * ldc]);
    c40 = _mm256_add_pd(t, c40);
    _mm256_storeu_pd(&C[0 + 4 * ldc], c40);

    t  = _mm256_loadu_pd(&C[4 + 4 * ldc]);
    c41 = _mm256_add_pd(t, c41);
    _mm256_storeu_pd(&C[4 + 4 * ldc], c41);

    t  = _mm256_loadu_pd(&C[0 + 5 * ldc]);
    c50 = _mm256_add_pd(t, c50);
    _mm256_storeu_pd(&C[0 + 5 * ldc], c50);

    t  = _mm256_loadu_pd(&C[4 + 5 * ldc]);
    c51 = _mm256_add_pd(t, c51);
    _mm256_storeu_pd(&C[4 + 5 * ldc], c51);

    return 0;
}

#elif MR == 4 && NR == 8

static int dgemm_kernel_fast(int m, int n, int k,
                             double alpha,
                             const double *A,
                             const double *B,
                             double *C, int ldc)
{
    __m256d c0 = _mm256_setzero_pd();
    __m256d c1 = _mm256_setzero_pd();
    __m256d c2 = _mm256_setzero_pd();
    __m256d c3 = _mm256_setzero_pd();
    __m256d c4 = _mm256_setzero_pd();
    __m256d c5 = _mm256_setzero_pd();
    __m256d c6 = _mm256_setzero_pd();
    __m256d c7 = _mm256_setzero_pd();

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

        __m256d b4 = _mm256_set1_pd(B[p + 4 * k]);
        c4 = _mm256_fmadd_pd(a0, b4, c4);

        __m256d b5 = _mm256_set1_pd(B[p + 5 * k]);
        c5 = _mm256_fmadd_pd(a0, b5, c5);

        __m256d b6 = _mm256_set1_pd(B[p + 6 * k]);
        c6 = _mm256_fmadd_pd(a0, b6, c6);

        __m256d b7 = _mm256_set1_pd(B[p + 7 * k]);
        c7 = _mm256_fmadd_pd(a0, b7, c7);
    }

    c0 = _mm256_mul_pd(c0, valpha);
    c1 = _mm256_mul_pd(c1, valpha);
    c2 = _mm256_mul_pd(c2, valpha);
    c3 = _mm256_mul_pd(c3, valpha);
    c4 = _mm256_mul_pd(c4, valpha);
    c5 = _mm256_mul_pd(c5, valpha);
    c6 = _mm256_mul_pd(c6, valpha);
    c7 = _mm256_mul_pd(c7, valpha);

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

    c_old = _mm256_loadu_pd(&C[0 + 4 * ldc]);
    c4 = _mm256_add_pd(c_old, c4);
    _mm256_storeu_pd(&C[0 + 4 * ldc], c4);

    c_old = _mm256_loadu_pd(&C[0 + 5 * ldc]);
    c5 = _mm256_add_pd(c_old, c5);
    _mm256_storeu_pd(&C[0 + 5 * ldc], c5);

    c_old = _mm256_loadu_pd(&C[0 + 6 * ldc]);
    c6 = _mm256_add_pd(c_old, c6);
    _mm256_storeu_pd(&C[0 + 6 * ldc], c6);

    c_old = _mm256_loadu_pd(&C[0 + 7 * ldc]);
    c7 = _mm256_add_pd(c_old, c7);
    _mm256_storeu_pd(&C[0 + 7 * ldc], c7);

    return 0;
}

#else

static int dgemm_kernel_fast(int m, int n, int k,
                             double alpha,
                             const double *A,
                             const double *B,
                             double *C, int ldc)
{
    __m256d valpha = _mm256_set1_pd(alpha);
    int ii, jj, p;

    for (jj = 0; jj < n; jj += 4) {
        int jrem = n - jj;
        if (jrem > 4) jrem = 4;
        for (ii = 0; ii < m; ii += 4) {
            int irem = m - ii;
            if (irem > 4) irem = 4;

            if (irem < 4 || jrem < 4) {
                int i, j;
                for (j = 0; j < jrem; j++) {
                    for (i = 0; i < irem; i++) {
                        double sum = 0.0;
                        for (p = 0; p < k; p++) {
                            sum += A[ii + i + p * m] * B[p + (jj + j) * k];
                        }
                        C[ii + i + (jj + j) * ldc] += alpha * sum;
                    }
                }
            } else {
                __m256d c0 = _mm256_setzero_pd();
                __m256d c1 = _mm256_setzero_pd();
                __m256d c2 = _mm256_setzero_pd();
                __m256d c3 = _mm256_setzero_pd();

                for (p = 0; p < k; p++) {
                    __m256d a = _mm256_loadu_pd(&A[ii + p * m]);
                    __m256d b0 = _mm256_set1_pd(B[p + (jj + 0) * k]);
                    __m256d b1 = _mm256_set1_pd(B[p + (jj + 1) * k]);
                    __m256d b2 = _mm256_set1_pd(B[p + (jj + 2) * k]);
                    __m256d b3 = _mm256_set1_pd(B[p + (jj + 3) * k]);

                    c0 = _mm256_fmadd_pd(a, b0, c0);
                    c1 = _mm256_fmadd_pd(a, b1, c1);
                    c2 = _mm256_fmadd_pd(a, b2, c2);
                    c3 = _mm256_fmadd_pd(a, b3, c3);
                }

                c0 = _mm256_mul_pd(c0, valpha);
                c1 = _mm256_mul_pd(c1, valpha);
                c2 = _mm256_mul_pd(c2, valpha);
                c3 = _mm256_mul_pd(c3, valpha);

                __m256d t;
                t = _mm256_loadu_pd(&C[ii + (jj + 0) * ldc]);
                _mm256_storeu_pd(&C[ii + (jj + 0) * ldc], _mm256_add_pd(t, c0));
                t = _mm256_loadu_pd(&C[ii + (jj + 1) * ldc]);
                _mm256_storeu_pd(&C[ii + (jj + 1) * ldc], _mm256_add_pd(t, c1));
                t = _mm256_loadu_pd(&C[ii + (jj + 2) * ldc]);
                _mm256_storeu_pd(&C[ii + (jj + 2) * ldc], _mm256_add_pd(t, c2));
                t = _mm256_loadu_pd(&C[ii + (jj + 3) * ldc]);
                _mm256_storeu_pd(&C[ii + (jj + 3) * ldc], _mm256_add_pd(t, c3));
            }
        }
    }
    return 0;
}

#endif

int dgemm_kernel_avx2(int m, int n, int k,
                      double alpha,
                      const double *A,
                      const double *B,
                      double *C, int ldc)
{
    if (m < MR || n < NR) {
        return dgemm_kernel_scalar(m, n, k, alpha, A, B, C, ldc);
    }
    return dgemm_kernel_fast(m, n, k, alpha, A, B, C, ldc);
}
