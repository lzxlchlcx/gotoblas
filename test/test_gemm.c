#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "myblas.h"

#define EPS_D 1e-10
#define EPS_S 1e-4f

static void naive_dgemm(char ta, char tb, int m, int n, int k,
                        double alpha, const double *A, int lda,
                                      const double *B, int ldb,
                        double beta, double *C, int ldc)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            C[i + j * ldc] *= beta;
        }
    }
    for (int j = 0; j < n; j++) {
        for (int p = 0; p < k; p++) {
            double bval;
            if (tb == 'N' || tb == 'n') bval = B[p + j * ldb];
            else                        bval = B[j + p * ldb];
            for (int i = 0; i < m; i++) {
                double aval;
                if (ta == 'N' || ta == 'n') aval = A[i + p * lda];
                else                        aval = A[p + i * lda];
                C[i + j * ldc] += alpha * aval * bval;
            }
        }
    }
}

static void naive_sgemm(char ta, char tb, int m, int n, int k,
                        float alpha, const float *A, int lda,
                                     const float *B, int ldb,
                        float beta, float *C, int ldc)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            C[i + j * ldc] *= beta;
        }
    }
    for (int j = 0; j < n; j++) {
        for (int p = 0; p < k; p++) {
            float bval;
            if (tb == 'N' || tb == 'n') bval = B[p + j * ldb];
            else                        bval = B[j + p * ldb];
            for (int i = 0; i < m; i++) {
                float aval;
                if (ta == 'N' || ta == 'n') aval = A[i + p * lda];
                else                        aval = A[p + i * lda];
                C[i + j * ldc] += alpha * aval * bval;
            }
        }
    }
}

static int compare_d(const double *C_ref, const double *C, int m, int n, int ldc, double eps)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            double diff = fabs(C_ref[i + j * ldc] - C[i + j * ldc]);
            double denom = fabs(C_ref[i + j * ldc]);
            if (denom < 1e-15) denom = 1.0;
            if (diff > eps && diff / denom > eps) {
                printf("  FAIL at (%d,%d): ref=%e, got=%e, rel_err=%e\n",
                       i, j, C_ref[i + j * ldc], C[i + j * ldc], diff / denom);
                return -1;
            }
        }
    }
    return 0;
}

static int compare_s(const float *C_ref, const float *C, int m, int n, int ldc, float eps)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            float diff = fabsf(C_ref[i + j * ldc] - C[i + j * ldc]);
            float denom = fabsf(C_ref[i + j * ldc]);
            if (denom < 1e-7f) denom = 1.0f;
            if (diff > eps && diff / denom > eps) {
                printf("  FAIL at (%d,%d): ref=%e, got=%e, rel_err=%e\n",
                       i, j, C_ref[i + j * ldc], C[i + j * ldc], diff / denom);
                return -1;
            }
        }
    }
    return 0;
}

static void fill_random(double *A, int n) {
    for (int i = 0; i < n; i++) A[i] = (double)rand() / RAND_MAX - 0.5;
}
static void fill_random_s(float *A, int n) {
    for (int i = 0; i < n; i++) A[i] = (float)rand() / RAND_MAX - 0.5f;
}

static int test_dgemm_case(char ta, char tb, int m, int n, int k,
                           double alpha, double beta)
{
    int nrowa = (ta == 'N' || ta == 'n') ? m : k;
    int nrowb = (tb == 'N' || tb == 'n') ? k : n;
    int lda = nrowa + 2;
    int ldb = nrowb + 3;
    int ldc = m + 1;

    double *A = (double *)malloc(lda * k * sizeof(double));
    double *B = (double *)malloc(ldb * n * sizeof(double));
    double *C = (double *)malloc(ldc * n * sizeof(double));
    double *C_ref = (double *)malloc(ldc * n * sizeof(double));

    fill_random(A, lda * k);
    fill_random(B, ldb * n);
    fill_random(C, ldc * n);
    memcpy(C_ref, C, ldc * n * sizeof(double));

    naive_dgemm(ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C_ref, ldc);
    my_dgemm(ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

    int ret = compare_d(C_ref, C, m, n, ldc, EPS_D);

    free(A); free(B); free(C); free(C_ref);
    return ret;
}

static int test_sgemm_case(char ta, char tb, int m, int n, int k,
                           float alpha, float beta)
{
    int nrowa = (ta == 'N' || ta == 'n') ? m : k;
    int nrowb = (tb == 'N' || tb == 'n') ? k : n;
    int lda = nrowa + 2;
    int ldb = nrowb + 3;
    int ldc = m + 1;

    float *A = (float *)malloc(lda * k * sizeof(float));
    float *B = (float *)malloc(ldb * n * sizeof(float));
    float *C = (float *)malloc(ldc * n * sizeof(float));
    float *C_ref = (float *)malloc(ldc * n * sizeof(float));

    fill_random_s(A, lda * k);
    fill_random_s(B, ldb * n);
    fill_random_s(C, ldc * n);
    memcpy(C_ref, C, ldc * n * sizeof(float));

    naive_sgemm(ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C_ref, ldc);
    my_sgemm(ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

    int ret = compare_s(C_ref, C, m, n, ldc, EPS_S);

    free(A); free(B); free(C); free(C_ref);
    return ret;
}

int main(void)
{
    int pass = 0, fail = 0;

    printf("=== dgemm transpose combinations ===\n");
    char *trans[] = {"N", "T"};
    for (int ta = 0; ta < 2; ta++) {
        for (int tb = 0; tb < 2; tb++) {
            printf("  %s%s (m=16,n=16,k=16): ", trans[ta], trans[tb]);
            if (test_dgemm_case(trans[ta][0], trans[tb][0], 16, 16, 16, 1.0, 0.0) == 0) {
                printf("PASS\n"); pass++;
            } else { printf("FAIL\n"); fail++; }
        }
    }

    printf("\n=== dgemm edge cases ===\n");
    printf("  alpha=0, beta=0: ");
    if (test_dgemm_case('N', 'N', 8, 8, 8, 0.0, 0.0) == 0) { printf("PASS\n"); pass++; }
    else { printf("FAIL\n"); fail++; }

    printf("  alpha=0, beta=2: ");
    if (test_dgemm_case('N', 'N', 8, 8, 8, 0.0, 2.0) == 0) { printf("PASS\n"); pass++; }
    else { printf("FAIL\n"); fail++; }

    printf("  k=0: ");
    if (test_dgemm_case('N', 'N', 8, 8, 0, 1.0, 1.0) == 0) { printf("PASS\n"); pass++; }
    else { printf("FAIL\n"); fail++; }

    printf("  m=1: ");
    if (test_dgemm_case('N', 'N', 1, 16, 16, 1.0, 0.0) == 0) { printf("PASS\n"); pass++; }
    else { printf("FAIL\n"); fail++; }

    printf("  n=1: ");
    if (test_dgemm_case('N', 'N', 16, 1, 16, 1.0, 0.0) == 0) { printf("PASS\n"); pass++; }
    else { printf("FAIL\n"); fail++; }

    printf("\n=== dgemm non-aligned dimensions ===\n");
    int dims[][3] = {{5,7,11}, {3,3,3}, {17,13,19}, {100,100,100}};
    for (int d = 0; d < 4; d++) {
        printf("  m=%d,n=%d,k=%d: ", dims[d][0], dims[d][1], dims[d][2]);
        if (test_dgemm_case('N', 'N', dims[d][0], dims[d][1], dims[d][2], 1.0, 0.0) == 0) {
            printf("PASS\n"); pass++;
        } else { printf("FAIL\n"); fail++; }
    }

    printf("\n=== dgemm larger matrices ===\n");
    int sizes[] = {64, 128, 256};
    for (int s = 0; s < 3; s++) {
        printf("  %dx%dx%d: ", sizes[s], sizes[s], sizes[s]);
        if (test_dgemm_case('N', 'N', sizes[s], sizes[s], sizes[s], 1.0, 0.0) == 0) {
            printf("PASS\n"); pass++;
        } else { printf("FAIL\n"); fail++; }
    }

    printf("\n=== sgemm tests ===\n");
    printf("  NN (m=16,n=16,k=16): ");
    if (test_sgemm_case('N', 'N', 16, 16, 16, 1.0f, 0.0f) == 0) { printf("PASS\n"); pass++; }
    else { printf("FAIL\n"); fail++; }

    printf("  TT (m=16,n=16,k=16): ");
    if (test_sgemm_case('T', 'T', 16, 16, 16, 1.0f, 0.0f) == 0) { printf("PASS\n"); pass++; }
    else { printf("FAIL\n"); fail++; }

    printf("  Non-aligned (m=5,n=7,k=11): ");
    if (test_sgemm_case('N', 'N', 5, 7, 11, 1.0f, 0.0f) == 0) { printf("PASS\n"); pass++; }
    else { printf("FAIL\n"); fail++; }

    printf("  128x128x128: ");
    if (test_sgemm_case('N', 'N', 128, 128, 128, 1.0f, 0.0f) == 0) { printf("PASS\n"); pass++; }
    else { printf("FAIL\n"); fail++; }

    printf("\n=== Summary: %d passed, %d failed ===\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
