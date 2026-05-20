#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "myblas.h"

static double get_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void fill_random(double *A, int n) {
    for (int i = 0; i < n; i++) A[i] = (double)rand() / RAND_MAX - 0.5;
}

static void bench_dgemm(int n)
{
    int m = n, k = n;
    int lda = m, ldb = k, ldc = m;

    double *A = (double *)malloc(lda * k * sizeof(double));
    double *B = (double *)malloc(ldb * n * sizeof(double));
    double *C = (double *)malloc(ldc * n * sizeof(double));

    fill_random(A, lda * k);
    fill_random(B, ldb * n);
    memset(C, 0, ldc * n * sizeof(double));

    my_dgemm('N', 'N', m, n, k, 1.0, A, lda, B, ldb, 0.0, C, ldc);

    int niter = 5;
    if (n <= 128) niter = 20;
    else if (n <= 256) niter = 10;

    double start = get_time();
    for (int i = 0; i < niter; i++) {
        my_dgemm('N', 'N', m, n, k, 1.0, A, lda, B, ldb, 0.0, C, ldc);
    }
    double elapsed = get_time() - start;

    double gflops = (2.0 * m * n * k * niter) / (elapsed * 1e9);
    printf("  dgemm %4dx%4dx%4d: %6.2f GFLOPS (%d iterations, %.3fs)\n",
           m, n, k, gflops, niter, elapsed);

    free(A); free(B); free(C);
}

int main(void)
{
    printf("=== GEMM Benchmark ===\n\n");

    int sizes[] = {64, 128, 256, 512, 1024};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < nsizes; i++) {
        bench_dgemm(sizes[i]);
    }

    printf("\n");
    return 0;
}
