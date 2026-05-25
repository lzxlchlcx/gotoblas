#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "myblas.h"
#include <cblas.h>
#include <openblas_config.h>

static double get_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void fill_random(double *A, int n) {
    for (int i = 0; i < n; i++) A[i] = (double)rand() / RAND_MAX - 0.5;
}

static double bench_myblas(int n, int niter)
{
    int m = n, k = n;
    int lda = m, ldb = k, ldc = m;

    double *A = (double *)malloc(lda * k * sizeof(double));
    double *B = (double *)malloc(ldb * n * sizeof(double));
    double *C = (double *)calloc(ldc * n, sizeof(double));

    fill_random(A, lda * k);
    fill_random(B, ldb * n);

    double start = get_time();
    for (int i = 0; i < niter; i++) {
        my_dgemm('N', 'N', m, n, k, 1.0, A, lda, B, ldb, 0.0, C, ldc);
    }
    double elapsed = get_time() - start;

    free(A); free(B); free(C);
    return (2.0 * m * n * k * niter) / (elapsed * 1e9);
}

static double bench_openblas(int n, int niter)
{
    int m = n, k = n;
    int lda = m, ldb = k, ldc = m;

    double *A = (double *)malloc(lda * k * sizeof(double));
    double *B = (double *)malloc(ldb * n * sizeof(double));
    double *C = (double *)calloc(ldc * n, sizeof(double));

    fill_random(A, lda * k);
    fill_random(B, ldb * n);

    double start = get_time();
    for (int i = 0; i < niter; i++) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, 1.0, A, lda, B, ldb, 0.0, C, ldc);
    }
    double elapsed = get_time() - start;

    free(A); free(B); free(C);
    return (2.0 * m * n * k * niter) / (elapsed * 1e9);
}

int main(void)
{
    int sizes[] = {64, 128, 256, 512, 1024, 2048};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("=== MyBLAS vs OpenBLAS Comparison ===\n");
    // TODO: Detect CPU model programmatically
    printf("CPU: Apple M5\n\n");

    for (int t = 1; t <= 4; t *= 4) {
        myblas_set_num_threads(t);

        openblas_set_num_threads(t);

        printf("--- %d thread(s) ---\n", t);
        printf("%10s  %12s  %12s  %6s\n",
               "Size", "MyBLAS", "OpenBLAS", "Speedup");            
        printf("%10s  %12s  %12s  %6s\n",
               "", "(GFLOPS)", "(GFLOPS)", "");

        for (int i = 0; i < nsizes; i++) {
            myblas_log_reset();
            int n = sizes[i];
            int niter = 5;
            if (n <= 128) niter = 20;
            else if (n <= 256) niter = 10;
            else if (n == 2048) niter = 3;

            double gflops_m = bench_myblas(n, niter);
            double gflops_o = bench_openblas(n, niter);
            double speedup = gflops_o / gflops_m;

            printf("%10d  %12.2f  %12.2f  %5.1fx\n",
                n, gflops_m, gflops_o, speedup);
                
                myblas_log_print();
        }
        printf("\n");
    }



    return 0;
}
