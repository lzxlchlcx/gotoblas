#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "myblas.h"
#include "driver/gemm_internal.h"
#include "config/generic.h"

void my_sgemm(char transa, char transb,
              int m, int n, int k,
              float alpha,  const float *A, int lda,
                            const float *B, int ldb,
              float beta,         float *C, int ldc)
{
    int ta, tb;
    int nrowa, nrowb;

    if (transa == 'N' || transa == 'n')      ta = 0;
    else if (transa == 'T' || transa == 't')  ta = 1;
    else if (transa == 'C' || transa == 'c')  ta = 1;
    else {
        fprintf(stderr, "my_sgemm: invalid transa='%c'\n", transa);
        return;
    }

    if (transb == 'N' || transb == 'n')      tb = 0;
    else if (transb == 'T' || transb == 't')  tb = 1;
    else if (transb == 'C' || transb == 'c')  tb = 1;
    else {
        fprintf(stderr, "my_sgemm: invalid transb='%c'\n", transb);
        return;
    }

    nrowa = ta ? k : m;
    nrowb = tb ? n : k;

    if (m < 0) { fprintf(stderr, "my_sgemm: m=%d < 0\n", m); return; }
    if (n < 0) { fprintf(stderr, "my_sgemm: n=%d < 0\n", n); return; }
    if (k < 0) { fprintf(stderr, "my_sgemm: k=%d < 0\n", k); return; }
    if (lda < nrowa) { fprintf(stderr, "my_sgemm: lda=%d < %d\n", lda, nrowa); return; }
    if (ldb < nrowb) { fprintf(stderr, "my_sgemm: ldb=%d < %d\n", ldb, nrowb); return; }
    if (ldc < m)     { fprintf(stderr, "my_sgemm: ldc=%d < %d\n", ldc, m); return; }

    if (m == 0 || n == 0) return;

    if (k == 0 || alpha == 0.0f) {
        if (beta == 0.0f) {
            for (int j = 0; j < n; j++)
                for (int i = 0; i < m; i++)
                    C[i + j * ldc] = 0.0f;
        } else if (beta != 1.0f) {
            for (int j = 0; j < n; j++)
                for (int i = 0; i < m; i++)
                    C[i + j * ldc] *= beta;
        }
        return;
    }

    gemm_config_t cfg;
    gemm_config_generic_float(&cfg);

    size_t sa_size = (size_t)cfg.P * cfg.Q * sizeof(float) + 4096;
    size_t sb_size = (size_t)cfg.Q * cfg.R * sizeof(float) + 4096;
    float *sa = (float *)malloc(sa_size + cfg.offset_a);
    float *sb = (float *)malloc(sb_size + cfg.offset_b);
    if (!sa || !sb) {
        fprintf(stderr, "my_sgemm: failed to allocate packing buffers\n");
        free(sa); free(sb);
        return;
    }

    gemm_arg_t arg;
    memset(&arg, 0, sizeof(arg));
    arg.m = m; arg.n = n; arg.k = k;
    arg.A = A; arg.B = B; arg.C = C;
    arg.lda = lda; arg.ldb = ldb; arg.ldc = ldc;
    arg.alpha_s = alpha; arg.beta_s = beta;
    arg.transa = ta; arg.transb = tb;
    arg.nthreads = myblas_get_num_threads();

    if (beta == 0.0f) {
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                C[i + j * ldc] = 0.0f;
    } else if (beta != 1.0f) {
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                C[i + j * ldc] *= beta;
    }

    if (arg.nthreads > 1 && (double)m * n * k > 65536.0) {
        gemm_parallel_float(&arg, &cfg, &gemm_kernel_generic_float);
    } else {
        gemm_driver_float(&arg, &cfg, &gemm_kernel_generic_float, sa, sb);
    }

    free(sa);
    free(sb);
}
