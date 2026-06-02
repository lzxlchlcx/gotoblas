#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "myblas.h"
#include "driver/gemm_internal.h"
#include "config/generic.h"
#include "util/myblas_log.h"

#ifdef USE_CUDA
#include "kernel/cuda/gemm_gpu.h"
#endif

#ifdef __AVX2__
#include "config/haswell.h"
extern int cpu_supports_avx2(void);
#endif

void my_dgemm(char transa, char transb,
              int m, int n, int k,
              double alpha, const double *A, int lda,
                            const double *B, int ldb,
              double beta,        double *C, int ldc)
{
    int ta, tb;
    int nrowa, nrowb;

    if (transa == 'N' || transa == 'n')      ta = 0;
    else if (transa == 'T' || transa == 't')  ta = 1;
    else if (transa == 'C' || transa == 'c')  ta = 1;
    else {
        fprintf(stderr, "my_dgemm: invalid transa='%c'\n", transa);
        return;
    }

    if (transb == 'N' || transb == 'n')      tb = 0;
    else if (transb == 'T' || transb == 't')  tb = 1;
    else if (transb == 'C' || transb == 'c')  tb = 1;
    else {
        fprintf(stderr, "my_dgemm: invalid transb='%c'\n", transb);
        return;
    }

    nrowa = ta ? k : m;
    nrowb = tb ? n : k;

    if (m < 0) { fprintf(stderr, "my_dgemm: m=%d < 0\n", m); return; }
    if (n < 0) { fprintf(stderr, "my_dgemm: n=%d < 0\n", n); return; }
    if (k < 0) { fprintf(stderr, "my_dgemm: k=%d < 0\n", k); return; }
    if (lda < nrowa) { fprintf(stderr, "my_dgemm: lda=%d < %d\n", lda, nrowa); return; }
    if (ldb < nrowb) { fprintf(stderr, "my_dgemm: ldb=%d < %d\n", ldb, nrowb); return; }
    if (ldc < m)     { fprintf(stderr, "my_dgemm: ldc=%d < %d\n", ldc, m); return; }

    if (m == 0 || n == 0) return;

    if (k == 0 || alpha == 0.0) {
        if (beta == 0.0) {
            for (int j = 0; j < n; j++)
                for (int i = 0; i < m; i++)
                    C[i + j * ldc] = 0.0;
        } else if (beta != 1.0) {
            for (int j = 0; j < n; j++)
                for (int i = 0; i < m; i++)
                    C[i + j * ldc] *= beta;
        }
        return;
    }

    MYBLAS_LOG_TIMER_START(log_t0);

#ifdef USE_CUDA
    if (gpu_should_dispatch(m, n, k, 0) && gpu_dispatch_init() == 0) {
        if (gpu_dgemm(gpu_dispatch_handle(), transa, transb, m, n, k,
                      alpha, A, lda, B, ldb, beta, C, ldc) == 0) {
            MYBLAS_LOG_RECORD_CALL_ELAPSED(m, n, k, ta, tb, log_t0);
            return;
        }
    }
#endif

    gemm_config_t cfg;
    const gemm_kernel_table_t *kernels;

#ifdef __AVX2__
    static int avx2_checked = 0;
    static int has_avx2 = 0;
    if (!avx2_checked) {
        has_avx2 = cpu_supports_avx2();
        avx2_checked = 1;
    }
    if (has_avx2) {
        gemm_config_avx2_double(&cfg);
        kernels = &gemm_kernel_avx2_double;
    } else {
        gemm_config_generic_double(&cfg);
        kernels = &gemm_kernel_generic_double;
    }
#else
    gemm_config_generic_double(&cfg);
    kernels = &gemm_kernel_generic_double;
#endif

    size_t sa_size = (size_t)cfg.P * cfg.Q * sizeof(double) + 4096;
    size_t sb_size = (size_t)cfg.Q * cfg.R * sizeof(double) + 4096;
    double *sa = (double *)malloc(sa_size + cfg.offset_a);
    double *sb = (double *)malloc(sb_size + cfg.offset_b);
    if (!sa || !sb) {
        fprintf(stderr, "my_dgemm: failed to allocate packing buffers\n");
        free(sa); free(sb);
        return;
    }

    gemm_arg_t arg;
    memset(&arg, 0, sizeof(arg));
    arg.m = m; arg.n = n; arg.k = k;
    arg.A = A; arg.B = B; arg.C = C;
    arg.lda = lda; arg.ldb = ldb; arg.ldc = ldc;
    arg.alpha_d = alpha; arg.beta_d = beta;
    arg.transa = ta; arg.transb = tb;
    arg.nthreads = myblas_get_num_threads();

    if (beta == 0.0) {
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                C[i + j * ldc] = 0.0;
    } else if (beta != 1.0) {
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                C[i + j * ldc] *= beta;
    }

    if (arg.nthreads > 1 && (double)m * n * k > 65536.0) {
        gemm_parallel_double(&arg, &cfg, kernels);
    } else {
        gemm_driver_double(&arg, &cfg, kernels, sa, sb);
    }

    MYBLAS_LOG_RECORD_CALL_ELAPSED(m, n, k, ta, tb, log_t0);

    free(sa);
    free(sb);
}
