#include <string.h>
#include "driver/gemm_internal.h"
#include "util/myblas_log.h"

void gemm_driver_double(const gemm_arg_t *arg, const gemm_config_t *cfg,
                        const gemm_kernel_table_t *kernels,
                        double *sa, double *sb)
{
    int m = arg->m, n = arg->n, k = arg->k;
    int lda = arg->lda, ldb = arg->ldb, ldc = arg->ldc;
    double alpha = arg->alpha_d;
    const double *A = (const double *)arg->A;
    const double *B = (const double *)arg->B;
    double *C = (double *)arg->C;
    int transa = arg->transa, transb = arg->transb;

    int P = cfg->P, Q = cfg->Q, R = cfg->R;
    int MR = cfg->MR, NR = cfg->NR;

    pack_func pack_a = (transa == 0) ? kernels->pack_a_nn : kernels->pack_a_tn;
    pack_func pack_b = (transb == 0) ? kernels->pack_b_nn : kernels->pack_b_tn;

    int col0, k0, row0;
    int col_rem, k_rem, row_rem;

    for (col0 = 0; col0 < n; col0 += R) {
        col_rem = n - col0;
        if (col_rem > R) col_rem = R;

        for (k0 = 0; k0 < k; k0 += Q) {
            k_rem = k - k0;
            if (k_rem > Q) k_rem = Q;

            int col1_rem;
            int col1;
            for (col1 = col0; col1 < col0 + col_rem; col1 += col1_rem) {
                col1_rem = (col0 + col_rem) - col1;
                if (col1_rem > NR) col1_rem = NR;

                MYBLAS_LOG_TIMER_START(_log_pack_b);
                if (transb == 0)
                    pack_b(k_rem, col1_rem, &B[k0 + col1 * ldb], ldb,
                           sb + k_rem * (col1 - col0));
                else
                    pack_b(k_rem, col1_rem, &B[col1 + k0 * ldb], ldb,
                           sb + k_rem * (col1 - col0));
                MYBLAS_LOG_TIMER_END(_log_pack_b, "pack_b");

                for (row0 = 0; row0 < m; row0 += P) {
                    row_rem = m - row0;
                    if (row_rem > P) row_rem = P;

                    int row1_rem;
                    int row1;
                    for (row1 = row0; row1 < row0 + row_rem; row1 += row1_rem) {
                        row1_rem = (row0 + row_rem) - row1;
                        if (row1_rem > MR) row1_rem = MR;

                        MYBLAS_LOG_TIMER_START(_log_pack_a);
                        if (transa == 0)
                            pack_a(row1_rem, k_rem, &A[row1 + k0 * lda], lda, sa);
                        else
                            pack_a(row1_rem, k_rem, &A[k0 + row1 * lda], lda, sa);
                        MYBLAS_LOG_TIMER_END(_log_pack_a, "pack_a");

                        MYBLAS_LOG_TIMER_START(_log_kernel);
                        kernels->kernel(row1_rem, col1_rem, k_rem,
                                        alpha,
                                        sa,
                                        sb + k_rem * (col1 - col0),
                                        &C[row1 + col1 * ldc], ldc);
                        MYBLAS_LOG_TIMER_END(_log_kernel, "kernel");
                    }
                }
            }
        }
    }
}

void gemm_driver_float(const gemm_arg_t *arg, const gemm_config_t *cfg,
                       const sgemm_kernel_table_t *kernels,
                       float *sa, float *sb)
{
    int m = arg->m, n = arg->n, k = arg->k;
    int lda = arg->lda, ldb = arg->ldb, ldc = arg->ldc;
    float alpha = arg->alpha_s;
    const float *A = (const float *)arg->A;
    const float *B = (const float *)arg->B;
    float *C = (float *)arg->C;
    int transa = arg->transa, transb = arg->transb;

    int P = cfg->P, Q = cfg->Q, R = cfg->R;   // 外层块大小: M, K, N 维度
    int MR = cfg->MR, NR = cfg->NR;           // 微块大小: M, N 维度

    spack_func pack_a = (transa == 0) ? kernels->pack_a_nn : kernels->pack_a_tn;
    spack_func pack_b = (transb == 0) ? kernels->pack_b_nn : kernels->pack_b_tn;

    int col0, k0, row0;       // 外层块起始索引: N, K, M 维度
    int col_rem, k_rem, row_rem; // 当前块末尾剩余量

    for (col0 = 0; col0 < n; col0 += R) {
        col_rem = n - col0;
        if (col_rem > R) col_rem = R;

        for (k0 = 0; k0 < k; k0 += Q) {
            k_rem = k - k0;
            if (k_rem > Q) k_rem = Q;

            int col1_rem;
            int col1;
            for (col1 = col0; col1 < col0 + col_rem; col1 += col1_rem) {
                col1_rem = (col0 + col_rem) - col1;
                if (col1_rem > NR) col1_rem = NR;

                if (transb == 0)
                    pack_b(k_rem, col1_rem, &B[k0 + col1 * ldb], ldb,
                           sb + k_rem * (col1 - col0));
                else
                    pack_b(k_rem, col1_rem, &B[col1 + k0 * ldb], ldb,
                           sb + k_rem * (col1 - col0));

                for (row0 = 0; row0 < m; row0 += P) {
                    row_rem = m - row0;
                    if (row_rem > P) row_rem = P;

                    int row1_rem;
                    int row1;
                    for (row1 = row0; row1 < row0 + row_rem; row1 += row1_rem) {
                        row1_rem = (row0 + row_rem) - row1;
                        if (row1_rem > MR) row1_rem = MR;

                        if (transa == 0)
                            pack_a(row1_rem, k_rem, &A[row1 + k0 * lda], lda, sa);
                        else
                            pack_a(row1_rem, k_rem, &A[k0 + row1 * lda], lda, sa);

                        kernels->kernel(row1_rem, col1_rem, k_rem,
                                        alpha,
                                        sa,
                                        sb + k_rem * (col1 - col0),
                                        &C[row1 + col1 * ldc], ldc);
                    }
                }
            }
        }
    }
}
