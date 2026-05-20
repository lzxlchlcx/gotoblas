#include <string.h>
#include "driver/gemm_internal.h"

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

    int js, ls, is;
    int min_j, min_l, min_i;

    for (js = 0; js < n; js += R) {
        min_j = n - js;
        if (min_j > R) min_j = R;

        for (ls = 0; ls < k; ls += Q) {
            min_l = k - ls;
            if (min_l > Q) min_l = Q;

            int min_jj;
            int jjs;
            for (jjs = js; jjs < js + min_j; jjs += min_jj) {
                min_jj = (js + min_j) - jjs;
                if (min_jj > NR) min_jj = NR;

                if (transb == 0)
                    pack_b(min_l, min_jj, &B[ls + jjs * ldb], ldb,
                           sb + min_l * (jjs - js));
                else
                    pack_b(min_l, min_jj, &B[jjs + ls * ldb], ldb,
                           sb + min_l * (jjs - js));

                for (is = 0; is < m; is += P) {
                    min_i = m - is;
                    if (min_i > P) min_i = P;

                    int min_ii;
                    int iis;
                    for (iis = is; iis < is + min_i; iis += min_ii) {
                        min_ii = (is + min_i) - iis;
                        if (min_ii > MR) min_ii = MR;

                        if (transa == 0)
                            pack_a(min_ii, min_l, &A[iis + ls * lda], lda, sa);
                        else
                            pack_a(min_ii, min_l, &A[ls + iis * lda], lda, sa);

                        kernels->kernel(min_ii, min_jj, min_l,
                                        alpha,
                                        sa,
                                        sb + min_l * (jjs - js),
                                        &C[iis + jjs * ldc], ldc);
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

    int P = cfg->P, Q = cfg->Q, R = cfg->R;
    int MR = cfg->MR, NR = cfg->NR;

    spack_func pack_a = (transa == 0) ? kernels->pack_a_nn : kernels->pack_a_tn;
    spack_func pack_b = (transb == 0) ? kernels->pack_b_nn : kernels->pack_b_tn;

    int js, ls, is;
    int min_j, min_l, min_i;

    for (js = 0; js < n; js += R) {
        min_j = n - js;
        if (min_j > R) min_j = R;

        for (ls = 0; ls < k; ls += Q) {
            min_l = k - ls;
            if (min_l > Q) min_l = Q;

            int min_jj;
            int jjs;
            for (jjs = js; jjs < js + min_j; jjs += min_jj) {
                min_jj = (js + min_j) - jjs;
                if (min_jj > NR) min_jj = NR;

                if (transb == 0)
                    pack_b(min_l, min_jj, &B[ls + jjs * ldb], ldb,
                           sb + min_l * (jjs - js));
                else
                    pack_b(min_l, min_jj, &B[jjs + ls * ldb], ldb,
                           sb + min_l * (jjs - js));

                for (is = 0; is < m; is += P) {
                    min_i = m - is;
                    if (min_i > P) min_i = P;

                    int min_ii;
                    int iis;
                    for (iis = is; iis < is + min_i; iis += min_ii) {
                        min_ii = (is + min_i) - iis;
                        if (min_ii > MR) min_ii = MR;

                        if (transa == 0)
                            pack_a(min_ii, min_l, &A[iis + ls * lda], lda, sa);
                        else
                            pack_a(min_ii, min_l, &A[ls + iis * lda], lda, sa);

                        kernels->kernel(min_ii, min_jj, min_l,
                                        alpha,
                                        sa,
                                        sb + min_l * (jjs - js),
                                        &C[iis + jjs * ldc], ldc);
                    }
                }
            }
        }
    }
}
