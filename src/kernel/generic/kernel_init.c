#include "driver/gemm_internal.h"

extern int dgemm_kernel_generic(int m, int n, int k, double alpha,
                                const double *A, const double *B,
                                double *C, int ldc);
extern int dgemm_pack_a_nn(int m, int k, const double *A, int lda, double *A_packed);
extern int dgemm_pack_a_tn(int m, int k, const double *A, int lda, double *A_packed);
extern int dgemm_pack_b_nn(int k, int n, const double *B, int ldb, double *B_packed);
extern int dgemm_pack_b_tn(int k, int n, const double *B, int ldb, double *B_packed);
extern int dgemm_beta_generic(int m, int n, double beta, double *C, int ldc);

extern int sgemm_kernel_generic(int m, int n, int k, float alpha,
                                const float *A, const float *B,
                                float *C, int ldc);
extern int sgemm_pack_a_nn(int m, int k, const float *A, int lda, float *A_packed);
extern int sgemm_pack_a_tn(int m, int k, const float *A, int lda, float *A_packed);
extern int sgemm_pack_b_nn(int k, int n, const float *B, int ldb, float *B_packed);
extern int sgemm_pack_b_tn(int k, int n, const float *B, int ldb, float *B_packed);
extern int sgemm_beta_generic(int m, int n, float beta, float *C, int ldc);

gemm_kernel_table_t gemm_kernel_generic_double = {
    .kernel    = dgemm_kernel_generic,
    .pack_a_nn = dgemm_pack_a_nn,
    .pack_a_tn = dgemm_pack_a_tn,
    .pack_b_nn = dgemm_pack_b_nn,
    .pack_b_tn = dgemm_pack_b_tn,
    .beta      = dgemm_beta_generic,
};

sgemm_kernel_table_t gemm_kernel_generic_float = {
    .kernel    = sgemm_kernel_generic,
    .pack_a_nn = sgemm_pack_a_nn,
    .pack_a_tn = sgemm_pack_a_tn,
    .pack_b_nn = sgemm_pack_b_nn,
    .pack_b_tn = sgemm_pack_b_tn,
    .beta      = sgemm_beta_generic,
};
