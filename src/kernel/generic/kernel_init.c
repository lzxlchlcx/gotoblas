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

/* DGEMM generic kernel table (fallback, no SIMD):
 *   kernel    — 4×4 微内核: C += α·A_packed·B_packed
 *   pack_a_nn — 打包 A (不转置): 按 NR×MR 连续格式重排
 *   pack_a_tn — 打包 A (已转置): 同 pack_a_nn，但输入 Aᵀ
 *   pack_b_nn — 打包 B (不转置): 按 NR 分块连续格式重排
 *   pack_b_tn — 打包 B (已转置): 同 pack_b_nn，但输入 Bᵀ
 *   beta      — C = β·C 缩放 */
gemm_kernel_table_t gemm_kernel_generic_double = {
    .kernel    = dgemm_kernel_generic,
    .pack_a_nn = dgemm_pack_a_nn,
    .pack_a_tn = dgemm_pack_a_tn,
    .pack_b_nn = dgemm_pack_b_nn,
    .pack_b_tn = dgemm_pack_b_tn,
    .beta      = dgemm_beta_generic,
};

/* SGEMM generic kernel table (single-precision fallback):
 *   作用同上，操作类型为 float */
sgemm_kernel_table_t gemm_kernel_generic_float = {
    .kernel    = sgemm_kernel_generic,
    .pack_a_nn = sgemm_pack_a_nn,
    .pack_a_tn = sgemm_pack_a_tn,
    .pack_b_nn = sgemm_pack_b_nn,
    .pack_b_tn = sgemm_pack_b_tn,
    .beta      = sgemm_beta_generic,
};
