#ifndef MYBLAS_GEMM_INTERNAL_H
#define MYBLAS_GEMM_INTERNAL_H

#include <stddef.h>

typedef int (*gemm_kernel_func)(int m, int n, int k,
    double alpha, const double *A, const double *B,
    double *C, int ldc);

typedef int (*sgemm_kernel_func)(int m, int n, int k,
    float alpha, const float *A, const float *B,
    float *C, int ldc);

typedef int (*pack_func)(int m, int n, const double *A, int lda, double *A_packed);
typedef int (*spack_func)(int m, int n, const float *A, int lda, float *A_packed);

typedef int (*gemm_beta_func)(int m, int n, double beta, double *C, int ldc);
typedef int (*sgemm_beta_func)(int m, int n, float beta, float *C, int ldc);

typedef struct {
    int m, n, k;
    const void *A, *B;
    void *C;
    int lda, ldb, ldc;
    double alpha_d, beta_d;
    float  alpha_s, beta_s;
    int transa, transb;
    int nthreads;
    int precision;
} gemm_arg_t;

typedef struct {
    int P;
    int Q;
    int R;
    int MR;
    int NR;
    int dtb_entries;
    size_t offset_a;
    size_t offset_b;
} gemm_config_t;

typedef struct {
    gemm_kernel_func   kernel;
    pack_func          pack_a_nn;
    pack_func          pack_a_tn;
    pack_func          pack_b_nn;
    pack_func          pack_b_tn;
    gemm_beta_func     beta;
} gemm_kernel_table_t;

typedef struct {
    sgemm_kernel_func  kernel;
    spack_func         pack_a_nn;
    spack_func         pack_a_tn;
    spack_func         pack_b_nn;
    spack_func         pack_b_tn;
    sgemm_beta_func    beta;
} sgemm_kernel_table_t;

extern gemm_kernel_table_t  gemm_kernel_generic_double;
extern sgemm_kernel_table_t gemm_kernel_generic_float;

void gemm_driver_double(const gemm_arg_t *arg, const gemm_config_t *cfg,
                        const gemm_kernel_table_t *kernels,
                        double *sa, double *sb);

void gemm_driver_float(const gemm_arg_t *arg, const gemm_config_t *cfg,
                       const sgemm_kernel_table_t *kernels,
                       float *sa, float *sb);

void gemm_parallel_double(const gemm_arg_t *arg, const gemm_config_t *cfg,
                          const gemm_kernel_table_t *kernels);

void gemm_parallel_float(const gemm_arg_t *arg, const gemm_config_t *cfg,
                         const sgemm_kernel_table_t *kernels);

#endif
