#ifndef MYBLAS_H
#define MYBLAS_H

#ifdef __cplusplus
extern "C" {
#endif

void my_dgemm(char transa, char transb,
              int m, int n, int k,
              double alpha, const double *A, int lda,
                            const double *B, int ldb,
              double beta,        double *C, int ldc);

void my_sgemm(char transa, char transb,
              int m, int n, int k,
              float alpha,  const float *A, int lda,
                            const float *B, int ldb,
              float beta,         float *C, int ldc);

void myblas_set_num_threads(int num_threads);
int  myblas_get_num_threads(void);

#ifdef __cplusplus
}
#endif

#endif
