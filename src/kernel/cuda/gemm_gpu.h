#ifndef MYBLAS_GEMM_GPU_H
#define MYBLAS_GEMM_GPU_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gpu_gemm_handle_t gpu_gemm_handle_t;

int gpu_is_available(void);
int gpu_get_compute_capability(void);
size_t gpu_get_memory(void);

gpu_gemm_handle_t *gpu_gemm_handle_create(void);
void gpu_gemm_handle_destroy(gpu_gemm_handle_t *handle);

int gpu_dgemm(gpu_gemm_handle_t *handle,
              char transa, char transb,
              int m, int n, int k,
              double alpha, const double *A, int lda,
              const double *B, int ldb,
              double beta, double *C, int ldc);

int gpu_sgemm(gpu_gemm_handle_t *handle,
              char transa, char transb,
              int m, int n, int k,
              float alpha, const float *A, int lda,
              const float *B, int ldb,
              float beta, float *C, int ldc);

int gpu_dgemm_version(int version,
                      char transa, char transb,
                      int m, int n, int k,
                      double alpha, const double *A, int lda,
                      const double *B, int ldb,
                      double beta, double *C, int ldc);

int gpu_sgemm_version(int version,
                      char transa, char transb,
                      int m, int n, int k,
                      float alpha, const float *A, int lda,
                      const float *B, int ldb,
                      float beta, float *C, int ldc);

int gpu_dgemm_device_version(int version,
                             char transa, char transb,
                             int m, int n, int k,
                             double alpha, const double *dA, int lda,
                             const double *dB, int ldb,
                             double beta, double *dC, int ldc);

int gpu_sgemm_device_version(int version,
                             char transa, char transb,
                             int m, int n, int k,
                             float alpha, const float *dA, int lda,
                             const float *dB, int ldb,
                             float beta, float *dC, int ldc);

int gpu_dispatch_init(void);
void gpu_dispatch_cleanup(void);
int gpu_should_dispatch(int m, int n, int k, int precision);
gpu_gemm_handle_t *gpu_dispatch_handle(void);

#ifdef __cplusplus
}
#endif

#endif
