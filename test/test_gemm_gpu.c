#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "myblas.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "kernel/cuda/gemm_gpu.h"
#endif

#define EPS_D 1e-9
#define EPS_S 5e-4f
#define EPS_TF32 2e-2f

static void fill_random_d(double *A, int n)
{
    for (int i = 0; i < n; i++) A[i] = (double)rand() / RAND_MAX - 0.5;
}

static void fill_random_s(float *A, int n)
{
    for (int i = 0; i < n; i++) A[i] = (float)rand() / RAND_MAX - 0.5f;
}

static void ref_dgemm(char ta, char tb, int m, int n, int k,
                      double alpha, const double *A, int lda,
                      const double *B, int ldb, double beta, double *C, int ldc)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            double sum = 0.0;
            for (int p = 0; p < k; p++) {
                double a = (ta == 'N' || ta == 'n') ? A[i + p * lda] : A[p + i * lda];
                double b = (tb == 'N' || tb == 'n') ? B[p + j * ldb] : B[j + p * ldb];
                sum += a * b;
            }
            C[i + j * ldc] = beta * C[i + j * ldc] + alpha * sum;
        }
    }
}

static void ref_sgemm(char ta, char tb, int m, int n, int k,
                      float alpha, const float *A, int lda,
                      const float *B, int ldb, float beta, float *C, int ldc)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                float a = (ta == 'N' || ta == 'n') ? A[i + p * lda] : A[p + i * lda];
                float b = (tb == 'N' || tb == 'n') ? B[p + j * ldb] : B[j + p * ldb];
                sum += a * b;
            }
            C[i + j * ldc] = beta * C[i + j * ldc] + alpha * sum;
        }
    }
}

static int compare_d(const double *ref, const double *got, int m, int n, int ldc)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            double r = ref[i + j * ldc];
            double diff = fabs(r - got[i + j * ldc]);
            double denom = fabs(r) > 1e-14 ? fabs(r) : 1.0;
            if (diff > EPS_D && diff / denom > EPS_D) {
                printf("FAIL at (%d,%d): ref=%e got=%e rel=%e\n", i, j, r, got[i + j * ldc], diff / denom);
                return -1;
            }
        }
    }
    return 0;
}

static int compare_s(const float *ref, const float *got, int m, int n, int ldc, float eps)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            float r = ref[i + j * ldc];
            float diff = fabsf(r - got[i + j * ldc]);
            float denom = fabsf(r) > 1e-6f ? fabsf(r) : 1.0f;
            if (diff > eps && diff / denom > eps) {
                printf("FAIL at (%d,%d): ref=%e got=%e rel=%e\n", i, j, r, got[i + j * ldc], diff / denom);
                return -1;
            }
        }
    }
    return 0;
}

#ifdef USE_CUDA
static int test_dgemm_case(int version, char ta, char tb, int m, int n, int k)
{
    int lda = ((ta == 'N' || ta == 'n') ? m : k) + 2;
    int ldb = ((tb == 'N' || tb == 'n') ? k : n) + 3;
    int ldc = m + 1;
    double *A = (double *)malloc((size_t)lda * ((ta == 'N' || ta == 'n') ? k : m) * sizeof(double));
    double *B = (double *)malloc((size_t)ldb * ((tb == 'N' || tb == 'n') ? n : k) * sizeof(double));
    double *C = (double *)malloc((size_t)ldc * n * sizeof(double));
    double *R = (double *)malloc((size_t)ldc * n * sizeof(double));
    fill_random_d(A, lda * ((ta == 'N' || ta == 'n') ? k : m));
    fill_random_d(B, ldb * ((tb == 'N' || tb == 'n') ? n : k));
    fill_random_d(C, ldc * n);
    memcpy(R, C, (size_t)ldc * n * sizeof(double));
    ref_dgemm(ta, tb, m, n, k, 1.25, A, lda, B, ldb, -0.5, R, ldc);
    int ret = gpu_dgemm_version(version, ta, tb, m, n, k, 1.25, A, lda, B, ldb, -0.5, C, ldc);
    if (ret == 0) ret = compare_d(R, C, m, n, ldc);
    free(A); free(B); free(C); free(R);
    return ret;
}

static int test_sgemm_case(int version, char ta, char tb, int m, int n, int k)
{
    int lda = ((ta == 'N' || ta == 'n') ? m : k) + 2;
    int ldb = ((tb == 'N' || tb == 'n') ? k : n) + 3;
    int ldc = m + 1;
    float *A = (float *)malloc((size_t)lda * ((ta == 'N' || ta == 'n') ? k : m) * sizeof(float));
    float *B = (float *)malloc((size_t)ldb * ((tb == 'N' || tb == 'n') ? n : k) * sizeof(float));
    float *C = (float *)malloc((size_t)ldc * n * sizeof(float));
    float *R = (float *)malloc((size_t)ldc * n * sizeof(float));
    fill_random_s(A, lda * ((ta == 'N' || ta == 'n') ? k : m));
    fill_random_s(B, ldb * ((tb == 'N' || tb == 'n') ? n : k));
    fill_random_s(C, ldc * n);
    memcpy(R, C, (size_t)ldc * n * sizeof(float));
    ref_sgemm(ta, tb, m, n, k, 1.25f, A, lda, B, ldb, -0.5f, R, ldc);
    int ret = gpu_sgemm_version(version, ta, tb, m, n, k, 1.25f, A, lda, B, ldb, -0.5f, C, ldc);
    if (ret == 0) ret = compare_s(R, C, m, n, ldc, version >= 4 ? EPS_TF32 : EPS_S);
    free(A); free(B); free(C); free(R);
    return ret;
}

static int test_dgemm_device_case(int version, int m, int n, int k)
{
    int lda = m + 2;
    int ldb = k + 3;
    int ldc = m + 1;
    size_t asize = (size_t)lda * k * sizeof(double);
    size_t bsize = (size_t)ldb * n * sizeof(double);
    size_t csize = (size_t)ldc * n * sizeof(double);
    double *A = (double *)malloc(asize);
    double *B = (double *)malloc(bsize);
    double *C = (double *)malloc(csize);
    double *R = (double *)malloc(csize);
    double *dA = 0, *dB = 0, *dC = 0;
    int ret = -1;

    fill_random_d(A, lda * k);
    fill_random_d(B, ldb * n);
    fill_random_d(C, ldc * n);
    memcpy(R, C, csize);
    ref_dgemm('N', 'N', m, n, k, 1.25, A, lda, B, ldb, -0.5, R, ldc);

    if (cudaMalloc((void **)&dA, asize) != cudaSuccess) goto cleanup;
    if (cudaMalloc((void **)&dB, bsize) != cudaSuccess) goto cleanup;
    if (cudaMalloc((void **)&dC, csize) != cudaSuccess) goto cleanup;
    if (cudaMemcpy(dA, A, asize, cudaMemcpyHostToDevice) != cudaSuccess) goto cleanup;
    if (cudaMemcpy(dB, B, bsize, cudaMemcpyHostToDevice) != cudaSuccess) goto cleanup;
    if (cudaMemcpy(dC, C, csize, cudaMemcpyHostToDevice) != cudaSuccess) goto cleanup;
    if (gpu_dgemm_device_version(version, 'N', 'N', m, n, k, 1.25, dA, lda, dB, ldb, -0.5, dC, ldc) != 0) goto cleanup;
    if (cudaMemcpy(C, dC, csize, cudaMemcpyDeviceToHost) != cudaSuccess) goto cleanup;
    ret = compare_d(R, C, m, n, ldc);

cleanup:
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(A); free(B); free(C); free(R);
    return ret;
}

static int test_sgemm_device_case(int version, int m, int n, int k)
{
    int lda = m + 2;
    int ldb = k + 3;
    int ldc = m + 1;
    size_t asize = (size_t)lda * k * sizeof(float);
    size_t bsize = (size_t)ldb * n * sizeof(float);
    size_t csize = (size_t)ldc * n * sizeof(float);
    float *A = (float *)malloc(asize);
    float *B = (float *)malloc(bsize);
    float *C = (float *)malloc(csize);
    float *R = (float *)malloc(csize);
    float *dA = 0, *dB = 0, *dC = 0;
    int ret = -1;

    fill_random_s(A, lda * k);
    fill_random_s(B, ldb * n);
    fill_random_s(C, ldc * n);
    memcpy(R, C, csize);
    ref_sgemm('N', 'N', m, n, k, 1.25f, A, lda, B, ldb, -0.5f, R, ldc);

    if (cudaMalloc((void **)&dA, asize) != cudaSuccess) goto cleanup;
    if (cudaMalloc((void **)&dB, bsize) != cudaSuccess) goto cleanup;
    if (cudaMalloc((void **)&dC, csize) != cudaSuccess) goto cleanup;
    if (cudaMemcpy(dA, A, asize, cudaMemcpyHostToDevice) != cudaSuccess) goto cleanup;
    if (cudaMemcpy(dB, B, bsize, cudaMemcpyHostToDevice) != cudaSuccess) goto cleanup;
    if (cudaMemcpy(dC, C, csize, cudaMemcpyHostToDevice) != cudaSuccess) goto cleanup;
    if (gpu_sgemm_device_version(version, 'N', 'N', m, n, k, 1.25f, dA, lda, dB, ldb, -0.5f, dC, ldc) != 0) goto cleanup;
    if (cudaMemcpy(C, dC, csize, cudaMemcpyDeviceToHost) != cudaSuccess) goto cleanup;
    ret = compare_s(R, C, m, n, ldc, version >= 4 ? EPS_TF32 : EPS_S);

cleanup:
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(A); free(B); free(C); free(R);
    return ret;
}
#endif

int main(void)
{
#ifndef USE_CUDA
    printf("CUDA support is not enabled in this build.\n");
    return 0;
#else
    if (!gpu_is_available()) {
        printf("No CUDA GPU available.\n");
        return 0;
    }
    printf("CUDA GPU detected: SM %d, %.1f GiB\n", gpu_get_compute_capability(), gpu_get_memory() / 1073741824.0);

    int pass = 0, fail = 0;
    char trans[2] = {'N', 'T'};
    int dims[][3] = {{5, 7, 11}, {17, 13, 19}, {64, 64, 64}};

    for (int v = 0; v <= 4; v++) {
        for (int ta = 0; ta < 2; ta++) {
            for (int tb = 0; tb < 2; tb++) {
                for (int d = 0; d < 3; d++) {
                    if (test_dgemm_case(v, trans[ta], trans[tb], dims[d][0], dims[d][1], dims[d][2]) == 0) pass++;
                    else fail++;
                    if (test_sgemm_case(v, trans[ta], trans[tb], dims[d][0], dims[d][1], dims[d][2]) == 0) pass++;
                    else fail++;
                }
            }
        }
    }

    if (test_sgemm_device_case(4, 64, 64, 64) == 0) pass++;
    else fail++;
    if (test_sgemm_device_case(4, 17, 13, 19) == 0) pass++;
    else fail++;
    if (test_dgemm_device_case(3, 64, 64, 64) == 0) pass++;
    else fail++;

    printf("GPU GEMM tests: %d passed, %d failed\n", pass, fail);
    return fail ? 1 : 0;
#endif
}
